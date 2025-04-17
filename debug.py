import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import random
from collections import deque
from env import HighwayEnv, convert_highd_sample_to_gail_expert
import numpy as np
import pandas as pd
from tqdm import tqdm, trange


class HighwayEnvBuffer:
    def __init__(self, h_dim, v_dim, f_dim, act_dim, m_dim=10, b_dim=2, size=800, gamma=0.99, lam=0.95, device='cuda'):
        self.gamma, self.lam = gamma, lam
        self.kn_buf = torch.zeros((size, h_dim, v_dim, f_dim), device=device)
        self.lm_buf = torch.zeros((size, m_dim), device=device)
        self.bd_buf = torch.zeros((size, b_dim), device=device)
        self.mk_buf = torch.zeros((size, v_dim), device=device)
        self.act_buf = torch.zeros((size, v_dim, act_dim), device=device)
        self.logp_buf = torch.zeros((size, v_dim), device=device)
        self.rew_buf = torch.zeros((size, v_dim), device=device)
        self.val_buf = torch.zeros((size, v_dim), device=device)
        self.adv_buf = torch.zeros((size, v_dim), device=device)
        self.ret_buf = torch.zeros((size, v_dim), device=device)
        self.v_dim = v_dim
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.device = device

    def store(self, kn, lm, bd, mk, act, logp, rew, val):
        assert self.ptr < self.max_size
        self.kn_buf[self.ptr]  = kn
        self.lm_buf[self.ptr]  = lm
        self.bd_buf[self.ptr]  = bd
        self.mk_buf[self.ptr]   = mk
        self.act_buf[self.ptr]  = act.detach()
        self.logp_buf[self.ptr] = logp.detach()
        self.rew_buf[self.ptr]  = rew
        self.val_buf[self.ptr]  = val.detach()
        self.ptr += 1

    def finish_path(self, last_val):
        """
        last_val: (M,)  value estimates for step after the end
        """
        slice_ = slice(self.path_start_idx, self.ptr)
        last_val.to(self.device)
        # append last_val to compute deltas
        rews = torch.cat([self.rew_buf[slice_], last_val], dim=0)  # (L+1, M)
        vals = torch.cat([self.val_buf[slice_], last_val], dim=0)  # (L+1, M)
        # GAE deltas
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]                  # (L, M)

        # compute advantage
        adv = torch.zeros_like(deltas, device=self.device)
        gae = torch.zeros(self.v_dim, device=self.device)
        for t in reversed(range(deltas.shape[0])):
            gae = deltas[t] + self.gamma * self.lam * gae
            adv[t] = gae
        self.adv_buf[slice_] = adv

        # rewards-to-go
        ret = torch.zeros_like(rews)
        ret[-1] = last_val
        for t in reversed(range(rews.shape[0]-1)):
            ret[t] = rews[t] + self.gamma * ret[t+1]
        self.ret_buf[slice_] = ret[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer full
        # normalize advantages
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std() + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(kn=self.kn_buf,
                    lm=self.lm_buf,
                    bd=self.bd_buf,
                    mk=self.mk_buf,
                    act=self.act_buf,
                    logp=self.logp_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf.detach())
        # reset pointer
        self.ptr = 0
        self.path_start_idx = 0
        return {k: v for k,v in data.items()}
    
def compute_seq_lengths(time_dep):
    """
    Given time_dep of shape (B, T, F), return a tensor of sequence lengths,
    where a timestep is considered valid if not all features are NaN.
    """
    # valid if not all F values are NaN:
    valid_mask = ~torch.all(torch.isnan(time_dep), dim=-1)  # shape (B, T)
    seq_lengths = valid_mask.sum(dim=1)  # (B,)
    seq_lengths[seq_lengths == 0] = 1  # avoid zeros
    return seq_lengths

#########################################
# Actor Network (Policy)
#########################################
class PPOActor(nn.Module):
    def __init__(
        self,
        h_dim, v_dim, f_dim,
        lstm_hidden=64,
        global_dim=12,
        set_dim=32,                   # <— new: dimension of set encoder
        combined_hidden=64,
        output_size=2,
        device='cuda'
    ):
        super().__init__()
        self.T, self.M, self.F = h_dim, v_dim, f_dim
        self.lstm_hidden = lstm_hidden

        # 1) set encoder: φ embeds each vehicle repr into D_set
        self.set_dim = set_dim
        self.phi = nn.Sequential(
            nn.Linear(lstm_hidden, 2*set_dim),
            nn.ReLU(),
            nn.Linear(2*set_dim, set_dim),
        )

        # 2) per‐vehicle LSTM
        self.lstm = nn.LSTM(input_size=self.F, hidden_size=lstm_hidden,
                            batch_first=True)

        # 3) global info
        self.global_fc      = nn.Linear(10, global_dim)
        self.boundary_fc    = nn.Linear(2,  global_dim)
        self.global_combine = nn.Linear(2*global_dim, global_dim)

        # 4) final head now takes [veh_repr, global, set_repr]
        in_dim = lstm_hidden + global_dim + set_dim
        self.combine_fc = nn.Sequential(
            nn.Linear(in_dim, combined_hidden),
            nn.ReLU(),
            nn.Linear(combined_hidden, output_size),
        )

        # 5) learnable log‐std
        self.log_std = nn.Parameter(torch.zeros(1, 1, output_size))
        self.device  = device
        self.to(device)

    def forward(self, time_dep, lane_markers, boundaries):
        if time_dep.dim() == 3:
            time_dep = time_dep.unsqueeze(0) 
        if lane_markers.dim() == 1:
            lane_markers = lane_markers.unsqueeze(0) 
        if boundaries.dim() == 1:
            boundaries = boundaries.unsqueeze(0) 
        N,T,M,_F = *time_dep.shape, 
        # --- preprocess & run LSTM just like before ---
        #  a) compute existence mask from NaNs over (T,F)
        exists_mask = (~torch.isnan(time_dep)
                       .all(dim=-1)    # (N,T,M)
                       .any(dim=1)     # (N,M)
                      ).to(self.device)  # True if vehicle ever appears

        #  b) permute / flatten for LSTM
        td = time_dep.permute(0,2,1,3).contiguous().view(N*M, T, _F)
        td_clean = torch.nan_to_num(td, nan=0.0)
        seq_lens = compute_seq_lengths(td)   # you already have this
        packed   = rnn_utils.pack_padded_sequence(
                        td_clean, seq_lens.cpu(),
                        batch_first=True,
                        enforce_sorted=False
                    )
        _, (hn, _) = self.lstm(packed)
        veh_repr = hn.squeeze(0).view(N, M, self.lstm_hidden)  # (N,M,H)

        # --- build set representation ---
        # 1) embed each vehicle:  (N,M,set_dim)
        phi_h = self.phi(veh_repr)
        # 2) mask out nonexistent slots
        phi_h = phi_h * exists_mask.unsqueeze(-1).float()
        # 3) pool across M: (N, set_dim)
        set_repr = phi_h.sum(dim=1)
        # 4) broadcast back to per‐slot: (N,M,set_dim)
        set_repr = set_repr.unsqueeze(1).expand(-1, M, -1)

        # --- global info as before ---
        lm_clean = torch.nan_to_num(lane_markers, nan=0.0)
        lm_mask  = (~torch.isnan(lane_markers)).float().mean(dim=1, keepdim=True)
        gl = F.relu(self.global_fc(lm_clean)) * lm_mask
        gb = F.relu(self.boundary_fc(boundaries))
        g_comb = torch.cat([gl, gb], dim=1)
        g_info = F.relu(self.global_combine(g_comb))
        g_info = g_info.unsqueeze(1).expand(-1, M, -1)  # (N,M,global_dim)

        # --- combine everything per‐vehicle ---
        x = torch.cat([veh_repr, g_info, set_repr], dim=-1)  # (N,M,H+G+S)
        mean = self.combine_fc(x)                            # (N,M,2)
        log_std = self.log_std.expand(N, M, -1)

        return mean, log_std
    
    def get_action(self, time_dep, lane_markers, boundaries, agent_mask):
        """
        Sample actions from the policy distribution.
        
        Returns:
           action: (N, M, output_size)
           log_prob: (N, M, output_size) or summed over output_size per vehicle.
        """
        _agent_mask = agent_mask.unsqueeze(0) if agent_mask.dim() == 1 else agent_mask
        mean, log_std = self.forward(time_dep, lane_markers, boundaries)
        # print (mean, log_std)
        std = torch.exp(log_std)
        # Create a normal distribution per vehicle slot.
        dist = torch.distributions.Normal(mean, std)
        # Sample actions using reparameterization (this allows for differentiable sampling).
        action = dist.rsample()  # shape (N, M, output_size)
        # Compute log probabilities.
        log_prob = dist.log_prob(action)  # shape (N, M, output_size)
        log_prob = log_prob.sum(dim=-1)  # aggregate over action dimensions, shape (N, M)
        # Mask out non-agent slots: set log_prob to 0 for non-agent vehicles.
        # This means that when computing the loss, only entries with agent_mask==1 will contribute.
        masked_log_prob = log_prob * _agent_mask  # agent_mask is assumed to be float, with 1 or 0.
        masked_action = action * _agent_mask.unsqueeze(-1) 
        
        return masked_action, masked_log_prob
    
#########################################
# Critic Network (Value Function)
#########################################
class PPOCritic(nn.Module):
    def __init__(
        self,
        h_dim, v_dim, f_dim,
        lstm_hidden=64,
        global_dim=12,
        set_dim=32,                   # new: dimension of set encoder
        combined_hidden=64,
        device='cuda'
    ):
        super().__init__()
        self.T, self.M, self.F = h_dim, v_dim, f_dim
        self.lstm_hidden = lstm_hidden

        # 1) per‑vehicle LSTM
        self.lstm = nn.LSTM(input_size=self.F, hidden_size=lstm_hidden,
                            batch_first=True)

        # 2) Deep‑Set encoder φ: lstm_hidden → set_dim
        self.set_dim = set_dim
        self.phi = nn.Sequential(
            nn.Linear(lstm_hidden, 2*set_dim),
            nn.ReLU(),
            nn.Linear(2*set_dim, set_dim),
        )

        # 3) global info
        self.global_fc      = nn.Linear(10, global_dim)
        self.boundary_fc    = nn.Linear(2,  global_dim)
        self.global_combine = nn.Linear(2*global_dim, global_dim)

        # 4) final head now takes [veh_repr, global, set_repr]
        in_dim = lstm_hidden + global_dim + set_dim
        self.combine_fc = nn.Sequential(
            nn.Linear(in_dim, combined_hidden),
            nn.ReLU(),
            nn.Linear(combined_hidden, 1),
        )

        self.device = device
        self.to(device)
    
    def forward(self, time_dep, lane_markers, boundaries, agent_mask):
        """
        Forward pass.
        
        Parameters:
           time_dep: (N, T, M, F)
           lane_markers: (N, 10)
           boundaries: (N, 2)
           agent_mask: (N, M) binary mask for valid vehicles.
           
        Returns:
           values: (N, 1) scalar state-value estimates.
        """
        if time_dep.dim() == 3:
            time_dep = time_dep.unsqueeze(0) 
        if lane_markers.dim() == 1:
            lane_markers = lane_markers.unsqueeze(0) 
        if boundaries.dim() == 1:
            boundaries = boundaries.unsqueeze(0) 
        N, T, M, _F = time_dep.shape
        
        td = time_dep.permute(0,2,1,3).contiguous().view(N*M, T, _F)
        td_clean = torch.nan_to_num(td, nan=0.0)

        # compute seq lengths for pack_padded
        valid = ~torch.isnan(td).all(dim=-1)   # (N*M, T)
        seq_lens = valid.sum(dim=1).clamp(min=1)
        packed = rnn_utils.pack_padded_sequence(
            td_clean, seq_lens.cpu(),
            batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        veh_repr = hn.squeeze(0).view(N, M, self.lstm_hidden)  # (N,M,H)

        # --- 2) Deep‑Set pooling ---
        # a) φ on each slot
        phi_h = self.phi(veh_repr)                         # (N,M,set_dim)
        # b) mask out non‑existent vehicles
        phi_h = phi_h * agent_mask.unsqueeze(-1).float()
        # c) sum across M
        set_repr = phi_h.sum(dim=1)                        # (N,set_dim)
        # d) broadcast back
        set_repr_b = set_repr.unsqueeze(1).expand(-1, M, -1)  # (N,M,set_dim)

        # --- 3) global features (same as before) ---
        lm_clean = torch.nan_to_num(lane_markers, nan=0.0)   # (N,10)
        lm_mask  = (~torch.isnan(lane_markers)).float().mean(dim=1, keepdim=True)
        gl       = F.relu(self.global_fc(lm_clean)) * lm_mask
        gb       = F.relu(self.boundary_fc(boundaries))
        gc       = torch.cat([gl, gb], dim=1)
        g_info   = F.relu(self.global_combine(gc))
        # broadcast per vehicle
        g_info_b = g_info.unsqueeze(1).expand(-1, M, -1)    # (N,M,global_dim)

        # --- 4) combine everything per slot & compute values ---
        x = torch.cat([veh_repr, g_info_b, set_repr_b], dim=-1)  # (N,M,in_dim)
        v_slots = self.combine_fc(x).squeeze(-1)                 # (N,M)

        # --- 5) mask out invalid slots ---
        agent_values = v_slots * agent_mask.float()
        return agent_values
    
class SetEncoder(nn.Module):
    def __init__(self, f_dim, d_dim):
        super().__init__()
        # φ: F → D
        self.phi = nn.Sequential(
            nn.Linear(f_dim, 2*d_dim),
            nn.ReLU(),
            nn.Linear(2*d_dim, d_dim),
        )
        # optional ρ: D → D_out
        self.rho = nn.Sequential(
            nn.Linear(d_dim, 2*d_dim),
            nn.ReLU(),
            nn.Linear(2*d_dim, d_dim),
        )

    def forward(self, x):
        """
        x: (N, T, M, F)  with NaNs for non‑existent vehicles
        returns: (N, T, D)
        """
        # 1) existence mask: True where vehicle exists
        exists = ~torch.isnan(x[...,0])             # (N,T,M)

        # 2) replace NaNs with zeros (so φ(0)=0 if φ has bias=0)
        x0 = torch.where(exists.unsqueeze(-1), x, torch.zeros_like(x))

        # 3) embed every vehicle
        h = self.phi(x0)                             # (N,T,M,D)

        # 4) mask & sum (or mean)
        h = h * exists.unsqueeze(-1)                 # zero out missing
        sum_h = h.sum(dim=2)                         # (N,T,D)
        
        # if you want a mean:
        # cnt = exists.sum(dim=2, keepdim=True).clamp(min=1)
        # sum_h = sum_h / cnt                         # (N,T,D)

        # 5) optional ρ
        y = self.rho(sum_h)                          # (N,T,D)

        return y
    
def ppo_update(actor, critic, buffer, 
               actor_lr=3e-4, critic_lr=1e-3, 
               clip_ratio=0.2, train_iters=80):
    data = buffer.get()
    kn, lm, bd, mk, act, logp_old, ret, adv = data.values()

    # optimizers
    a_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    c_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    for _ in range(train_iters):
        # Policy loss
        dist, logp = actor.get_action(kn, lm, bd, mk)
        ratio = torch.exp(logp - logp_old)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.mean(torch.min(ratio * adv, clipped_ratio * adv))

        # Value loss
        value = critic(kn, lm, bd, mk)
        value_loss = torch.mean((ret - value).pow(2))

        # Update actor
        a_optimizer.zero_grad()
        policy_loss.backward()
        a_optimizer.step()

        # Update critic
        c_optimizer.zero_grad()
        value_loss.backward()
        c_optimizer.step()

    print (f"---- policy loss: {policy_loss.item()} -- value loss: {value_loss.item()} mean reward: {torch.mean(ret)} ----")


if __name__ == '__main__':
    render = True
    demo = False
    generation = True

    expert_data, df = convert_highd_sample_to_gail_expert(
        sample_csv=r"./data/26_sample_tracks.csv",
        meta_csv=r"E:\Data\highd-dataset-v1.0\data\26_recordingMeta.csv",
        forward=False,
        p_agent=0.90
    )
    device = 'cuda'

    # Create your environment instance.
    env = HighwayEnv(generation_mode=generation, demo_mode=demo, T=50, device=device)
    # Optionally, set expert data:
    # expert_data, df = convert_highd_sample_to_gail_expert(...); env.set_expert_data(expert_data)

    # Create PPO actor (policy) and critic networks.
    # Let T = history length, M = max number of vehicles, F = features (e.g., 4).

    # Uncomment and update the following line when expert_data is available:
    env.set_expert_data(expert_data)
    steps_per_epoch = 300
    epochs = 50

    actor  = PPOActor(50, 100, 7, device=device)
    critic = PPOCritic(50, 100, 7, device=device)
    buf = HighwayEnvBuffer(50, 100, 7, 2, size=steps_per_epoch, device=device)

    for epoch in trange(epochs):
        _obs, ep_ret, ep_len = env.reset(), 0, 0
        _ = env.render() if render else 0
        obs = _obs.values()
        for t in range(steps_per_epoch):
            action, logp = actor.get_action(*obs)
            # mask out lateral accelerations?
            action[...,1] = 0
            value = critic(*obs).detach()
            next_obs, rew, done, _ = env.step(action.squeeze(0).detach())
            _ = env.render() if render else 0
            ep_ret += rew; ep_len += 1
            buf.store(*obs, action.squeeze(0), logp.squeeze(0), rew.squeeze(0), value.squeeze(0))
            obs = next_obs.values()
            terminal = done or (t==steps_per_epoch-1)
            if terminal:
                last_val = critic(*obs).detach()
                buf.finish_path(last_val)
                _obs, ep_ret, ep_len = env.reset(), 0, 0
                _ = env.render() if render else 0
                obs = _obs.values()

        # after collecting data, perform PPO update
        ppo_update(actor, critic, buf)

        print(f"Epoch {epoch+1}/{epochs} complete")

    env.close()