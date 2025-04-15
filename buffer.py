import random
from collections import deque
import torch
from env import HighwayEnv, convert_highd_sample_to_gail_expert
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

class HighwayEnvMemoryBuffer:
    def __init__(self, capacity):
        """
        Initialize the memory buffer.     
        Parameters:
          capacity (int): Maximum number of transitions to store in the buffer.
        """
        self.capacity = capacity
        # A deque with a maximum length automatically discards the oldest elements
        self.buffer = deque(maxlen=capacity)


    def push(self, state, action, reward, next_state, done):
        """
        Saves a transition into the buffer.
        
        Parameters:
          state (torch.Tensor): The input state of shape (T, N, M).
          action (torch.Tensor): The taken actions of shape (N, 2).
          reward (float): The reward obtained after taking the action.
          next_state (torch.Tensor): The next state observation (shape as state).
          done (bool): Whether this transition terminated an episode.
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size, state_keys=['time_dependent', 'lane_markers', 'boundary_lines', 'agent_mask']):
        """
        Samples a random batch of transitions from the buffer.
        
        Parameters:
          batch_size (int): Number of transitions to sample.
          state_keys (list[str]): Keys of heterogeneous states
        Returns:
          A tuple of batches: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Stack the tensor transitions along a new batch dimension.
        return (    [ torch.stack([obs[key] for obs in states]) for key in state_keys ] ,
                    torch.stack(actions),
                    torch.tensor(rewards, dtype=torch.float32),
                    [ torch.stack([obs[key] for obs in next_states]) for key in state_keys ],
                    torch.tensor(dones, dtype=torch.float32)    )
    
    def __len__(self):
        return len(self.buffer)
    

if __name__ == '__main__':
    # Create the buffer
    buffer = HighwayEnvMemoryBuffer(300)

    expert_data, df = convert_highd_sample_to_gail_expert(
        sample_csv=r"./data/26_sample_tracks.csv",
        meta_csv=r"E:\Data\highd-dataset-v1.0\data\26_recordingMeta.csv",
        forward=False,
        p_agent=0.90
    )
    # Create the environment and set expert data.
    env = HighwayEnv(dt=0.2, T=50, generation_mode=False, demo_mode=True)
    # Uncomment and update the following line when expert_data is available:
    env.set_expert_data(expert_data)
    # 
    obs = env.reset() 

    num_steps = 300
    #
    for step in trange(num_steps):
        # For demonstration, sample random actions for each vehicle slot.
        # Action shape: (N_max, 2). Since have set DEMO_MODE=TRUE, the actions will be overwritten by the agents
        action = torch.full((env.N_max, 2), 0.0)
        # Step the environment: we get new observation, reward, done, and info.
        next_obs, reward, done, info = env.step(action)

        buffer.push(obs, action, reward, next_obs, done)

    
