import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import argparse
from itertools import product

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="D:/Productivity/Projects/RiskFieldImitation-main/data")
    parser.add_argument('--distance-search-dims', type=int, nargs='+', default=[0,1,2,3,4,6,7])
    parser.add_argument('--feature-dims', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--max-sample', type=int, default=5000000)
    parser.add_argument('--batch-size', type=int, default=5000000)
    parser.add_argument('--class-dims', type=int, nargs='+', default=[4,6,7])
    parser.add_argument('--epsilon', type=float, nargs='+', default=0.1)
    parser.add_argument('--x-norm', type=float, default=20)
    parser.add_argument('--y-norm', type=float, default=5)
    parser.add_argument('--mu', type=int, default=100)
    parser.add_argument('--data-destination', type=str, default="D:/Productivity/Projects/RiskFieldImitation-main/data")
    args = parser.parse_args()
    return args

def remove_similar_data(data, epsilon=0.1, use_for_loop=True, feature_dims=[0,1,2,3,4,5,6,7]):
    """
    Remove similar data points by keeping only one representative per group.
    
    Args:
        data (torch.Tensor): (N, D) tensor of data points.
        epsilon (float): Distance threshold to consider points as similar.

    Returns:
        torch.Tensor: Filtered tensor with reduced similar points.
    """
    if use_for_loop:
        device = data.device  # Ensure we work on GPU if needed
        selected_indices = []  # Stores unique indices
        used_mask = torch.zeros(data.shape[0], dtype=torch.bool, device=device)  # Track selected points
        for i in trange(data.shape[0]):
            if used_mask[i]:  # Skip if already covered
                continue
            selected_indices.append(i)  # Select this point
            distances = torch.norm(data[:,feature_dims] - data[i][feature_dims], dim=1)  # Compute distances to all points
            used_mask |= (distances < epsilon)  # Mark all nearby points as used
        return data[selected_indices]
    # Use tqdm for a progress bar
    else:
        device = data.device  # Ensure we work on the correct device
        remaining_indices = torch.arange(data.shape[0], device=device)  # Track remaining indices
        selected_points = []  # Store filtered points
        with tqdm(total=data.shape[0], desc="Processing", unit="samples") as pbar:
            while remaining_indices.numel() > 0:
                ref_idx = remaining_indices[0]  # Take the first available index
                ref_point = data[ref_idx].unsqueeze(0)  # Reshape for broadcasting
                selected_points.append(ref_point.squeeze(0))  # Keep the reference point

                # Compute distances between the reference point and remaining points
                distances = torch.norm(data[remaining_indices] - ref_point, dim=1)

                # Keep only points that are farther than epsilon
                mask = distances >= epsilon
                remaining_indices = remaining_indices[mask]  # Filter indices

                # Update progress bar
                pbar.update(len(data) - len(remaining_indices))

        return torch.stack(selected_points)

def get_processed_tensors(data, normalize=True):
    # add direction attributes
    vehicle_mean_speed = data.groupby(['ego_id','recording_id']).ego_xVel.transform('mean')
    vehicle_abs_speed = vehicle_mean_speed.abs() 
    data['dir'] = vehicle_mean_speed / vehicle_abs_speed
    # add lane changing attributes
    ego_pre_lc = data.groupby(['ego_id','recording_id']).ego_pre_lc.transform('any')
    ego_post_lc = data.groupby(['ego_id','recording_id']).ego_post_lc.transform('any')
    ego_lc = ego_pre_lc | ego_post_lc
    foe_pre_lc = data.groupby(['foe_id','recording_id']).foe_pre_lc.transform('any')
    foe_post_lc = data.groupby(['foe_id','recording_id']).foe_post_lc.transform('any')
    foe_lc = foe_pre_lc | foe_post_lc
    data['ego_lc'] = ego_lc
    data['foe_lc'] = foe_lc
    ego_xVel = np.abs(data.ego_xVel.values)
    ego_yVel = np.abs(data.ego_yVel.values) * data.dir.values
    rel_xVel = -((data.ego_xVel - data.foe_xVel)*(data.ego_x - data.foe_x) / abs(data.ego_x - data.foe_x)).values
    rel_yVel = -((data.ego_yVel - data.foe_yVel)*(data.ego_y - data.foe_y) / abs(data.ego_y - data.foe_y)).values
    ego_lc = data.ego_lc
    foe_lc = data.foe_lc
    ego_class = data.ego_truck.values
    foe_class = data.foe_truck.values
    X = np.clip( abs(data.ego_x - data.foe_x) -  0.5*(data.ego_length + data.foe_length), a_min=0, a_max=None).values *((data.foe_x - data.ego_x) / abs(data.ego_x - data.foe_x)).values * data.dir.values
    Y = np.clip( abs(data.ego_y - data.foe_y) -  0.5*(data.ego_width + data.foe_width), a_min=0, a_max=None).values *((data.foe_y - data.ego_y) / abs(data.ego_y - data.foe_y)).values * data.dir.values
    np.nan_to_num(X, 0)
    np.nan_to_num(Y, 0)
    X = torch.Tensor(X).to('cuda').detach()
    Y = torch.Tensor(Y).to('cuda').detach()
    Input = np.array(
    [
        ego_xVel, ego_yVel, rel_xVel, rel_yVel, ego_lc, foe_lc, ego_class, foe_class
    ]
    ).T
    np.nan_to_num(Input, 0)
    Input = torch.Tensor(Input).to('cuda')
    if normalize:
        Input[:,0] = Input[:,0] / torch.quantile(Input[:,0], 0.80)
        Input[:,1] = Input[:,1] / 1
        Input[:,2] = Input[:,2] / 15
        Input[:,3] = Input[:,3] / 1
    dataset = torch.cat([Input, X.view(-1,1)], dim=1)
    dataset = torch.cat([dataset, Y.view(-1,1)], dim=1)
    _x = torch.tensor((data.foe_x - data.ego_x).values, device='cuda')
    _y = torch.tensor((data.foe_x - data.ego_x).values, device='cuda')
    dataset = torch.cat([dataset, _x.view(-1,1)], dim=1)
    dataset = torch.cat([dataset, _y.view(-1,1)], dim=1)
    return dataset

def find_neighbors_batch(data, query_point, epsilons=[0.01,0.02,0.03,0.05,0.1,0.15], mu=100, search_dims=[0,1,2,3,4,6,7], batch_size=10000, k=0):
    """
    Find all points in 'data' within 'epsilon' distance of 'query_point' using brute-force search.
    """
    num_samples = data.shape[0]
    indices_list = [] 
    for i in range(0, num_samples, batch_size):
        bound = i + batch_size if i+batch_size < num_samples else num_samples
        batch = data[i:bound] 
        distances = torch.norm(batch[:,search_dims] - query_point[search_dims], dim=1)
        mask = distances <= epsilons[k]
        found_indices = torch.nonzero(mask, as_tuple=True)[0] + i
        indices_list.append(found_indices.to('cuda'))
    indices = torch.cat(indices_list) if indices_list else torch.tensor([]).to('cuda')
    _k = k+1
    if len(indices)<mu and _k<len(epsilons):
        indices = find_neighbors_batch(data, query_point, epsilons=epsilons, mu=mu, search_dims=search_dims, batch_size=batch_size, k=_k)
    return indices

if __name__ == '__main__':
    Args = add_arguments()
    print (f"---- Reading raw data ----")
    data = pd.concat([ pd.read_csv(f"{Args.data_path}/{r:02}_spacings_2.csv",index_col=0) for r in trange(1,61)])
    print (f"---- Generating tensor ----")
    data = get_processed_tensors(data).to('cuda')
    data[:,8] = data[:,8]/Args.x_norm
    data[:,9] = data[:,9]/Args.y_norm
    reduced_data = remove_similar_data(data, Args.epsilon)
    torch.save(reduced_data, f"{Args.data_destination}/reduced_data")