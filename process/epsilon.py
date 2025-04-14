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
    parser.add_argument('--generate-by-class', type=int, default=1)
    parser.add_argument('--class-dims', type=int, nargs='+', default=[4,6,7])
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.02,0.05,0.10,0.15])
    parser.add_argument('--mu', type=int, default=100)
    parser.add_argument('--data-destination', type=str, default="D:/Productivity/Projects/RiskFieldImitation-main/data")
    args = parser.parse_args()
    return args

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
    X = np.clip( abs(data.ego_x - data.foe_x) -  0.5*(data.ego_length - data.foe_length), a_min=0, a_max=None).values *((data.foe_x - data.ego_x) / abs(data.ego_x - data.foe_x)).values * data.dir.values
    Y = np.clip( abs(data.ego_y - data.foe_y) -  0.5*(data.ego_width - data.foe_width), a_min=0, a_max=None).values *((data.foe_y - data.ego_y) / abs(data.ego_y - data.foe_y)).values * data.dir.values
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
        Input[:,0] = Input[:,0] / torch.quantile(Input[:,0], 0.95)
        Input[:,1] = Input[:,1] / 1
        Input[:,2] = Input[:,2] / 15
        Input[:,2] = Input[:,2] / 1.25
    dataset = torch.cat([Input, X.view(-1,1)], dim=1)
    dataset = torch.cat([dataset, Y.view(-1,1)], dim=1)
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
    tensor = get_processed_tensors(data).cpu().numpy()
    del data
    fdims = Args.feature_dims
    epsilons = Args.epsilons
    if Args.generate_by_class:
        AllStatus = list(product([0,1], repeat=3))
        for status in AllStatus:
            print (f"---- Generating samples for class {status}, epsilon={epsilons} ----")
            subset = tensor
            for c,cl in enumerate([4,6,7]):
                subset = subset[subset[:,cl]==status[c]]
            print (f"---- Found {len(subset)} samples within class {status} ----")
            resample_size = Args.max_sample if len(subset) > Args.max_sample else len(subset)
            _key = np.random.choice(range(len(subset)),resample_size)
            subset = torch.Tensor(subset[_key]).to('cuda')
            Indices_ = [ find_neighbors_batch(subset[:,0:8], 
                                                subset[k,0:8], 
                                                epsilons,
                                                mu = Args.mu,
                                                search_dims = Args.distance_search_dims,
                                                batch_size = Args.batch_size,
                                                )[0] for k in trange(len(subset))]
            print (f"---- Saving samples for class {status} ----")
            torch.save(subset, f"{Args.data_destination}/class_{status}_dataset")
            print (f"---- Saving epsilon buffer indices for class {status} ----")
            torch.save(Indices_, f"{Args.data_destination}/class_{status}_buffer_indices")
            del Indices_
            del subset
    else:
        subset = tensor
        resample_size = Args.max_sample if len(subset) > Args.max_sample else len(subset)
        _key = np.random.choice(range(len(subset)),resample_size)
        subset = tensor[_key].to('cuda')
        Indices_ = [ find_neighbors_batch(subset[:,0:8], 
                                                    subset[k,0:8], 
                                                    epsilons,
                                                    mu = Args.mu,
                                                    search_dims = Args.distance_search_dims,
                                                    batch_size = Args.batch_size,
                                                    )[0] for k in trange(len(subset))]
        print (f"---- Saving samples for all class ----")
        torch.save(subset, f"{Args.data_destination}/class_all_dataset")
        print (f"---- Saving epsilon buffer indices for all class ----")
        torch.save(Indices_, f"{Args.data_destination}/class_all_buffer_indices")
        del Indices_
        del subset