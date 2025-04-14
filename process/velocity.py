# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm, trange


def compute_velocity(irange, data):
    output = []
    for i in tqdm(irange):
        track = data.iloc[i]
        xs = np.array(track.x_position)
        ts = np.array(track.timestamp)
        x_velocity = (xs[1:len(xs)] - xs[0:len(xs)-1])/(ts[1:len(ts)] - ts[0:len(ts)-1])
        x_velocity = np.append(x_velocity, x_velocity[-1])
        x_velocity = x_velocity * track.direction
        x_velocity[x_velocity<0] = 0
        x_velocity = x_velocity.tolist()
        
        ys = np.array(track.y_position)
        y_velocity = (ys[1:len(ys)] - ys[0:len(ys)-1])/(ts[1:len(ts)] - ts[0:len(ts)-1])
        y_velocity = np.append(y_velocity, y_velocity[-1])
        y_velocity = y_velocity.tolist()
        
        output.append([i, x_velocity, y_velocity])
        
        
        # print (i, len(x_velocity), len(y_velocity), len(track.timestamp))
    return output



if __name__ == '__main__':
    data = pd.read_json(r"E:\Data\Motion I-24\637b023440527bf2daa5932f__post1.json")
    # n_process = 10
    # splits = np.array_split(np.arange(len(data)), n_process)
    # PARAMS = [(irange, data) for irange in splits]
    # with multiprocessing.Pool(n_process) as pool:
    #     runs = [pool.apply_async(compute_velocity, p) for p in PARAMS]
    #     results = [ r.get() for r in runs]
    
    
    results = compute_velocity(np.arange(len(data)), data)
    
    data['x_velocity'] = [track[1] for track in tqdm(results)]
    data['y_velocity'] = [track[2] for track in tqdm(results)]
    data['tid'] = np.arange(len(data))
    
    
    data.to_json(r'E:\Data\Motion I-24\nov-21-tracks.json')
    
    data[data.direction==1].iloc[343].x_velocity
    
    checked = [ len(data.iloc[i].x_velocity) == len(data.iloc[i].timestamp) for i in trange(len(data)) ]
    
    
    set(checked)
    
    data = pd.read_json(r'E:\Data\Motion I-24\nov-21-tracks.json')
    edata = data.explode(['timestamp', 'x_position', 'y_position', 'x_velocity', 'y_velocity']).reset_index(drop=True)
    
    # data['x_velocity'] = [[] for i in range(len(data))]
    # data['y_velocity'] = [[] for i in range(len(data))]
    
    # for track in tqdm(results):
    #     i = track[0]
    #     data.iloc[i, data.columns.get_loc('x_velocity')] = track[1]
    #     data.iloc[i, data.columns.get_loc('y_velocity')] = track[2]        
        
    
    # data.iloc[14214, data.columns.get_loc('x_velocity')] 
    