# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time
import copy

tqdm.pandas()


def plot_segment_xy(data):
    def plot_each_id(subtrj):
        #print (subtrj)
        plt.plot(subtrj.x_position, subtrj.y_position, color='grey', linewidth=0.5)
    data.progress_apply(lambda subtrj: plot_each_id(subtrj), axis=1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    
def plot_section_xy(data, xlim=[-np.inf, np.inf], ylim=[-np.inf, np.inf], tlim=[-np.inf, np.inf], direction=0):
    data = data if direction==0 else data[data.direction==direction]
    def plot_each_id(subtrj):
        x = np.array(subtrj.x_position)
        y = np.array(subtrj.y_position)
        t = np.array(subtrj.timestamp)
        
        s_x = np.where((x>=xlim[0])&(x<xlim[1]))[0]
        s_y = np.where((y>=ylim[0])&(y<ylim[1]))[0]
        s_t = np.where((t>=tlim[0])&(t<tlim[1]))[0]
        
        #print (s_x, s_y, s_t)
        
        s = np.intersect1d(s_x, s_y)
        s = np.intersect1d(s, s_t)
        #print (s)
        p = plt.plot(x[s], y[s], color='red', linewidth=1) if len(s) else 0
        sc = plt.scatter(x[s], y[s], color='red', s=1) if len(s) else 0
    data.progress_apply(lambda subtrj: plot_each_id(subtrj), axis=1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    
def plot_section_xt(data, xlim=[-np.inf, np.inf], ylim=[-np.inf, np.inf], tlim=[-np.inf, np.inf], direction=0):
    data = data if direction==0 else data[data.direction==direction]
    def plot_each_id(subtrj):
        x = np.array(subtrj.x_position)
        y = np.array(subtrj.y_position)
        t = np.array(subtrj.timestamp)
        
        s_x = np.where((x>=xlim[0])&(x<xlim[1]))[0]
        s_y = np.where((y>=ylim[0])&(y<ylim[1]))[0]
        s_t = np.where((t>=tlim[0])&(t<tlim[1]))[0]
        
        #print (s_x, s_y, s_t)
        
        s = np.intersect1d(s_x, s_y)
        s = np.intersect1d(s, s_t)
        #print (s)
        if direction:
            p = plt.plot(t[s], x[s]*direction, color='blue', linewidth=0.5) if len(s) else 0
        else:
            p = p = plt.plot(t[s], x[s], color='blue', linewidth=0.5) if len(s) else 0
    data.progress_apply(lambda subtrj: plot_each_id(subtrj), axis=1)
    ax = plt.gca()
    #ax.set_aspect('equal', adjustable='box')
    plt.savefig('results/trj.png',dpi=1200)
    plt.show()
    
def compile_data(data, xlim=[-np.inf, np.inf], ylim=[-np.inf, np.inf], tlim=[-np.inf, np.inf], direction=0):
    df = pd.DataFrame([], columns=data.columns)
    data = data if direction==0 else data[data.direction==direction]
    for i in trange(len(data)):
        track = data.iloc[i]
        for t in range(len(track.timestamp)):
            row = copy.deepcopy(track)
            
            if not tlim[0] <= track.timestamp[t] < tlim[1]:
                continue
            if not xlim[0] <= track.x_position[t] < xlim[1]:
                continue
            if not ylim[0] <= track.y_position[t] < ylim[1]:
                continue
            
            row.timestamp = track.timestamp[t]
            row.x_position = track.x_position[t]
            row.y_position = track.y_position[t]
            df = df.append(row)
    return df
            

    



if __name__ == '__main__':
    path = 'D:/Productivity/PythonProjects/Motion I-24/data/637d8ea678f0cb97981425dd__post3.json'
    data = pd.read_json(path)
    
    
    
    data.columns
    
    data.timestamp.iloc[0]
    data.first_timestamp.iloc[0]
    data.last_timestamp.iloc[0]
    
    data._id
    
    data.local_fragment_id
    
    
    datetime.fromtimestamp(data.last_timestamp.iloc[0])
    
    
    datetime.fromtimestamp(data.timestamp.iloc[0][10]) - datetime.fromtimestamp(data.first_timestamp.iloc[0])
    
    
    
    data.last_timestamp.iloc[0] - data.first_timestamp.iloc[0]
    
    
    
    
    datetime.fromtimestamp(data.last_timestamp.max()) - datetime.fromtimestamp(data.first_timestamp.min())


    len(data)
    vids = [ data._id.iloc[i]['$oid'] for i in range(len(data)) ]
    len(data), len(set(vids))


    durations = [ (datetime.fromtimestamp(data.last_timestamp.iloc[i]) - datetime.fromtimestamp(data.first_timestamp.iloc[i])).total_seconds() for i in range(len(data))]
    durations = np.array(durations)
    
    plt.hist(durations[durations<500], bins=80, color='grey')
    plt.show()
    
    plot_segment_xy(data)
    
    
    plot_section_xy(data, 
                    xlim=[59.19*5280, 59.21*5280],
                    ylim=[0,72],
                    tlim=[time.mktime(datetime(2022, 11, 23, 7, 36, 0, 0).timetuple())-0.20, 
                          time.mktime(datetime(2022, 11, 23, 7, 36, 0, 0).timetuple())+0.20
                      ],
                    direction=-1
                )
    
    
    plot_section_xt(data, 
                    xlim=[59.1*5280, 59.3*5280],
                    ylim=[12,24],
                    tlim=[time.mktime(datetime(2022, 11, 23, 7, 35, 0, 0).timetuple()), 
                          time.mktime(datetime(2022, 11, 23, 7, 37, 0, 0).timetuple())
                      ],
                    direction=-1
                )
    
    
    time.mktime(datetime(2022, 12, 2, 7, 38, 57).timetuple())
    
    
    (data.ending_x.max() - data.starting_x.min())/5280
    
    
    data.starting_x.min()/5280
    data.ending_x.max()/5280
    
    
    data.direction
    
    data.compute_node_id
    
    edata = data.explode(['timestamp', 'x_position', 'y_position']).reset_index(drop=True)
    

    edata.to_csv('data/nov-23.csv',index=False)    
    
    edata[(edata.timestamp<=time.mktime(datetime(2022, 11, 23, 7, 37, 0, 0).timetuple()))&(edata.timestamp>=time.mktime(datetime(2022, 11, 23, 7, 35, 0, 0).timetuple()))]
    
    
    datetime.fromtimestamp(edata.timestamp.min()) 
    datetime.fromtimestamp(edata.timestamp.max()) 

    data1 = edata[(edata.timestamp<=time.mktime(datetime(2022, 11, 23, 8, 0, 0, 0).timetuple()))&(edata.timestamp>=time.mktime(datetime(2022, 11, 23, 7, 0, 0, 0).timetuple()))]
    data1.to_csv('E:\Data\Motion I-24/nov-23-7am-8am.csv',index=False)    
    
    data2 = edata[(edata.timestamp<=time.mktime(datetime(2022, 11, 23, 9, 0, 0, 0).timetuple()))&(edata.timestamp>=time.mktime(datetime(2022, 11, 23, 8, 0, 0, 0).timetuple()))]
    data2.to_csv('E:\Data\Motion I-24/nov-23-8am-9am.csv',index=False)    
    
    data3 = edata[(edata.timestamp<=time.mktime(datetime(2022, 11, 23, 10, 0, 0, 0).timetuple()))&(edata.timestamp>=time.mktime(datetime(2022, 11, 23, 9, 0, 0, 0).timetuple()))]
    data3.to_csv('E:\Data\Motion I-24/nov-23-9am-10am.csv',index=False)    
    
    data4 = edata[(edata.timestamp<=time.mktime(datetime(2022, 11, 23, 11, 0, 0, 0).timetuple()))&(edata.timestamp>=time.mktime(datetime(2022, 11, 23, 10, 0, 0, 0).timetuple()))]
    data4.to_csv('E:\Data\Motion I-24/nov-23-10am-11am.csv',index=False)   
    
    
    
    edata.timestamp.max()
    edata.timestamp.min()
    
    #max(data[data.direction==-1]['y_position'].max())
    
    # arr = np.array([5,6,7,8,9,10])
    # np.where((arr<8)&(arr>5))[0]
