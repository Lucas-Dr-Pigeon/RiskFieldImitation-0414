import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import e, sqrt
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import multiprocessing
import scipy
import argparse

# calibrate a subject field under what conditions?
# 1. wo/w lane-changing (lateral speed < 0.2)
# 2. foe vehicle type car/truck (length>7.2)
# 3. free-flow/synchronized/jamed speed
# 4. perception (only front vehicles are counted)

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=10)
    parser.add_argument('--compile-spacing-data', type=int, default=0)
    parser.add_argument('--calibrate-risk-field', type=int, default=0)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--egolc', type=int, default=0)
    parser.add_argument('--foelc', type=int, default=0)
    parser.add_argument('--ignore-lc', type=int, default=1)
    parser.add_argument('--truck', type=int, default=0)
    parser.add_argument('--vb', type=float, default=1)
    parser.add_argument('--frac', type=float, default=0.85)
    parser.add_argument('--velocities', type=float, nargs='+', default=range(0,52))
    parser.add_argument('--lc-window', type=int, default=125)
    args = parser.parse_args()
    return args

class Vehicle:
    def __init__(self, id, pos, vel, accel=[0,0], dim=[4.5, 1.7]):
        self.id = id
        self.pos = pos
        self.vel = vel
        self.accel = accel
        self.dim = dim
        # define a vehicle type
        self.istruck = 1 if dim[0]>=7.2 else 0
        # define a lc status
        self.islc = 1 if abs(vel[1])>0.1 else 0

    def get_subjective_risk(self, foe, w0x=1, w1x=1, w0y=1, w1y=1, bx=2, by=2):
        gx = w0x + w1x*self.vel[0] + self.dim[0]/2 + foe.dim[0]/2  
        gy = w0y + w1y*self.vel[1] + self.dim[1]/2 + foe.dim[1]/2
        x = abs(self.pos[0] - foe.pos[0])
        y = abs(self.pos[1] - foe.pos[1])
        risk = e**(-abs(x/gx)**bx - abs(y/gy)**by)
        return risk


class HighD:
    def __init__(self, tracks=None, boundary=None, markers=None):
        self.tracks = tracks
        self.boundary = boundary
        self.markers = markers

    def compile_spacing_data(self, recordingIDs= np.arange(1,61), folder=r'E:\Data\highd-dataset-v1.0\data', lc_window=125, marker='ignore', skiprows=5, perception=False, dump=True):
        # process the dataset into something like.. ego_x, ego_y, foe_x, foe_y, ego_vel, ego_truck, ego_lc, foe_vel, foe_truck, foe_lc, ego_length, ego_width, foe_length, foe_width, y_marker1, y_marker2, y_bound1, y_bound2
        # lc: <string> 'disallow': no ego vehicle lane-changing, 'all': only lane-changing vehicles are considered as ego vehicle
        # marker: <string> 'ignore' 'include'
        # columns = ['ego_id', 'foe_id', 'recording_id', 'frame', 'ego_x', 'ego_y', 'foe_x', 'foe_y', 'ego_vel_x', 'ego_vel_y', 'ego_accel_x', 'ego_accel_y', 'ego_truck', 'ego_lc', 'ego_length', 'ego_width', 
        #            'foe_vel_x', 'foe_vel_y', 'foe_accel_x', 'foe_accel_y', 'foe_truck', 'foe_lc', 'foe_length', 'foe_width', 'y_marker1', 'y_marker2', 'y_bound1', 'y_bound2']
        data = pd.DataFrame([])
        # print (f"--{recordingIDs}--")
        # surroundColumns
        surroundColumns = ['precedingId','leftPrecedingId','leftAlongsideId','rightPrecedingId','rightAlongsideId'] if perception else ['precedingId','followingId','leftPrecedingId','leftAlongsideId','leftFollowingId','rightPrecedingId','rightAlongsideId','rightFollowingId'] 
        for r in tqdm(recordingIDs):
            rid = f"{r:02}"
            path_tracks = f"{folder}\{rid}_tracks.csv"
            path_meta = f"{folder}\{rid}_recordingMeta.csv"
            tracks = pd.read_csv(path_tracks).iloc[::skiprows, :]
            tracks.x += tracks.width/2
            tracks.y += tracks.height/2
            meta = pd.read_csv(path_meta)
            #
            tracks_by_id = tracks.groupby('id')
            # get all vehicle lane-changing status 
            lcstatus = tracks.groupby('id').laneId.nunique()
            # loop through all rows
            for i in trange(len(tracks)):
                row = tracks.iloc[i]
                # frame id corresponding to the row
                frame = row.frame
                #
                egotrack = tracks_by_id.get_group(row.id)
                # construct the ego object
                ego = Vehicle(row.id, pos=[row.x,row.y], vel=[row.xVelocity,row.yVelocity], accel=[row.xAcceleration,row.yAcceleration], dim=[row.width,row.height])
                # overwrite ego lc status
                ego_postlc = 1 if egotrack[(egotrack.frame>=frame-lc_window)&(egotrack.frame<frame)].laneId.nunique()>1 else 0
                ego_prelc = 1 if egotrack[(egotrack.frame>frame)&(egotrack.frame<=frame+lc_window)].laneId.nunique()>1 else 0
                # get surrounding vehicle ID
                _surroundIDs = row[surroundColumns].values
                surroundIDs = _surroundIDs[_surroundIDs>0].astype(int)
                # 
                curTrj = tracks[tracks.frame==frame]
                # loop through all surrounding vehicles
                for foeid in surroundIDs:
                    # get row corresponding to the current foe
                    if foeid not in curTrj.id.values:
                        continue
                    #
                    foetrack = tracks_by_id.get_group(foeid)
                    # print ('foe: ', foeid, curTrj[curTrj.id==foeid])
                    foerow = curTrj[curTrj.id==foeid].iloc[0]
                    # construct a foe object
                    foe = Vehicle(foeid, pos=[foerow.x,foerow.y], vel=[foerow.xVelocity,foerow.yVelocity], accel=[foerow.xAcceleration,foerow.yAcceleration], dim=[foerow.width,foerow.height])
                    # overwrite foe lc status
                    foe_postlc = 1 if foetrack[(foetrack.frame>=frame-lc_window)&(foetrack.frame<frame)].laneId.nunique()>1 else 0
                    foe_prelc = 1 if foetrack[(foetrack.frame>frame)&(foetrack.frame<=frame+lc_window)].laneId.nunique()>1 else 0
                    # get spacing data
                    spacings = {
                        'ego_x': ego.pos[0], 'ego_y': ego.pos[1], 'foe_x': foe.pos[0], 'foe_y': foe.pos[1], 
                        'ego_length': ego.dim[0], 'foe_length': foe.dim[0],
                        'ego_width': ego.dim[1], 'foe_width': foe.dim[1],
                        'ego_truck': ego.istruck, 'foe_truck': foe.istruck,
                        'ego_pre_lc': ego_prelc, 'foe_pre_lc': foe_prelc,
                        'ego_post_lc': ego_postlc, 'foe_post_lc': foe_postlc,
                        'ego_xVel': ego.vel[0], 'ego_yVel': ego.vel[1],
                        'ego_xAccel': ego.accel[0], 'ego_yAccel': ego.accel[1],
                        'foe_xVel': foe.vel[0], 'foe_yVel': foe.vel[1],
                        'foe_xAccel': foe.accel[0], 'foe_yAccel': foe.accel[1],
                        'y_marker1': 0 if marker == 'ignore' else 0,
                        'y_marker2': 0 if marker == 'ignore' else 0,
                        'y_bound1': 0 if marker == 'ignore' else 0,
                        'y_bound2': 0 if marker == 'ignore' else 0,
                        'recording_id': r, 'ego_id': ego.id, 'foe_id': foe.id,
                        'frame': frame, 
                    }
                    data = data.append(spacings, ignore_index=True)
            if dump:
                data.to_csv(f"../data/{rid}_spacings_2.csv")
                data = pd.DataFrame([]) 
        return 0 if dump else data  

    def multi_processing_compile_spacing_data(self, n_processes=10, lc_window=125):
        recordingSplits = np.array_split(np.arange(1,61), n_processes)
        with multiprocessing.Pool(n_processes) as pool:
            runs = [pool.apply_async(self.compile_spacing_data, kwds={"recordingIDs":p, "lc_window":lc_window} ) for p in recordingSplits]
            results = [ r.get() for r in runs]
        return results
    
def bootstrapping_calibrate_field(velocityGroups, data, egolc=False, foelc=False, truck=False, ignore_lc=False, frac=0.85, runs=20, vb=1):
    results = np.zeros((6,len(velocityGroups),runs))
    for v in range(len(velocityGroups)):
        vel = velocityGroups[v]
        vData = data[(abs(data.ego_vel)>=vel-vb)&(abs(data.ego_vel)<vel+vb)]
        vehicles = vData.double_id.unique()
        print (f"---------- estimating risk field for velocity group {vel} m/s ----------")
        results[0,v,:] = vel
        for it in trange(runs):
            # _data = data[(abs(data.ego_vel)>=vel-vb)&(abs(data.ego_vel)<vel+vb)].sample(frac=frac).reset_index(drop=True)
            sampled_ids = np.random.choice(vehicles, int(len(vehicles)*frac))
            _data = vData[vData.double_id.isin(sampled_ids)]
            if not ignore_lc:
                _data = vData[(vData.ego_lc==egolc)&(vData.foe_lc==foelc)]
            if truck:
                _data = vData[vData.foe_truck==1]
            res = calibrate_subjective_field_1(_data, gx=10, gy=2, bx=2, by=2, method='COBYLA')
            results[1,v,it] = len(vData.double_id.unique())
            results[2,v,it] = res[0][0]
            results[3,v,it] = res[1][0]
            results[4,v,it] = res[2]
            results[5,v,it] = res[3]
    return results

def multiprocessing_bootstrapping_calibrate_field(velocityGroups, data, processes=10, egolc=False, foelc=False, truck=False, ignore_lc=True, frac=0.85, runs=20, vb=1):
    velocitySplits = np.array_split(velocityGroups, int(processes*1.5))
    sampleSplits = [ data[(abs(data.ego_vel)>=min(Vels)-vb)&(abs(data.ego_vel)<max(Vels)+vb)] for Vels in velocitySplits ]
    with multiprocessing.Pool(processes) as pool:
        workers = [pool.apply_async(bootstrapping_calibrate_field, kwds={"velocityGroups": velocitySplits[p],
                                                                      "data": sampleSplits[p],
                                                                      "egolc": egolc,
                                                                      "foelc": foelc,
                                                                      "truck": truck,
                                                                      "ignore_lc": ignore_lc,
                                                                      "frac": frac,
                                                                      "runs": runs,
                                                                      "vb": vb,
                                                                      } ) for p in range(processes)]
        results = [ r.get() for r in workers]
    return np.concatenate(results, axis=1)
    

def calibrate_subjective_field_1(data, gx=3, gy=3, bx=0, by=0, method='COBYLA'):
    input = data
    _x = abs(input.ego_x - input.foe_x) - (input.ego_length/2 + input.foe_length/2) + 1e-4
    _y = abs(input.ego_y - input.foe_y) - (input.ego_width/2 + input.foe_width/2) + 1e-4
    _x = _x.clip(0)
    _y = _y.clip(0)
    def jlll_gx(gx, gy, bx, by):
        return np.sum(np.log(1 + 1e-4 - e**(-abs(_x/gx)**bx -abs(_y/gy)**by)))
    def jlll_gy(gy, gx, bx, by):
        return np.sum(np.log(1 + 1e-4 - e**(-abs(_x/gx)**bx -abs(_y/gy)**by)))
    def get_objective_gx(gx, gy, bx, by):
        return scipy.misc.derivative(jlll_gx, gx, args=(gy,bx,by), n=2)
    def get_objective_gy(gy, gx, bx, by):
        return scipy.misc.derivative(jlll_gy, gy, args=(gx,bx,by), n=2)
    def get_objective_beta(Beta, Gamma):
        # gx, gy = np.exp(np.array(Gamma))
        # bx, by = np.exp(np.array(Beta)) + 2
        gx, gy = Gamma
        bx, by = Beta
        risks = e**(-abs(_x/gx)**bx -abs(_y/gy)**by)-1e-4
        minimize_obj = - np.sum(np.log(1-risks))
        return minimize_obj
    def con_gamma_x(gx):
        return gx-1e-3
    def con_gamma_y(gy):
        return gy-1e-3
    def con_beta_x_low(bx):
        return bx-2
    def con_beta_y_low(by):
        return by-2
    def con_beta_x_up(bx):
        return 10-bx
    def con_beta_y_up(by):
        return 10-by
    for out_it in range(100):
        outlast_gx = gx
        outlast_gy = gy
        outlast_bx = bx
        outlast_by = by
        for it in range(100):
            last_gx = gx
            last_gy = gy
            res_gx = minimize(get_objective_gx, gx, args=(gy,bx,by), constraints=[{'type':'ineq','fun':con_gamma_x}], method=method)
            res_gy = minimize(get_objective_gy, gy, args=(gx,bx,by), constraints=[{'type':'ineq','fun':con_gamma_y}], method=method)
            gx = res_gx.x
            gy = res_gy.x
            if (abs(last_gx-gx)<0.1) and (abs(last_gy-gy)<0.1):
                break
        res_beta = minimize(get_objective_beta, [bx,by], args=[gx,gy], constraints=[{'type':'ineq','fun':con_beta_x_low},{'type':'ineq','fun':con_beta_y_low},{'type':'ineq','fun':con_beta_x_up},{'type':'ineq','fun':con_beta_y_up}], method=method)
        last_bx = bx
        last_by = by
        bx, by = res_beta.x
        # print (f"---gx:{gx} gy:{gy} bx:{bx} by:{by}---")
        if (abs(outlast_gx-gx)<0.1) and (abs(outlast_gy-gy)<0.1) and (abs(outlast_bx-bx)<0.1) and (abs(outlast_by-by)<0.1):
            break
    return gx, gy, bx, by










if __name__ == '__main__':
    Args = add_arguments()
    if Args.compile_spacing_data:
        highd = HighD()
        print (f"---------- compiling spacing data -----------")
        highd.multi_processing_compile_spacing_data(10, lc_window=Args.lc_window)
    elif Args.calibrate_risk_field:
        print (f"---------- loading spacing data -----------")
        data = pd.concat([ pd.read_csv(f"./data/new_{r:02}_spacings.csv",index_col=0) for r in np.arange(1,61)])
        data['double_id'] =  data.recording_id.astype(str) + data.ego_id.astype(str)
        processes = Args.processes
        velocities = Args.velocities
        egolc = Args.egolc
        foelc = Args.foelc
        truck = Args.truck
        ignore_lc = Args.ignore_lc
        frac = Args.frac
        runs = Args.runs
        vb = Args.vb
        results = multiprocessing_bootstrapping_calibrate_field(velocities, data,
                                                                processes = processes,
                                                                egolc = egolc,
                                                                foelc = foelc,
                                                                truck = truck,
                                                                ignore_lc = ignore_lc,
                                                                frac = frac,
                                                                runs = runs,
                                                                vb = vb
                                                                )
        np.save(f"./results/ggd/Velocities_{velocities[0]}_{velocities[-1]}_egolc_{egolc}_foelc_{foelc}_truck_{truck}_ignore_{ignore_lc}_vb_{vb}_bootstrap_vehicles.npy", results)


    # 
