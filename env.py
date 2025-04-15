import gym
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def convert_highd_sample_to_gail_expert(sample_csv, meta_csv, forward=True, p_agent=0.5, smooth_dt=0.2):
    """
    Converts the HIGHD sample dataset into expert data for GAIL.
    
    Assumptions:
      - The sample dataset (e.g. highd_sample.csv) contains columns including:
          'frame', 'id', 'x', 'y', 'width', 'height', 'vx', 'vy', 'ax', 'ay'
      - Vehicles traveling in the target direction are selected.
        Here we assume that forward traffic has positive vx.
      - The lane meta file (e.g. 25_tracksMeta.csv) contains a column named 'lane_position'
        that provides lane marker positions.
      - The sample frame rate is 25 Hz (i.e. frame interval = 1/25 sec). We resample every 5 frames for a simulation step of 0.2 sec.
    """
    # Read the CSV files
    df = pd.read_csv(sample_csv)
    meta = pd.read_csv(meta_csv)

    # Smooth y acceleration
    df = overwrite_y_acceleration_expert(df, dt=smooth_dt)
    
    # Filter for one direction; here we assume forward means vx > 0.
    forward_mask = df.groupby('id').xVelocity.transform('mean') > 0
    df = df[forward_mask] if forward else df[~forward_mask]

    marks = meta['lowerLaneMarkings'].iloc[0] if forward else meta['upperLaneMarkings'].iloc[0]
    marks = np.array(marks.split(';'), dtype=float)

    boundarylines = marks[[0,-1]]
    lanemarkers = marks[1:-1]
    boundarylines[0] -= 0.25
    boundarylines[1] += 0.25
    lanecenters = (marks[:-1] + marks[1:])/2
    lanecenters = lanecenters if forward else lanecenters[::-1]
    
    # Compute vehicle center positions
    df['center_x'] = df['x'] + 0.5 * df['width']
    df['center_y'] = df['y'] + 0.5 * df['height']

    # Get road start:
    road_start = df['center_x'].min() + 30.0 if forward else df['center_x'].max() - 30.0 
    road_end = df['center_x'].max() - 30.0 if forward else df['center_x'].min() + 30.0
    
    # Screenline check
    segment_mask = ((df['center_x'] + 0.5 * df['width']) >= road_start ) & ((df['center_x'] - 0.5 * df['width']) <= road_end ) if forward else \
                    ((df['center_x'] - 0.5 * df['width']) <= road_start ) & ((df['center_x'] + 0.5 * df['width']) >= road_end )
    df = df[segment_mask]
    # # Resample frames: select frames where frame % 5 == 0 (0.2 sec intervals)
    # df = df[df['frame'] % 5 == 0]

    # Get each vehicles' target lane centers
    veh_last_y = df.sort_values('frame').groupby('id').last()['y']
    laneindices = np.argmin( np.abs(lanecenters - veh_last_y.values.reshape(-1,1)), axis=1)
    last_lanecenters = lanecenters[laneindices]
    veh_last_lanecenters = pd.Series(last_lanecenters, index=veh_last_y.index)
    df['yTarget'] = df['id'].map(veh_last_lanecenters)

    # Reindex lane centers
    df['_laneId'] = np.argmin( np.abs(lanecenters - df.center_y.values.reshape(-1,1)), axis=1)

    # Get truck ratios per lane
    num_cars = np.array([ df[(df._laneId==l)&(df.width<10)].id.nunique() for l in range(len(lanecenters))])
    num_trucks = np.array([ df[(df._laneId==l)&(df.width>=10)].id.nunique() for l in range(len(lanecenters))])
    lane_truck_ratios = num_trucks / (num_cars + num_trucks)

    # Sort frames
    frames = sorted(df['frame'].unique())
    expert_frames = []
    vehicle_agent_status = {}  # Track agent assignment by vehicle id across frames.
    for frame in frames:
        frame_data = df[df['frame'] == frame].copy()
        agent_status_list = []
        for idx, row in frame_data.iterrows():
            veh_id = row['id']
            # If new vehicle, assign agent status randomly.
            if veh_id not in vehicle_agent_status:
                vehicle_agent_status[veh_id] = (np.random.rand() < p_agent)
            agent_status_list.append(vehicle_agent_status[veh_id])
        frame_data['agent_status'] = agent_status_list
        expert_frames.append(frame_data)
    
    lane_arrival_rates = get_lane_arrival_rates(df)
    lane_change_ratios = get_lane_change_ratios(df)
    lane_avg_speeds = get_lane_average_speed(df)
    
    expert_data = {
        'frames': expert_frames,
        'lanemarkers': lanemarkers,
        'boundarylines': boundarylines,
        'lanecenters': lanecenters,
        'start': road_start,
        'end': road_end,
        'lane_arrival_rates': lane_arrival_rates,
        'lane_change_ratios': lane_change_ratios,
        'lane_avg_speeds': lane_avg_speeds,
        'lane_truck_ratios': lane_truck_ratios
    }
    
    return expert_data, df

def get_lane_arrival_rates(df):
    # Assume df is your DataFrame with columns: 'id', 'frame', 'laneId'
    global_first_frame = df['frame'].min()
    # Get the time length of dataset in seconds
    lenT = (df['frame'].max() - df['frame'].min()) * (1/25)
    # For each vehicle, get the first appearance (minimum frame)
    first_appearance = df.sort_values('frame').groupby('id', as_index=False).first()
    # Exclude vehicles that first appear in the global first frame
    first_appearance_after = first_appearance[first_appearance['frame'] > global_first_frame]
    # Create a mapping (or simply keep the columns 'id' and 'laneId')
    first_appearance_laneId = first_appearance_after[['id', '_laneId']]
    # Get the number of vehicles spawned on each lane
    lane_vehicles_spawned = first_appearance_laneId.groupby('_laneId').size()
    # Get the number arrival ratio per lane, from left to right
    lane_arrival_ratio = lane_vehicles_spawned / lenT
    # Reverse if backward
    lane_arrival_ratio = lane_arrival_ratio.values 
    return lane_arrival_ratio

def get_lane_change_ratios(df):
    # For each vehicle, get the first appearance (minimum frame)
    first_appearance = df.sort_values('frame').groupby('id', as_index=False).first()
    # last appearance 
    last_appearance = df.sort_values('frame').groupby('id', as_index=False).last()
    # get arrival and departure lane map
    arrival_lane_map = first_appearance[['id', '_laneId']].set_index('id')
    departure_lane_map = last_appearance[['id', '_laneId']].set_index('id')
    # Combine into a DataFrame.
    arr_depart = pd.DataFrame({
        'arrival': arrival_lane_map._laneId,
        'departure': departure_lane_map._laneId
    }, index=arrival_lane_map.index)
    # Create a contingency table that counts vehicles for each (arrival, departure) pair.
    table = pd.crosstab(arr_depart['arrival'], arr_depart['departure'])
    # Determine all lanes that appear in either series.
    all_lanes = sorted(set(arrival_lane_map._laneId.unique()) | set(departure_lane_map._laneId.unique()))
    # Reindex the table to ensure it has rows and columns for all lanes.
    # This fills missing rows/columns with 0.
    table = table.reindex(index=all_lanes, columns=all_lanes, fill_value=0)
    # Normalize each row (arrival lane) so that they sum to 1.
    M = table.div(table.sum(axis=1), axis=0).values
    # reverse the table if backward
    lcRatio = M 
    return lcRatio

def get_lane_average_speed(df):
    lane_vel = df.groupby('_laneId').xVelocity.mean().values
    lane_avg_vel = lane_vel 
    return lane_avg_vel

def overwrite_y_acceleration_expert(df, dt=1/25):
    step = int(dt * 25)
    def compute_for_vehicle(veh):
        veh = veh.sort_values('frame').copy()
        new_ay = []
        n = len(veh)
        for t in range(n):
            if t + step < n:
                ay = (veh['yVelocity'].iloc[t+step] - veh['yVelocity'].iloc[t]) / dt
            else:
                ay = 0
            new_ay.append(ay)
        veh['yAcceleration'] = new_ay
        return veh
    # Apply the computation for each vehicle (grouped by 'id').
    df = df.groupby('id', group_keys=False).apply(compute_for_vehicle)
    return df


class HighwayEnv(gym.Env):
    def __init__(self,
                 road_start=0.0,
                 road_end=1000.0,
                 N_max=100,
                 dt=0.2,
                 T=4,
                 arrival_rate=[0.1, 0.1, 0.1],
                 avg_speed=[20.0, 20.0, 20.0],
                 lanemarkers=[34.69,38.42],
                 boundarylines=[30.6,42.63],
                 lane_change_ratio=[[0.8,0.15,0.05],[0.1,0.8,0.1],[0.05,0.15,0.8]],
                 lane_truck_ratio=[0.1, 0.0, 0.1],
                 generation_mode=False,
                 demo_mode=False,
                 total_steps=600,
                 remove_crash=True,
                 ):
        super(HighwayEnv, self).__init__()
        
        # Road parameters (to be updated via expert data if available)
        self.road_start = road_start
        self.road_end = road_end
        # Lane marker positions (y-values)
        self.lanemarkers = lanemarkers     
        # Two values: [lower_boundary, upper_boundary]
        self.boundarylines = boundarylines      
        # get the lane centers' positions
        self.lanecenters = np.concatenate((boundarylines[:1], lanemarkers, boundarylines[-1:]))   # Get the lane center positions
        # the max number of vehicles allowed on the road
        self.N_max = N_max
        # the second between consecutive frames
        self.dt = dt
        # the length of history in feature sets
        self.T = T
        # the arrival rate of traffic each lanes
        self.lane_arrival_rate = arrival_rate
        # the average speed each lanes
        self.lane_avg_speed = avg_speed
        # the intensity of lane changing between lanes
        self.lane_change_ratio = lane_change_ratio
        # the truck ratio per lane
        self.lane_truck_ratio = lane_truck_ratio
        # set if the environment is in a generation mode
        self.generation_mode = generation_mode
        # set if the environment is in a demo mode
        self.demo_mode = demo_mode
        # Kinematic state now has 7 elements: [x, y, xVelocity, yVelocity, length, width, id]
        self.M1 = 8
        # Time-independent features remain: [vehicle_length, vehicle_width, agent_indicator, target]
        self.M2 = 2
        # Number of total steps to roll-out
        self.total_steps = total_steps
        # Remove colliding agents
        self.remove_crash_agents = remove_crash
        # Define observation space.
        # time_dependent observations now only include the first 4 columns.
        self.observation_space = gym.spaces.Dict({
            'time_dependent': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.T, self.N_max, 7), dtype=np.float32),
            # 'time_independent': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.N_max, self.M2), dtype=np.float32),
            'lane_markers': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),  # arbitrary max shape
            'boundary_lines': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            # 'mask': gym.spaces.Box(low=0, high=1, shape=(self.N_max,), dtype=np.float32),
            'agent_mask': gym.spaces.Box(low=0, high=1, shape=(self.N_max,), dtype=np.float32)
        })

        # Action space: (N_max, 2) for [xAcceleration, yAcceleration]
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.N_max, 2), dtype=np.float32)
        
        # Initialize state tensors.
        # kinematics shape: (N_max, 7): [x, y, xVelocity, yVelocity, length, width, id]
        self.kinematics = torch.full((self.N_max, self.M1), float('nan') )
        self.exists = torch.zeros(self.N_max, dtype=torch.bool)
        self.agent_mask = torch.zeros(self.N_max, dtype=torch.bool)
        # Time-independent features: [vehicle_length, vehicle_width, agent_indicator]
        # self.features = torch.full((self.N_max, self.M2), float('nan'))
        self.history = torch.full((self.T, self.N_max, self.M1), float('nan'))
        
        # Track vehicle ids separately for easier lookup.
        self.vehicle_ids = torch.full((self.N_max,), -1, dtype=torch.int64)
        
        # Placeholder for expert data.
        self.expert_data = None
        self.expert_frame_idx = 0


    def set_expert_data(self, expert_data, rectify_y=True):
        """
        Provide the expert data (from convert_highd_sample_to_gail_expert) to the environment.
        The expert_data should contain keys: 'frames', 'lanemarkers', 'boundarylines', 'start', 'end'.
        """
        self.expert_frame_idx = 0
        self.expert_data = expert_data

    def reset(self):
        """
        When resetting, if expert data is provided, initialize the environment using the first frame.
        This includes setting road start/end and lane markers / boundary lines.
        """
        self.kinematics = torch.full((self.N_max, self.M1), float('nan') )
        self.exists = torch.zeros(self.N_max, dtype=torch.bool)
        self.agent_mask = torch.zeros(self.N_max, dtype=torch.bool)
        # self.features = torch.full((self.N_max, self.M2), float('nan'))
        self.vehicle_ids = torch.full((self.N_max,), -1, dtype=torch.int64)
        self.history = torch.full((self.T, self.N_max, self.M1), float('nan'))
        self.expert_mask = torch.zeros(self.N_max, dtype=torch.bool)
        self.cumu_id = 100
        # 
        self.dead = set()
        
        if self.expert_data is not None:
            self.road_start = self.expert_data['start']
            self.road_end = self.expert_data['end']
            first_frame = self.expert_data['frames'][0]
            self.lanemarkers = self.expert_data['lanemarkers']
            self.boundarylines = self.expert_data['boundarylines']
            self.lanecenters = self.expert_data['lanecenters']
            self.lane_arrival_rate = self.expert_data['lane_arrival_rates']
            self.lane_change_ratio = self.expert_data['lane_change_ratios']
            self.lane_avg_speed = self.expert_data['lane_avg_speeds']
            self.lane_truck_ratio = self.expert_data['lane_truck_ratios']
            self.total_steps = len(self.expert_data['frames'])
            for i, (_, row) in enumerate(first_frame.iterrows()):
                if i >= self.N_max:
                    break
                x = row['center_x']
                y = row['center_y']
                xVel = row['xVelocity']
                yVel = row['yVelocity']
                # Correction: 'width' is vehicle length, 'height' is vehicle width.
                length = row['width']
                width = row['height']
                target = row['yTarget']
                vehicle_id = int(row['id'])
                self.kinematics[i] = torch.tensor([x, y, xVel, yVel, length, width, vehicle_id, target], dtype=torch.float32)
                self.exists[i] = True
                self.vehicle_ids[i] = vehicle_id
                if self.generation_mode:
                    is_agent = False 
                else:
                    # Get 'right-half' vehicle mask. Set these vehicles as 'Background vehicle'
                    right_ = x >= self.road_end - (self.road_end - self.road_start)*0.75 if self.road_end > self.road_start else\
                                x <= self.road_end + (self.road_start - self.road_end)*0.75
                    is_agent = False if right_ else bool(row['agent_status']) 
                self.agent_mask[i] = is_agent
                # self.features[i] = torch.tensor([length, width, target, 1.0 if is_agent else 0.0], dtype=torch.float32)
            self.expert_mask = ~self.agent_mask
            self.expert_frame_idx = 0
            self.cumu_id = first_frame.id.max()
        # get the first kinematics array
        self.history[-1] = self.kinematics
        # get static states
        self.lane_markers_tensor = torch.full((10,), float('nan')) 
        self.lane_markers_tensor[:len(self.lanemarkers)] = torch.tensor(self.lanemarkers).clone()
        self.boundary_lines_tensor = torch.tensor(self.boundarylines, dtype=torch.float32) if self.boundarylines.size else torch.tensor([])
        # get the initial states
        return self._get_obs()

    def step(self, actions):
        """
        In each simulation step:
          1. For background vehicles (agent_mask == False) in the current expert frame,
             if a vehicle no longer appears (its id is missing in the current expert frame),
             it is considered to have exited the road; its kinematics, exist mask, and agent mask are cleared.
          2. For background vehicles that remain, override their actions with the expert data accelerations.
          3. Then update the physics, remove vehicles exceeding road limits, and spawn new vehicles as before.
          4. Finally, advance the expert frame index by int(self.dt * 25).
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
            
        dt = self.dt
        exists_float = self.exists.unsqueeze(1).float()  # (N_max, 1)
        positions = self.kinematics[:, 0:2]
        velocities = self.kinematics[:, 2:4]
        dims = self.kinematics[:, 4:6]  # vehicle length and width (constant)
        ids = self.kinematics[:, 6].unsqueeze(1)  # vehicle ids
        targets = self.kinematics[:, 7].unsqueeze(1)
        
        '''
        Stage I: Update Agent & BV Kinematics 
        '''
        # If expert data is available, process the current expert frame.
        if self.expert_data is not None and self.expert_frame_idx < len(self.expert_data['frames']):
            current_expert_frame = self.expert_data['frames'][self.expert_frame_idx]
            # Build a set of vehicle ids present in the expert frame.
            active_ids_expert = set(current_expert_frame['id'].values)
            # For each background (non-agent) vehicle currently in simulation,
            # if its id is not in the expert frame, consider it to have exited.
            for i in range(self.N_max):
                if self.exists[i] and not self.agent_mask[i]:
                    current_id = int(self.vehicle_ids[i].item())
                    if current_id not in active_ids_expert:
                        self.kinematics[i] = float('nan')
                        self.exists[i] = False
                        self.agent_mask[i] = False
                        self.vehicle_ids[i] = -1
            # For each background vehicle still present, override its acceleration.
            for i in range(self.N_max):
                if (self.exists[i] and not self.agent_mask[i]) or self.demo_mode:
                    current_id = int(self.vehicle_ids[i].item())
                    row = current_expert_frame[current_expert_frame['id'] == current_id]
                    if not row.empty:
                        row_val = row.iloc[0]
                        actions[i] = torch.tensor([row_val['xAcceleration'], row_val['yAcceleration']], dtype=torch.float32)
            # Note: We do not increment expert_frame_idx here.


        new_positions = positions + velocities * dt + 0.5 * actions * (dt ** 2)
        new_velocities = velocities + actions * dt
        updated_state = torch.cat([new_positions, new_velocities, dims, ids, targets], dim=1)
        self.kinematics = torch.where(exists_float.bool(), updated_state, self.kinematics)

        '''
        Stage II: Remove Exiting Vehicles
        '''
        # Remove vehicles that exceed the road's end (assumed along the x-axis).
        exceed = self.kinematics[:, 0]-0.5*self.kinematics[:, 4] > self.road_end if self.road_end > self.road_start else self.kinematics[:, 0]+0.5*self.kinematics[:, 4] < self.road_end
        self.kinematics[exceed] = float('nan')
        self.exists[exceed] = False
        self.agent_mask[exceed] = False
        self.dead.update(self.vehicle_ids[exceed].tolist())
        self.vehicle_ids[exceed] = -1
        # self.features[exceed] = float('nan')
        
        # print (self.kinematics[:,0])
        
        '''
        Stage III: Handle Colliding Agents
        '''
        # Convert to background vehicles that collide with boundaries.
        if self.boundarylines.size == 2:
            lower_bound, upper_bound = self.boundarylines
            collide = (self.kinematics[:, 1] < lower_bound) | (self.kinematics[:, 1] > upper_bound)
            self.kinematics[collide, 2:4] = 0
            self.agent_mask[collide] = False

        '''
        Stage IV: Spawn New Vehicles
        '''
        if not self.generation_mode:
            # Spawn new vehicles from expert data into empty slots, if not already present.
            if self.expert_data is not None and self.expert_frame_idx < len(self.expert_data['frames']):
                expert_frame = self.expert_data['frames'][self.expert_frame_idx]
                active_ids = set(self.vehicle_ids[self.exists].tolist())
                for idx, row in expert_frame.iterrows():
                    vehicle_id = int(row['id'])
                    if vehicle_id in active_ids or vehicle_id in self.dead:
                        continue
                    # add a screen-line
                    empty_slots = torch.nonzero(~self.exists, as_tuple=False)
                    if empty_slots.numel() > 0:
                        slot = int(empty_slots[0])
                        x = row['center_x']
                        y = row['center_y']
                        xVel = row['xVelocity']
                        yVel = row['yVelocity']
                        # Correction: 'width' is vehicle length, 'height' is vehicle width.
                        length = row['width']
                        width = row['height']
                        target = row['yTarget']
                        # Screenline check
                        screenline_check =  x + 0.5*length >= self.road_start if self.road_start < self.road_end else x - 0.5*length <= self.road_start
                        _ = print (row['id']) if not screenline_check else 0
                        # if not screenline_check:
                        #     continue
                        # Spawn this vehicle
                        self.kinematics[slot] = torch.tensor([x, y, xVel, yVel, length, width, vehicle_id, target], dtype=torch.float32)
                        self.exists[slot] = True
                        is_agent = bool(row['agent_status'])
                        self.agent_mask[slot] = is_agent
                        # self.features[slot] = torch.tensor([length, width, target, 1.0 if is_agent else 0.0], dtype=torch.float32)
                        self.vehicle_ids[slot] = vehicle_id
                # (Do not increment here either.)
            
            # Advance expert_frame_idx based on self.dt.
            if self.expert_data is not None:
                skip = int(self.dt * 25)  # For example, dt=0.2 -> skip=5 frames.
                self.expert_frame_idx += skip
        
        else:
            '''
            Implement generation spawning     
            
            '''
            # Spawn new vehicles for generator into empty slots, if not already present.
            empty_slots = torch.nonzero(~self.exists, as_tuple=False) 
            for l, lcenter in enumerate(self.lanecenters):
                ar = self.lane_arrival_rate[l] 
                p_spawn = 1 - np.exp(-ar * dt)
                if p_spawn < np.random.rand():
                    continue
                # Sample vehicle type
                truck = np.random.rand() < self.lane_truck_ratio[l]
                # Get vehicle parameters
                length = np.random.normal(15.7, 2.5) if truck else np.random.normal(4.6, 0.3) 
                width = np.random.normal(2.5, 0.07) if truck else np.random.normal(1.8, 0.01)
                # Get initial spawn positions
                x = self.road_start - 0.5*length if self.road_start < self.road_end else self.road_start + 0.5*length
                y = lcenter + np.random.normal(scale=0.2)
                xVel = self.lane_avg_speed[l]
                yVel = 0
                vehicle_id = self.cumu_id + 1
                target = np.random.choice(self.lanecenters, p=self.lane_change_ratio[l])
                # Kinematic tensor of newly spawned vehicle
                spawned = torch.tensor([x, y, xVel, yVel, length, width, vehicle_id, target], dtype=torch.float32)
                # Check if feasible to spawn this vehicle otherwise collide
                exist_kinematics = self.kinematics[self.exists] 
                spawn_collide = self.check_collision(spawned, exist_kinematics)
                # If collide, don't spawn
                if spawn_collide:
                    continue
                # Otherwise, spawn this vehicle
                slot = int(empty_slots[0])
                self.kinematics[slot] = spawned
                self.exists[slot] = True
                is_agent = True
                self.agent_mask[slot] = is_agent
                # self.features[slot] = torch.tensor([length, width, target, 1.0], dtype=torch.float32)
                self.vehicle_ids[slot] = vehicle_id
                self.cumu_id += 1
            # Advance expert_frame_idx based on self.dt.
            if self.expert_data is not None:
                skip = int(self.dt * 25)  # For example, dt=0.2 -> skip=5 frames.
                self.expert_frame_idx += skip

        # Update the history states / memories
        self.history[:-1] = self.history[1:].clone()
        self.history[-1] = self.kinematics.clone()
        
        # Fetch the reward
        reward = 0.0  # placeholder reward
        done = False
        info = {}
        return self._get_obs(), reward, done, info
    
    def check_collision(self, spawned, exist_kinematics):
        """
        Checks if a newly spawned vehicle collides with any of the existing vehicles.
        
        Parameters:
            spawned (torch.Tensor): A tensor of shape (8,) with elements:
                                    [x, y, xVelocity, yVelocity, length, width, vehicle_id, target].
            exist_kinematics (torch.Tensor): A tensor of shape (n, 7) with each row:
                                    [x, y, xVelocity, yVelocity, length, width, id].
        
        Returns:
            bool: True if there is a collision, False otherwise.
        """
        # For the newly spawned vehicle:
        spawn_x, spawn_y, _, _, spawn_length, spawn_width = spawned[:6]
        spawn_x_min = spawn_x - spawn_length / 2
        spawn_x_max = spawn_x + spawn_length / 2
        spawn_y_min = spawn_y - spawn_width / 2
        spawn_y_max = spawn_y + spawn_width / 2

        # For the existing vehicles:
        exist_x = exist_kinematics[:, 0]
        exist_y = exist_kinematics[:, 1]
        exist_length = exist_kinematics[:, 4]
        exist_width = exist_kinematics[:, 5]
        
        exist_x_min = exist_x - exist_length / 2
        exist_x_max = exist_x + exist_length / 2
        exist_y_min = exist_y - exist_width / 2
        exist_y_max = exist_y + exist_width / 2
        
        # Check for overlap in the x direction and y direction:
        overlap_x = (spawn_x_min < exist_x_max) & (spawn_x_max > exist_x_min)
        overlap_y = (spawn_y_min < exist_y_max) & (spawn_y_max > exist_y_min)
        
        # A collision occurs if there is overlap in both directions.
        collision = (overlap_x & overlap_y).any().item()  # .item() converts the tensor scalar to a Python bool
        return collision

    def _get_obs(self):
        """
        Returns the observation dictionary including:
         - time_dependent: (T, N_max, 4)  <-- only x, y, xVelocity, and yVelocity
         - time_independent: (N_max, 3)
         - lane_markers: lane marker positions (tensor)
         - boundary_lines: boundary line positions (tensor)
         - mask: vehicle existence mask
         - agent_mask: agent-controlled vehicle mask
        """
        
        time_dependent = self.history[:, :, (0,1,2,3,4,5,7)].clone()
        return {
            'time_dependent': time_dependent,
            # 'time_independent': self.features.clone(),
            'lane_markers': self.lane_markers_tensor,
            'boundary_lines': self.boundary_lines_tensor,
            # 'mask': self.exists.float().clone(),
            'agent_mask': self.agent_mask.float().clone()
        }

    def render(self, mode='human'):
        """
        Render the current state.
        The road is drawn as a rectangle using road_start and road_end along x,
        with y-extent determined by boundary_lines.
        Lane markers (if provided) are drawn as dashed lines.
        Agent vehicles are rendered in red; background vehicles in blue.
        """
        plt.figure(figsize=(12, 3))
        ax = plt.gca()
        
        road_length = self.road_end - self.road_start
        if self.boundarylines.size == 2:
            lower_bound, upper_bound = self.boundarylines
        else:
            lower_bound, upper_bound = 0, 10
        road_height = upper_bound - lower_bound
        
        road_rect = plt.Rectangle((self.road_start, lower_bound), road_length, road_height, color='gray', alpha=0.3)
        ax.add_patch(road_rect)
        
        ax.plot([self.road_start, self.road_end], [lower_bound, lower_bound], color='black', linewidth=2)
        ax.plot([self.road_start, self.road_end], [upper_bound, upper_bound], color='black', linewidth=2)
        
        for marker in self.lanemarkers:
            ax.plot([self.road_start, self.road_end], [marker, marker], color='white', linestyle='--', linewidth=1)
        
        kinematics_np = self.kinematics.cpu().numpy()
        exists_np = self.exists.cpu().numpy()
        agent_mask_np = self.agent_mask.cpu().numpy()
        for i in range(self.N_max):
            if exists_np[i]:
                x = kinematics_np[i, 0]
                y = kinematics_np[i, 1]
                length = kinematics_np[i, 4]
                width = kinematics_np[i, 5]
                color = 'red' if agent_mask_np[i] else 'blue'
                rect = plt.Rectangle((x - length/2, y - width/2), length, width, color=color)
                ax.add_patch(rect)
                ax.annotate(kinematics_np[i, -2], kinematics_np[i, :2])
        
        _ = ax.set_xlim(self.road_start, self.road_end) if self.road_start < self.road_end else ax.set_xlim(self.road_end, self.road_start)
        ax.set_ylim(upper_bound + 5, lower_bound - 5)
        ax.set_xlabel('Position along road (x)')
        ax.set_ylabel('Lateral position (y)')
        # ax.set_title('Highway Vehicle Simulation')
        ax.set_aspect('equal')
        plt.show()




# ===========================
# Example usage:
# ===========================
if __name__ == '__main__':
    # Assuming expert_data is provided by your version of convert_highd_sample_to_gail_expert()
    # For example:
    # expert_data = convert_highd_sample_to_gail_expert('/mnt/data/highd_sample.csv', '/mnt/data/25_tracksMeta.csv', forward=True, p_agent=0.5)
    
    # Create the environment and set expert data.
    env = HighwayEnv(p_agent=0.5)
    # Uncomment and update the following line when expert_data is available:
    # env.set_expert_data(expert_data)
    
    obs = env.reset()
    print("Initial observation:")
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    # Example: apply an action of [1.0, 0.5] for every vehicle slot.
    actions = torch.full((env.N_max, 2), 1.0)
    actions[:, 1] = 0.5
    obs, reward, done, info = env.step(actions)
    
    env.render()
    
    print("\nSample kinematics from the latest frame:")
    print(env.kinematics)