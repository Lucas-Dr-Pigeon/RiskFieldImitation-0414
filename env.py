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
    lanecenters = np.concatenate((boundarylines[:1], lanemarkers, boundarylines[-1:]))
    
    # Compute vehicle center positions
    df['center_x'] = df['x'] + 0.5 * df['width']
    df['center_y'] = df['y'] + 0.5 * df['height']

    # Get road start:
    road_start = df['center_x'].min() - 10.0 if forward else df['center_x'].max() + 10.0 
    road_end = df['center_x'].max() + 10.0 if forward else df['center_x'].min() - 10.0
    
    # # Resample frames: select frames where frame % 5 == 0 (0.2 sec intervals)
    # df = df[df['frame'] % 5 == 0]

    # Get each vehicles' target lane centers
    veh_last_y = df.sort_values('frame').groupby('id').last()['y']
    laneindices = np.argmin( np.abs(lanecenters - veh_last_y.values.reshape(-1,1)), axis=1)
    last_lanecenters = lanecenters[laneindices]
    veh_last_lanecenters = pd.Series(last_lanecenters, index=veh_last_y.index)
    df['yTarget'] = df['id'].map(veh_last_lanecenters)
    
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
    
    
    expert_data = {
        'frames': expert_frames,
        'lanemarkers': lanemarkers,
        'boundarylines': boundarylines,
        'lanecenters': lanecenters,
        'start': road_start,
        'end': road_end,
    }
    
    return expert_data, df

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
                 arrival_rate=0.1,
                 avg_speed=20.0,
                 lanemarkers=[34.69,38.42],
                 boundarylines=[30.6,42.63],
                 lanechange_intensity=[[0.8,0.15,0.05],[0.1,0.8,0.1],[0.05,0.15,0.8]],
                 p_agent=0.5):
        super(HighwayEnv, self).__init__()
        
        # Road parameters (to be updated via expert data if available)
        self.road_start = road_start
        self.road_end = road_end
        self.lanemarkers = lanemarkers     # Lane marker positions (y-values)
        self.boundarylines = boundarylines      # Two values: [lower_boundary, upper_boundary]
        self.lanecenters = np.concatenate((boundarylines[:1], lanemarkers, boundarylines[-1:]))   # Get the lane center positions
        
        self.N_max = N_max
        self.dt = dt
        self.T = T
        self.arrival_rate = arrival_rate
        self.avg_speed = avg_speed
        self.lanechange_intensity = lanechange_intensity
        self.p_agent = p_agent    # probability for a vehicle to be agent-controlled
        
        # Kinematic state now has 7 elements: [x, y, xVelocity, yVelocity, length, width, id]
        self.M1 = 8
        # Time-independent features remain: [vehicle_length, vehicle_width, agent_indicator, target]
        self.M2 = 4

        # Define observation space.
        # time_dependent observations now only include the first 4 columns.
        self.observation_space = gym.spaces.Dict({
            'time_dependent': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.T, self.N_max, 4), dtype=np.float32),
            'time_independent': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.N_max, self.M2), dtype=np.float32),
            'lane_markers': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),  # arbitrary max shape
            'boundary_lines': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'mask': gym.spaces.Box(low=0, high=1, shape=(self.N_max,), dtype=np.float32),
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
        self.features = torch.full((self.N_max, self.M2), float('nan'))
        self.history = torch.full((self.T, self.N_max, self.M1), float('nan'))
        
        # Track vehicle ids separately for easier lookup.
        self.vehicle_ids = torch.full((self.N_max,), -1, dtype=torch.int64)
        
        # Placeholder for expert data.
        self.expert_data = None
        self.expert_frame_idx = 0

        # check each lane

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
        self.features = torch.full((self.N_max, self.M2), float('nan'))
        self.vehicle_ids = torch.full((self.N_max,), -1, dtype=torch.int64)
        self.history = torch.full((self.T, self.N_max, self.M1), float('nan'))
        
        if self.expert_data is not None:
            first_frame = self.expert_data['frames'][0]
            self.road_start = self.expert_data['start']
            self.road_end = self.expert_data['end']
            self.lanemarkers = self.expert_data['lanemarkers']
            self.boundarylines = self.expert_data['boundarylines']
            self.lanecenters = self.expert_data['lanecenters']
            
            num_vehicles = min(len(first_frame), self.N_max)
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
                is_agent = bool(row['agent_status'])
                self.agent_mask[i] = is_agent
                self.features[i] = torch.tensor([length, width, target, 1.0 if is_agent else 0.0], dtype=torch.float32)
                self.vehicle_ids[i] = vehicle_id
            self.expert_frame_idx = 0
        else:
            self.kinematics[0] = torch.tensor([self.road_start,
                                                (self.road_end - self.road_start) / 2,
                                                self.default_speed,
                                                0.0,
                                                self.default_length,
                                                self.default_width,
                                                0], dtype=torch.float32)
            self.exists[0] = True
            is_agent = (np.random.rand() < self.p_agent)
            self.agent_mask[0] = is_agent
            self.features[0] = torch.tensor([self.default_length, self.default_width, target, 1.0 if is_agent else 0.0], dtype=torch.float32)
            self.vehicle_ids[0] = 0
        
        self.history[-1] = self.kinematics

        # get the last vehicles 

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
                if self.exists[i] and not self.agent_mask[i]:
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
        
        # Remove vehicles that exceed the road's end (assumed along the x-axis).
        exceed = self.kinematics[:, 0] > self.road_end
        self.kinematics[exceed] = float('nan')
        self.exists[exceed] = False
        self.agent_mask[exceed] = False
        self.vehicle_ids[exceed] = -1
        self.features[exceed] = float('nan')

        # print (self.kinematics[:,0])
        
        # Remove vehicles that collide with boundaries.
        if self.boundarylines.size == 2:
            lower_bound, upper_bound = self.boundarylines
            collide = (self.kinematics[:, 1] < lower_bound) | (self.kinematics[:, 1] > upper_bound)
            self.kinematics[collide, 2:4] = float('nan')
            self.agent_mask[collide] = False
        
        # Spawn new vehicles from expert data into empty slots, if not already present.
        if self.expert_data is not None and self.expert_frame_idx < len(self.expert_data['frames']):
            expert_frame = self.expert_data['frames'][self.expert_frame_idx]
            active_ids = set(self.vehicle_ids[self.exists].tolist())
            for idx, row in expert_frame.iterrows():
                vehicle_id = int(row['id'])
                if vehicle_id in active_ids:
                    continue
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
                    self.kinematics[slot] = torch.tensor([x, y, xVel, yVel, length, width, vehicle_id, target], dtype=torch.float32)
                    self.exists[slot] = True
                    is_agent = bool(row['agent_status'])
                    self.agent_mask[slot] = is_agent
                    self.features[slot] = torch.tensor([length, width, target, 1.0 if is_agent else 0.0], dtype=torch.float32)
                    self.vehicle_ids[slot] = vehicle_id
            # (Do not increment here either.)
        
        # Advance expert_frame_idx based on self.dt.
        if self.expert_data is not None:
            skip = int(self.dt * 25)  # For example, dt=0.2 -> skip=5 frames.
            self.expert_frame_idx += skip
        
        # self.history = torch.cat([self.history[1:], self.kinematics.unsqueeze(0)], dim=0)
        self.history[:-1] = self.history[1:].clone()
        self.history[-1] = self.kinematics.clone()
        
        reward = 0.0  # placeholder reward
        done = False
        info = {}
        return self._get_obs(), reward, done, info

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
        lane_markers_tensor = torch.tensor(self.lanemarkers, dtype=torch.float32) if self.lanemarkers.size else torch.tensor([])
        boundary_lines_tensor = torch.tensor(self.boundarylines, dtype=torch.float32) if self.boundarylines.size else torch.tensor([])
        time_dependent = self.history[:, :, 0:4].clone()
        return {
            'time_dependent': time_dependent,
            'time_independent': self.features.clone(),
            'lane_markers': lane_markers_tensor,
            'boundary_lines': boundary_lines_tensor,
            'mask': self.exists.float().clone(),
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
        
        car_length = 5.0
        car_width = 2.0
        kinematics_np = self.kinematics.cpu().numpy()
        exists_np = self.exists.cpu().numpy()
        agent_mask_np = self.agent_mask.cpu().numpy()
        for i in range(self.N_max):
            if exists_np[i]:
                x = kinematics_np[i, 0]
                y = kinematics_np[i, 1]
                color = 'red' if agent_mask_np[i] else 'blue'
                rect = plt.Rectangle((x - car_length/2, y - car_width/2), car_length, car_width, color=color)
                ax.add_patch(rect)
        
        ax.set_xlim(self.road_start - 10, self.road_end + 10)
        ax.set_ylim(upper_bound + 5, lower_bound - 5)
        ax.set_xlabel('Position along road (x)')
        ax.set_ylabel('Lateral position (y)')
        ax.set_title('Highway Vehicle Simulation')
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