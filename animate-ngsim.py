import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import torch
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from tqdm import tqdm, trange
import time
import argparse

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="./data/ngsim/i80 sample.csv")
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--start', type=int, default=-1)
    parser.add_argument('--xlim', type=float, nargs='+', default=(-1,1e11))
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--ttc1-coef', type=float, default=1.214)
    parser.add_argument('--ttc2-coef', type=float, default=1.214)
    parser.add_argument('--ttc1-bubble', type=float, default=0)
    parser.add_argument('--ttc2-bubble', type=float, default=0)
    parser.add_argument('--ttc1-thres', type=float, default=1.5)
    parser.add_argument('--ttc2-thres', type=float, default=5.0)
    parser.add_argument('--mode', type=str, default='circle')
    parser.add_argument('--ttc-dist', type=float, default=50)
    parser.add_argument('--grid-dist', type=float, default=1000)
    args = parser.parse_args()
    return args

def animate_ngsim_slider(data, lanes, start=-1, batch_size=100, coef1=1.414, coef2=1.414, bubble1=0, bubble2=0, T=1, thres1=1.5, thres2=1.5, mode='circle', grid_dist=1000, ttc_dist=50):
    df = data
    start = max(0, start)
    frames = np.arange(data.frame.min(), data.frame.max(), T).round(1)  # Take every T-th frame
    max_frame_idx = len(frames) - 1  # Max frame index for slider

    # Get upper and lower lane markers
    upper, lower = lanes

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.subplots_adjust(bottom=0.2)  # Make space for slider

    ax.set_xlim(df["x"].min(), df["x"].max())
    ax.set_ylim(df["y"].min() - 5, df["y"].max() + 5)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Vehicle Motion and TTC Warnings")
    ax.set_aspect("equal")  # Ensure x and y scales are equal

    # Create a slider axis and widget
    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor="gray")  
    frame_slider = Slider(slider_ax, "Frame", 0, max_frame_idx, valinit=0, valstep=1, color="black")

    vehicle_patches = []
    annotations = []
    lines = []
    
    running = True  # Auto-loop state (True = running, False = paused)
    last_slider_update = time.perf_counter()  # Track last time slider was changed
    frame_idx = start  # Current animation frame index
    last_update_time = time.perf_counter() # Ensure smooth updates

    def update(val, force=False):
        """Update function for animation & slider control"""
        nonlocal frame_idx, last_slider_update, last_update_time

        if not force and time.perf_counter() - last_update_time < 0.05:
            plt.pause(0.05)
        
        last_update_time = time.perf_counter()

        frame_idx = int(val) % len(frames)  # Ensure index stays within bounds
        frame_num = frames[frame_idx]
        ax.set_title(f"Frame {frame_num}")

        frame_data = df[df["frame"] == frame_num]

        for rect in vehicle_patches:
            rect.remove()
        vehicle_patches.clear()
        for text in annotations:
            text.remove()
        annotations.clear()
        for line in lines:
            line.remove()
        lines.clear()

        # Retrieve time to collision values

        ttc1, ego_indices, foe_indices, _, _, _, _ = calculate_ttc_ngsim(frame_data, batch_size=batch_size, bubble=bubble1, coef=coef1, mode=mode, grid_dist=grid_dist, ttc_dist=ttc_dist)
        ttc2, _, _, _, _, _, _ = calculate_ttc_ngsim(frame_data, batch_size=batch_size, bubble=bubble2, coef=coef2, mode=mode, grid_dist=grid_dist, ttc_dist=ttc_dist)
        assert len(ttc1) == len(ttc2)

        ego_indices = ego_indices.cpu().numpy().astype(int)
        foe_indices = foe_indices.cpu().numpy().astype(int)

        #
        unique_vehicles = frame_data.id.unique()
        vcolors = dict(zip(unique_vehicles, np.full(len(unique_vehicles),'white')))

        # print ( [ (ego_indices[i], foe_indices[i]) for i in range(len(ttc1))])

        for i in range(len(ttc1)):
            if ttc1[i] < thres1:
                ego_vehicle = frame_data.iloc[ego_indices[i]]
                foe_vehicle = frame_data.iloc[foe_indices[i]]
                line, = ax.plot(
                    [ego_vehicle["x"], foe_vehicle["x"]],
                    [ego_vehicle["y"], foe_vehicle["y"]],
                    color='red', lw=2, alpha=0.7, zorder=0
                )
                vcolors[ego_vehicle.id] = 'red'
                vcolors[foe_vehicle.id] = 'red'
                lines.append(line)

            elif ttc2[i] < thres2:
                ego_vehicle = frame_data.iloc[ego_indices[i]]
                foe_vehicle = frame_data.iloc[foe_indices[i]]
                line, = ax.plot(
                    [ego_vehicle["x"], foe_vehicle["x"]],
                    [ego_vehicle["y"], foe_vehicle["y"]],
                    color='orange', lw=2, alpha=0.7, zorder=0
                )
                vcolors[ego_vehicle.id] = 'orange' if vcolors[ego_vehicle.id] != 'red' else 'red'
                vcolors[foe_vehicle.id] = 'orange' if vcolors[foe_vehicle.id] != 'red' else 'red'
                lines.append(line)

                

        for _, row in frame_data.iterrows():
            rect = patches.Rectangle(
                (row["x"]-0.5*row["length"], row["y"]-0.5*row["width"]), row["length"], row["width"],
                ec='black', fc=vcolors[row.id], alpha=1.0
            )
            ax.add_patch(rect)
            vehicle_patches.append(rect)

            text = ax.text(
                row["x"]+0.5*row["length"]-0.5, row["y"]+0.5*row["width"],
                str(int(row["id"])), color='black', fontsize=8, ha="center", va="bottom", zorder=3
            )
            annotations.append(text)

        for l, lpos in enumerate(upper):
            lstyle = 'solid' if l == 0 or l == len(upper) - 1 else '--'
            lwidth = 1.2 if l == 0 or l == len(upper) - 1 else 1
            ax.axhline(lpos, linestyle=lstyle, color='grey', linewidth=lwidth)

        for l, lpos in enumerate(lower):
            lstyle = 'solid' if l == 0 or l == len(lower) - 1 else '--'
            lwidth = 1.2 if l == 0 or l == len(lower) - 1 else 1
            ax.axhline(lpos, linestyle=lstyle, color='grey', linewidth=lwidth)

        # Sync slider with animation (only if running)
        if running:
            frame_slider.set_val(frame_idx)

        fig.canvas.draw_idle()

    def on_slider_change(val):
        """Pause auto-loop when the user moves the slider"""
        nonlocal running, last_slider_update
        running = False  # Pause animation when slider is moved
        last_slider_update = time.perf_counter()   # Track the time of slider movement
        update(val)

    def check_resume():
        """Resume animation if the user hasn't moved the slider for 2 seconds"""
        nonlocal running, last_slider_update
        if not running and (time.perf_counter() - last_slider_update > 0.01):  # 2-second delay
            running = True

    def animate(_):
        """Auto-loop function for animation"""
        nonlocal frame_idx
        check_resume()  # Check if we should resume auto-loop
        if running:
            frame_idx = (frame_idx + 1) % len(frames)  # Increment frame index
            frame_slider.set_val(frame_idx)  # Sync slider with animation

    frame_slider.on_changed(on_slider_change)

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, repeat=True)
    update(start)  # Start at frame 0
    plt.show()


def get_interaction_vehicle_indices(tensor, max_dist=50, grid_dist=1000):
    """
    Finds nearby vehicle IDs within a distance threshold using a grid-based approach.
    Returns a tensor of shape (N, M), where N = input size and M = max neighbors (padded with 0).
    
    Args:
        tensor (torch.Tensor): Shape (N, 4) where:
            - Column 0: Frame number
            - Column 1: Vehicle ID
            - Column 2: X coordinate
            - Column 3: Y coordinate
        distance_threshold (float): Distance threshold to consider as "nearby".
        max_neighbors (int): Maximum number of neighbors to keep per vehicle (padded with 0).

    Returns:
        torch.Tensor: Shape (N, M), where each row contains up to `M` nearest vehicle IDs.
    """
    frames = tensor[:, 0]  # Extract frame column
    positions = tensor[:, 2:4]  # Extract x, y coordinates
    vids = tensor[:, 1]
    lengths = tensor[:, -3]
    widths = tensor[:, -2]
    directions = tensor[:, -1]
    indices = torch.arange(tensor.shape[0], device=tensor.device)  # Create an index for vehicle IDs
    
    allow_reverse = True
     # **Step 1: Assign grid IDs (divide x-coordinates by 50m)**
    grid_x = (positions[:, 0] // grid_dist).int()  # Grid ID based on x position

    unique_grid = torch.sort(torch.unique(grid_x))[0]

    keys = []
    neighbors = []

    for grid in unique_grid:
        neighbor_grid_mask = (grid_x - grid ==1) | (grid_x == grid)

        neighbor_grid_positions = positions[neighbor_grid_mask]
        neighbor_grid_frames = frames[neighbor_grid_mask]
        neighbor_grid_indices = indices[neighbor_grid_mask]
        neighbor_grid_directions = directions[neighbor_grid_mask]
        neighbor_lengths = lengths[neighbor_grid_mask]
        neighbor_widths = widths[neighbor_grid_mask]

        _neighbor_grid_position_x = torch.abs(neighbor_grid_positions[:,0].unsqueeze(1) - neighbor_grid_positions[:,0].unsqueeze(0)) - 0.5*(neighbor_lengths.unsqueeze(1)+neighbor_lengths.unsqueeze(0))
        
        
        neighbor_grid_position_x = torch.clamp(_neighbor_grid_position_x, min=0)
        neighbor_dist_mask = neighbor_grid_position_x < max_dist

        # neighbor_dist_mask = (torch.norm(neighbor_grid_positions.unsqueeze(1) - neighbor_grid_positions.unsqueeze(0), dim=2) < max_dist) 
        frame_mask = neighbor_grid_frames.unsqueeze(1) == neighbor_grid_frames.unsqueeze(0)
        direction_mask = neighbor_grid_directions.unsqueeze(1) == neighbor_grid_directions.unsqueeze(0)
        neighbor_mask = neighbor_dist_mask & frame_mask & direction_mask
        n_shape = neighbor_mask.shape[0]
        neighbor_mask[range(n_shape), range(n_shape)] = 0
        
        idx_x, idx_y = torch.where(neighbor_mask)

        keys.append (neighbor_grid_indices[idx_x])
        neighbors.append (neighbor_grid_indices[idx_y])

    pairs = torch.stack((torch.cat(keys), torch.cat(neighbors)), dim=1)
    pairs = pairs if allow_reverse else torch.sort(pairs, dim=1)[0]
    unique_pairs = torch.unique(pairs, dim=0)

    return unique_pairs


def calculate_ttc_ngsim(data, batch_size=1000000, bubble=0, coef=1.414, mode='circle', grid_dist=500, ttc_dist=1000):
    M = batch_size
    unique_frames = data['frame'].unique()
    batched_frames = [unique_frames[i:i+M] for i in range(0, len(unique_frames), M)]
    # outer_indices = tracks.index.to_numpy()
    _ttcdata = np.zeros((len(data),16))
    # Process each batch
    for batch in batched_frames:
        batch_data = data[data['frame'].isin(batch)]
        _indices = batch_data.index.to_numpy()
        # Process batch_data...
        tracks_tensor = torch.tensor(batch_data.values, device='cuda')

        pairs = get_interaction_vehicle_indices(tracks_tensor[:,(1,0,3,4,-3,-2,-1)], max_dist=ttc_dist, grid_dist=grid_dist)

        ego_indices = pairs[:,0]
        foe_indices = pairs[:,1]

        egos = tracks_tensor[ego_indices]
        foes = tracks_tensor[foe_indices]

        (ego_ids, _, _, ego_xs, ego_ys, ego_xVel, ego_yVel, ego_xAccel, ego_yAccel, ego_length, ego_width) = egos[:,0:11].split(1, dim=1)
        (foe_ids, _, _, foe_xs, foe_ys, foe_xVel, foe_yVel, foe_xAccel, foe_yAccel, foe_length, foe_width) = foes[:,0:11].split(1, dim=1)

        ego_pos = torch.cat([ego_xs, ego_ys], dim=1)
        ego_vel = torch.cat([ego_xVel, ego_yVel], dim=1)
        ego_accel = torch.cat([ego_xAccel, ego_yAccel], dim=1)
        ego_dim = torch.cat([ego_length, ego_width], dim=1)

        foe_pos = torch.cat([foe_xs, foe_ys], dim=1)
        foe_vel = torch.cat([foe_xVel, foe_yVel], dim=1)
        foe_accel = torch.cat([foe_xAccel, foe_yAccel], dim=1)
        foe_dim = torch.cat([foe_length, foe_width], dim=1)

        if mode == 'circle':
            ttcs = get_2D_ttc_circles_cuda_batched(ego_pos, ego_vel, ego_dim, foe_pos, foe_vel, foe_dim, buffer_coef=coef, bubble=bubble) 
        elif mode == 'rectangle':
            ttcs = rectangle_modified_time_to_collision_batched(ego_pos, ego_vel, ego_accel, ego_dim, foe_pos, foe_vel, foe_accel, foe_dim, T=10, dt=0.1, bubble=bubble)
        

    return ttcs, ego_indices, foe_indices, _indices, 0, ego_ids, foe_ids

def get_2D_ttc_circles_cuda_batched(
    ego_pos, ego_vel, ego_dim,
    foe_pos, foe_vel, foe_dim,
    buffer_coef=np.sqrt(2),
    bubble=0,
    batch_size=None
):
    """
    Compute Time-To-Collision (TTC) using multiple-circle approximation with CUDA batch processing.
    Prevents CUDA OOM errors by processing in batches.
    
    Args:
        ego_pos: (N, 2) tensor - Ego vehicle positions (x, y).
        ego_vel: (N, 2) tensor - Ego vehicle velocities (Vx, Vy).
        ego_dim: (N, 2) tensor - Ego vehicle dimensions (Length, Width).
        foe_pos: (N, 2) tensor - Foe vehicle positions (x, y).
        foe_vel: (N, 2) tensor - Foe vehicle velocities (Vx, Vy).
        foe_dim: (N, 2) tensor - Foe vehicle dimensions (Length, Width).
        buffer_coef: Scaling factor for safety buffer (default sqrt(2)).
        batch_size: Number of samples per batch (auto-calculated if None).

    Returns:
        ttc: (N,) tensor - Minimum Time-To-Collision values for each vehicle pair.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer_coef=(torch.tensor(buffer_coef, device='cuda'))
    # Move input tensors to the correct device
    ego_pos = ego_pos.to(dtype=torch.float32, device=device)
    ego_vel = ego_vel.to(dtype=torch.float32, device=device)
    ego_dim = ego_dim.to(dtype=torch.float32, device=device)

    foe_pos = foe_pos.to(dtype=torch.float32, device=device)
    foe_vel = foe_vel.to(dtype=torch.float32, device=device)
    foe_dim = foe_dim.to(dtype=torch.float32, device=device)

    N = ego_pos.shape[0]

    # Auto-adjust batch size if not provided
    if batch_size is None:
        batch_size = min(500000, N)  # Default: 500K samples per batch
    batch_size = max(N, 1)

    num_batches = (N + batch_size - 1) // batch_size  # Compute number of batches
    ttc_results = torch.empty(N, device=device, dtype=torch.float32)  # Allocate space

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)

        # Slice batch
        ego_pos_batch = ego_pos[start_idx:end_idx]
        ego_vel_batch = ego_vel[start_idx:end_idx]
        ego_dim_batch = ego_dim[start_idx:end_idx]

        foe_pos_batch = foe_pos[start_idx:end_idx]
        foe_vel_batch = foe_vel[start_idx:end_idx]
        foe_dim_batch = foe_dim[start_idx:end_idx]

        # Compute TTC for batch
        ttc_batch = get_2D_ttc_circles_cuda(
            ego_pos_batch, ego_vel_batch, ego_dim_batch,
            foe_pos_batch, foe_vel_batch, foe_dim_batch,
            buffer_coef, bubble=bubble
        )

        # Store results
        ttc_results[start_idx:end_idx] = ttc_batch

    return ttc_results


def get_approximate_circles_cuda(pos, dim, device="cuda"):
    """
    Approximates a rectangular vehicle as a series of circles along its length.
    Returns a tensor of shape (N, max_circles, 2) containing circle centers.
    """
    x, y = pos[:, 0], pos[:, 1]
    L, W = dim[:, 0], dim[:, 1]

    # Number of circles (at least 1, up to L/W)
    num_circles = torch.ceil(L / W).to(torch.int32)
    max_circles = num_circles.max().item()

    # Compute x positions for circles
    start_x = x - L / 2 + W / 2
    end_x = x + L / 2 - W / 2
    step = (end_x - start_x) / (num_circles - 1).clamp(min=1)

    # Generate indices for evenly spaced circles
    indices = torch.arange(max_circles, device=device).unsqueeze(0)
    valid_mask = indices < num_circles.unsqueeze(1)

    # Compute circle positions
    x_positions = start_x.unsqueeze(1) + indices * step.unsqueeze(1)  # Shape: (N, max_circles)
    y_positions = y.unsqueeze(1).expand(-1, max_circles).clone()  # Keep y constant

    # Mask invalid circles
    x_positions[~valid_mask] = torch.nan
    y_positions[~valid_mask] = torch.nan

    return torch.stack([x_positions, y_positions], dim=-1)  # Shape: (N, max_circles, 2)


def get_approximate_circles_cuda(pos, dim, device="cuda"):
    """
    Approximates a rectangular vehicle as a series of circles along its length.
    Returns a tensor of shape (N, max_circles, 2) containing circle centers.
    """
    x, y = pos[:, 0], pos[:, 1]
    L, W = dim[:, 0], dim[:, 1]

    # Number of circles (at least 1, up to L/W)
    num_circles = torch.ceil(L / W).to(torch.int32)
    max_circles = num_circles.max().item()

    # Compute x positions for circles
    start_x = x - L / 2 + W / 2
    end_x = x + L / 2 - W / 2
    step = (end_x - start_x) / (num_circles - 1).clamp(min=1)

    # Generate indices for evenly spaced circles
    indices = torch.arange(max_circles, device=device).unsqueeze(0)
    valid_mask = indices < num_circles.unsqueeze(1)

    # Compute circle positions
    x_positions = start_x.unsqueeze(1) + indices * step.unsqueeze(1)  # Shape: (N, max_circles)
    y_positions = y.unsqueeze(1).expand(-1, max_circles).clone()  # Keep y constant

    # Mask invalid circles
    x_positions[~valid_mask] = torch.nan
    y_positions[~valid_mask] = torch.nan

    return torch.stack([x_positions, y_positions], dim=-1)  # Shape: (N, max_circles, 2)

def get_2D_ttc_circles_cuda(
    ego_pos, ego_vel, ego_dim,
    foe_pos, foe_vel, foe_dim,
    buffer_coef=torch.sqrt(torch.tensor(2.0, device='cuda')),
    bubble=0
):
    """
    Compute Time-To-Collision (TTC) between vehicles approximated as multiple circles using CUDA acceleration.
    Assumes zero acceleration.
    
    Args:
        ego_pos: (N, 2) tensor - Ego vehicle positions (x, y).
        ego_vel: (N, 2) tensor - Ego vehicle velocities (Vx, Vy).
        ego_dim: (N, 2) tensor - Ego vehicle dimensions (Length, Width).
        foe_pos: (N, 2) tensor - Foe vehicle positions (x, y).
        foe_vel: (N, 2) tensor - Foe vehicle velocities (Vx, Vy).
        foe_dim: (N, 2) tensor - Foe vehicle dimensions (Length, Width).
        buffer_coef: Scaling factor for safety buffer (default sqrt(2)).

    Returns:
        ttc: (N,) tensor - Minimum Time-To-Collision values for each vehicle pair.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure tensors are on GPU
    ego_pos = ego_pos.clone().detach().to(dtype=torch.float32, device=device)
    ego_vel = ego_vel.clone().detach().to(dtype=torch.float32, device=device)
    ego_dim = ego_dim.clone().detach().to(dtype=torch.float32, device=device)

    foe_pos = foe_pos.clone().detach().to(dtype=torch.float32, device=device)
    foe_vel = foe_vel.clone().detach().to(dtype=torch.float32, device=device)
    foe_dim = foe_dim.clone().detach().to(dtype=torch.float32, device=device)

    # Compute circle centers for each vehicle
    ego_circles = get_approximate_circles_cuda(ego_pos, ego_dim, device)  # (N, C, 2)
    foe_circles = get_approximate_circles_cuda(foe_pos, foe_dim, device)  # (N, C, 2)

    # Compute circle radii based on vehicle width
    ego_r = (ego_dim[:, 1] / 2) * buffer_coef + bubble  # Shape: (N,)
    foe_r = (foe_dim[:, 1] / 2) * buffer_coef + bubble # Shape: (N,)
    total_r = ego_r.unsqueeze(1).unsqueeze(1) + foe_r.unsqueeze(1).unsqueeze(1)  # (N, 1, 1)

    # Compute all pairwise circle distances
    ego_exp = ego_circles.unsqueeze(2)  # (N, C, 1, 2)
    foe_exp = foe_circles.unsqueeze(1)  # (N, 1, C, 2)

    rel_pos = ego_exp - foe_exp  # Shape: (N, C, C, 2)
    rel_vel = ego_vel.unsqueeze(1).unsqueeze(1) - foe_vel.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 2)

    # Compute quadratic coefficients
    A = (rel_vel[..., 0]**2 + rel_vel[..., 1]**2)  # (N, C, C)
    B = 2 * (rel_pos[..., 0] * rel_vel[..., 0] + rel_pos[..., 1] * rel_vel[..., 1])  # (N, C, C)
    C = (rel_pos[..., 0]**2 + rel_pos[..., 1]**2 - total_r**2)  # (N, C, C)

    # Compute discriminant
    discriminant = B**2 - 4 * A * C
    # valid_mask = discriminant >= 0  # Valid solutions where discriminant is non-negative

    # sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0))  # Avoid NaNs

    sqrt_disc = torch.sqrt(discriminant)
    t1 = (-B + sqrt_disc) / (2 * A) 
    t2 = (-B - sqrt_disc) / (2 * A) 

    # Find the smallest positive TTC
    t1[t1 < 0] = float('inf')
    t2[t2 < 0] = float('inf')

    ttc = torch.min(t1, t2)  # Minimum positive solution for each circle pair

    # overlap mask 
    overlap_mask = C < 0
    ttc[overlap_mask] = 0
    
    # Handle invalid cases correctly
    ttc[torch.isnan(ttc)] = float('inf')  # Replace NaNs
    min_ttc = torch.amin(ttc, dim=(1, 2))  # Min over all circle interactions

    # Mask invalid results
    min_ttc = torch.clamp(min_ttc, max=1e9)  # Large value for "no collision"

    return min_ttc

def rectangle_modified_time_to_collision_batched(ego_pos, ego_vel, ego_accel, ego_dim,
                                                 foe_pos, foe_vel, foe_accel, foe_dim, 
                                                 T=10, dt=0.1, batch_size=512, device="cuda", bubble=0):
    """
    Computes the modified time-to-collision (MTTC) for large N samples in batches.
    
    Args:
        ego_pos, ego_vel, ego_accel, ego_dim: (N, 2) tensors for ego vehicles
        foe_pos, foe_vel, foe_accel, foe_dim: (N, 2) tensors for foe vehicles
        T: Total simulation time
        dt: Time step
        batch_size: Number of samples per batch to reduce CUDA memory usage
        device: "cuda" or "cpu"

    Returns:
        mttc: (N,) tensor with time to first overlap, or 1e9 if no overlap occurs.
    """

    N = ego_pos.shape[0]  # Total number of samples

    batch_size = max(N, 1)
    num_batches = (N + batch_size - 1) // batch_size  # Compute number of batches
    
    # Initialize MTTC results
    mttc_results = torch.full((N,), 1e9, dtype=torch.float32, device=device)  # Default to 1e9

    for i in range(num_batches):
        # Define batch range
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, N)

        # Extract batch
        ego_pos_batch = ego_pos[start_idx:end_idx].to(device)
        ego_vel_batch = ego_vel[start_idx:end_idx].to(device)
        ego_accel_batch = ego_accel[start_idx:end_idx].to(device)
        ego_dim_batch = ego_dim[start_idx:end_idx].to(device)

        foe_pos_batch = foe_pos[start_idx:end_idx].to(device)
        foe_vel_batch = foe_vel[start_idx:end_idx].to(device)
        foe_accel_batch = foe_accel[start_idx:end_idx].to(device)
        foe_dim_batch = foe_dim[start_idx:end_idx].to(device)

        # Simulate vehicle positions
        _, pos1, pos2 = simulate_rectangles_torch(ego_pos_batch, ego_vel_batch, ego_accel_batch, ego_dim_batch,
                                                  foe_pos_batch, foe_vel_batch, foe_accel_batch, foe_dim_batch,
                                                  T=T, dt=dt, device=device)

        # Compute absolute distance between centers
        delta_pos = torch.abs(pos1 - pos2)  # (N, 2) absolute difference in x and y

        # Compute half-length sums
        half_sizes = (ego_dim_batch.to('cuda')+ foe_dim_batch.to('cuda')+ bubble*2) / 2  # (N, 2) sum of half-lengths

        # Condition: overlap occurs if delta_pos < half_sizes in both x and y
        overlap = (delta_pos < half_sizes.unsqueeze(1))
        overlaps = overlap[:,:,0]* overlap[:,:,1]

        lN, lT =  overlaps.shape

        # Convert boolean to integer mask (1 where overlap occurs, 0 otherwise)
        overlap_mask = overlaps.to(dtype=torch.int32)

        # Create a time index tensor (broadcasted across N)
        time_index = torch.arange(lT, device=overlaps.device).expand(lN, lT)

        # Mask out non-overlapping values by setting them to a large value
        time_index_masked = torch.where(overlap_mask == 1, time_index, lT)  # If no overlap, set to T (max value)

        # Get the minimum (first occurrence) along T-axis
        collision_frame = torch.min(time_index_masked, dim=1).values  # Shape: (N,)

        # Compute MTTC
        mttc_batch = collision_frame * dt
        mttc_batch[mttc_batch >= T] = 1e9  # Set high value if no collision

        # Store results
        mttc_results[start_idx:end_idx] = mttc_batch

    return mttc_results

def simulate_rectangles_torch(ego_pos, ego_vel, ego_accel, ego_dim,
                              foe_pos, foe_vel, foe_accel, foe_dim, 
                              T=10, dt=0.1, device="cuda"):

    # Move computations to the specified device (CPU/GPU)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create time vector
    time_steps = torch.arange(0, T, dt, device=device)

    # Expand time for vectorized computation
    time_expanded = time_steps.view(-1, 1, 1)  # Shape: (num_steps, 1, 1)

    # Compute velocities over time (V(t) = V0 + A*t) while preventing reversal
    ego_vel_t = ego_vel + ego_accel * time_expanded
    foe_vel_t = foe_vel + foe_accel * time_expanded

    # Apply velocity constraints: If velocity reaches 0, it stays at 0
    ego_vel_t[:, :, 0] = torch.where((ego_vel[:, 0] > 0) & (ego_vel_t[:, :, 0] < 0), 0, ego_vel_t[:, :, 0])
    ego_vel_t[:, :, 1] = torch.where((ego_vel[:, 1] > 0) & (ego_vel_t[:, :, 1] < 0), 0, ego_vel_t[:, :, 1])
    ego_vel_t[:, :, 0] = torch.where((ego_vel[:, 0] < 0) & (ego_vel_t[:, :, 0] > 0), 0, ego_vel_t[:, :, 0])
    ego_vel_t[:, :, 1] = torch.where((ego_vel[:, 1] < 0) & (ego_vel_t[:, :, 1] > 0), 0, ego_vel_t[:, :, 1])

    foe_vel_t[:, :, 0] = torch.where((foe_vel[:, 0] > 0) & (foe_vel_t[:, :, 0] < 0), 0, foe_vel_t[:, :, 0])
    foe_vel_t[:, :, 1] = torch.where((foe_vel[:, 1] > 0) & (foe_vel_t[:, :, 1] < 0), 0, foe_vel_t[:, :, 1])
    foe_vel_t[:, :, 0] = torch.where((foe_vel[:, 0] < 0) & (foe_vel_t[:, :, 0] > 0), 0, foe_vel_t[:, :, 0])
    foe_vel_t[:, :, 1] = torch.where((foe_vel[:, 1] < 0) & (foe_vel_t[:, :, 1] > 0), 0, foe_vel_t[:, :, 1])

    # Compute positions over time: x(t) = x0 + integral of velocity
    ego_positions = ego_pos + torch.cumsum(ego_vel_t * dt, dim=0)
    foe_positions = foe_pos + torch.cumsum(foe_vel_t * dt, dim=0)

    return time_steps, torch.permute(ego_positions, (1,0,2)), torch.permute(foe_positions, (1,0,2))

def reformat_ngsim(data, metric=True, record=-1):
    _unit_coef = 0.3048 if metric else 1

    _id = data.Vehicle_ID.values 
    _frame = data.Frame_ID.values 
    _x = data.Local_Y.values * _unit_coef
    _y = data.Local_X.values * _unit_coef
    _len = data.v_Length.values * _unit_coef
    _wid = data.v_Width.values * _unit_coef
    _record = np.repeat(record, len(data))
    _dir = np.repeat(1, len(data))

    dt = 0.1
    dx = data.groupby('Vehicle_ID')['Local_Y'].diff()
    dy = data.groupby('Vehicle_ID')['Local_X'].diff()

    _vx = dx/dt * _unit_coef
    _vy = dy/dt * _unit_coef
    _ax = _vx.diff()/dt 
    _ay = _vy.diff()/dt 


    df = pd.DataFrame(
        {
            'id':_id,
            'frame':_frame,
            'record':_record,
            'x':_x-_len/2,
            'y':_y,
            'xVelocity':_vx,
            'yVelocity':_vy,
            'xAcceleration':_ax,
            'yAcceleration':_ay,
            'length':_len,
            'width':_wid,
            'direction':_dir,
        }
    )
    df = df.sort_values(by=['id','frame'])
    return df


if __name__ == '__main__':
    Args = add_arguments()
    print (f"---- Loading CSV NGSIM Trajectories: {Args.data_path}  ----")
    dat = pd.read_csv(Args.data_path)
    tracks = reformat_ngsim(dat, metric=True, record=-1).dropna()
    tracks = tracks[(tracks.x>Args.xlim[0])&(tracks.x<Args.xlim[1])]

    # meta = pd.read_csv(f"E:/Data/highd-dataset-v1.0/data/{Args.record:02}_recordingMeta.csv").iloc[0] 
    # upper = np.array(meta.upperLaneMarkings.split(';'), dtype=float)
    # lower = np.array(meta.lowerLaneMarkings.split(';'), dtype=float)
    # lanes = (upper, lower)
    lanes = ([],[-0.5,  3.5 ,  7.25, 11.  , 14.75, 18.5 , 22.5, 26.5])
    # Generate animation
    ani = animate_ngsim_slider(tracks, lanes, Args.start, T=max(0.1, Args.interval),
                               bubble1=Args.ttc1_bubble,
                               bubble2=Args.ttc2_bubble,
                               thres1=Args.ttc1_thres,
                               thres2=Args.ttc2_thres,
                               coef1=Args.ttc1_coef,
                               coef2=Args.ttc2_coef,
                               mode=Args.mode,
                               grid_dist=Args.grid_dist,
                               ttc_dist=Args.ttc_dist,
                               )

    # Display animation
    plt.show()
