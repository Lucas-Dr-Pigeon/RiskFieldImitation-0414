import numpy as np
import torch
import pandas as pd
from tqdm import tqdm, trange
from torch_geometric.utils import lexsort
import matplotlib.patches as patches
import argparse


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-path', type=str, default="./data")
    parser.add_argument('--ttc1-coef', type=float, default=1)
    parser.add_argument('--ttc2-coef', type=float, default=1)
    parser.add_argument('--ttc1-bubble', type=float, default=0)
    parser.add_argument('--ttc2-bubble', type=float, default=0.25)
    parser.add_argument('--ttc1-thres', type=float, default=5.0)
    parser.add_argument('--ttc2-thres', type=float, default=7.5)
    parser.add_argument('--batch-size', type=int, default=100000)
    parser.add_argument('--to-data-path', type=str, default="./data/")
    parser.add_argument('--ttc-mode', type=str, default="circle")
    parser.add_argument('--pair', type=int, default=1)
    parser.add_argument('--sequence', type=int, default=250)
    parser.add_argument('--visible', type=int, default=1)
    parser.add_argument('--start-stop-window', type=int, default=25)
    args = parser.parse_args()
    return args

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

    for i in trange(num_batches):
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
    batch_size = min(N, batch_size)
    num_batches = (N + batch_size - 1) // batch_size  # Compute number of batches
    
    # Initialize MTTC results
    mttc_results = torch.full((N,), 1e9, dtype=torch.float32, device=device)  # Default to 1e9

    for i in trange(num_batches):
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
        half_sizes = (ego_dim_batch.to('cuda')+ foe_dim_batch.to('cuda') + 2*bubble) / 2  # (N, 2) sum of half-lengths

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

def compute_time_until_next_ttc_below_threshold_vectorized(data, threshold=1.5, batch_size=500):
    """
    Compute the time (frames) until the next occurrence of `TTC < threshold` for each unique 
    ego-foe interaction (defined by ego_id, foe_id, recording_id).

    If the current row already has `TTC < threshold`, return 0.

    Args:
        data: (N, 5) tensor - Columns: [ego_id, foe_id, recording_id, frame_id, ttc]
        threshold: float - TTC threshold (default 1.5 seconds).
        batch_size: int - Number of unique (ego_id, foe_id, recording_id) pairs processed at once.

    Returns:
        time_until_next_ttc_below: (N,) tensor - Time until next TTC < threshold for each row, 
                                              in the original input order.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to GPU if available
    data = data.to(device)

    # Save original indices to restore order later
    original_indices = torch.arange(data.shape[0], device=device)

    # Extract columns
    ego_id, foe_id, recording_id, frame_id, ttc = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    # ✅ Step 1: Correct Multi-Column Sorting
    # Sorting by (ego_id, foe_id, recording_id, frame_id)
    sorted_indices = lexsort((frame_id, recording_id, foe_id, ego_id))

    # Apply sorting
    ego_id, foe_id, recording_id, frame_id, ttc, original_indices = (
        ego_id[sorted_indices], foe_id[sorted_indices], recording_id[sorted_indices],
        frame_id[sorted_indices], ttc[sorted_indices], original_indices[sorted_indices]
    )

    # ✅ Step 2: Identify Unique (ego_id, foe_id, recording_id) Pairs
    unique_pairs, pair_indices = torch.unique(
        torch.stack([ego_id, foe_id, recording_id], dim=1),
        dim=0, return_inverse=True
    )

    num_pairs = unique_pairs.shape[0]  # Total unique pairs
    num_batches = (num_pairs + batch_size - 1) // batch_size  # Compute number of batches

    # ✅ Step 3: Identify All Frames Where `TTC < threshold`
    below_threshold_mask = ttc < threshold  # Boolean mask where TTC < threshold
    valid_frame_ids = frame_id[below_threshold_mask]  # Frames where `TTC < threshold`
    valid_pair_indices = pair_indices[below_threshold_mask]  # Matching pair indices

    # Initialize the result tensor
    time_until_next_ttc_below = torch.full_like(frame_id, float('inf'), device=device)

    # ✅ Step 4: Process in Batches to Reduce Memory Usage
    for batch_idx in trange(num_batches):
        # Get the current batch of pairs
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_pairs)
        batch_pair_indices = torch.arange(start_idx, end_idx, device=device)

        # Get mask for rows belonging to this batch
        batch_mask = (pair_indices >= start_idx) & (pair_indices < end_idx)

        if batch_mask.sum() == 0:  # Skip if no data in batch
            continue

        # Get the frame_ids of the current batch
        batch_frame_ids = frame_id[batch_mask]

        # Get the valid events that belong to this batch
        batch_valid_mask = (valid_pair_indices >= start_idx) & (valid_pair_indices < end_idx)
        batch_valid_frame_ids = valid_frame_ids[batch_valid_mask]
        batch_valid_pair_indices = valid_pair_indices[batch_valid_mask]

        if batch_valid_frame_ids.numel() == 0:  # Skip empty batches
            continue

        # Compute `valid_frame_ids - frame_id` in a batched, vectorized manner
        all_time_diffs = batch_valid_frame_ids.unsqueeze(0) - batch_frame_ids.unsqueeze(1)  # (batch_size, num_valid_events)
        all_time_diffs[all_time_diffs <= 0] = float('inf')  # Ignore past events  

        # ✅ Step 5: Ensure we only consider valid pairs
        pair_match_mask = pair_indices[batch_mask].unsqueeze(1) == batch_valid_pair_indices.unsqueeze(0)  # Shape: (batch_size, num_valid_events)
        all_time_diffs[~pair_match_mask] = float('inf')  # Mask invalid pairs

        # Compute the minimum valid time difference per row
        min_time_diffs, _ = all_time_diffs.min(dim=1)

        # ✅ Step 6: Assign results back
        time_until_next_ttc_below[batch_mask] = min_time_diffs

    # ✅ Step 7: Post-processing
    # If the row itself is `TTC < threshold`, set time until next to `0`
    time_until_next_ttc_below[below_threshold_mask] = 0

    # If no future `TTC < threshold` exists, set time to `-1`
    time_until_next_ttc_below[time_until_next_ttc_below == float('inf')] = -1

    # ✅ Step 8: Restore Original Order
    restored_order = torch.argsort(original_indices)  # Get indices that map back to the original order
    time_until_next_ttc_below = time_until_next_ttc_below[restored_order]  # Restore order

    return time_until_next_ttc_below


def compute_transitions_vectorized_mode_1(data, thres1=1.5, thres2=3.0, device='cuda'):
    data = torch.tensor(data, device=device)

    ttc1 = compute_time_until_next_ttc_below_threshold_vectorized(data[:,(0,1,2,3,4)], threshold=thres1, batch_size=1000)
    ttc2 = compute_time_until_next_ttc_below_threshold_vectorized(data[:,(0,1,2,3,5)], threshold=thres2, batch_size=1000)

    # Step 1: Extract columns
    ego_id, foe_id, recording_id, frame_id = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    original_indices = torch.arange(data.shape[0], device=device)
    sorted_indices = lexsort((frame_id, recording_id, foe_id, ego_id))
    
    # Apply sorting
    ego_id, foe_id, recording_id, frame_id, original_indices, ttc1, ttc2 = (
        ego_id[sorted_indices], foe_id[sorted_indices], recording_id[sorted_indices],
        frame_id[sorted_indices], original_indices[sorted_indices], ttc1[sorted_indices], ttc2[sorted_indices]
    )

    # Step 2: Find unique (ego_id, foe_id, recording_id) groups
    unique_pairs, pair_indices = torch.unique(torch.stack([ego_id, foe_id, recording_id], dim=1), dim=0, return_inverse=True)

    states = torch.full_like(ttc1, 3, dtype=torch.int32, device='cuda')  # Default is state 3
    states[ttc2 == 0] = 2  # If TTC < 5, but not < 1.5
    states[ttc1 == 0] = 1  # If TTC < 1.5, override to state 1

    state_changes = (states[1:] != states[:-1])*(pair_indices[1:] == pair_indices[:-1])
    transition_indices = torch.where(state_changes)[0] + 1  # Convert to indices (offset by 1)
    
    first_frames_per_group = torch.full_like(frame_id, float('inf'))  # Initialize with large values
    first_frames_per_group = first_frames_per_group.scatter_reduce_(0, pair_indices, frame_id, reduce="amin")[:len(unique_pairs)]

    first_transition_per_group = torch.full((pair_indices.max() + 1,), torch.iinfo(torch.long).max, dtype=torch.long, device=transition_indices.device)

    # Step 3: Scatter reduce to find the first transition index per group
    first_transition_per_group.scatter_reduce_(0, pair_indices[transition_indices], transition_indices, reduce="amin")

    # Step 4: Compare transition indices with the first transition per group
    first_transition_mask = transition_indices == first_transition_per_group[pair_indices[transition_indices]]

    # Step 5: 
    n_transitions = len(transition_indices)
    transition_from_states = states[transition_indices-1]
    transition_to_states = states[transition_indices]
    transition_start_frames = torch.cat([torch.tensor([0], device=device), frame_id[transition_indices[0:n_transitions-1]]])
    transition_start_frames = transition_start_frames*~first_transition_mask + first_transition_mask*first_frames_per_group[pair_indices[transition_indices]]
    transition_end_frames = frame_id[transition_indices[0:n_transitions]] 

    transitions = torch.stack([ego_id[transition_indices], foe_id[transition_indices], recording_id[transition_indices], transition_start_frames, transition_end_frames, transition_from_states, transition_to_states], dim=1)
    return transitions

def get_processed_input_sequence(data, transition, sequence=75, from_state=2, to_state=1, device='cuda', surrounding=False, max_surround=10, start_stop=False):
    """
    Retrieve the inputs according to the start and end time of transitions.

    If the current row already has `TTC < threshold`, return 0.

    Args:
        data: (N, X) DataFrame - raw data, positions, velocity, and acceleration
        transition: (N, Y) - transition matrix, ego, foe, recording id, start frame, end frame, from state, to state.
        sequence: int - the number of sequences
    Returns:
        data_sequence: the sequence of vehicle kinematics before transition
    """
    transition = transition[ (transition['from']==from_state)&(transition['to']==to_state) ]  if from_state != None and to_state != None else transition # only process certain transitions
    Tran = torch.tensor(transition.values, device=device)

    # get lane centers

    laneCenters = np.full((60, 2, 4), 9999)
    for r in trange(1,61):
        meta = pd.read_csv(f"D:/Productivity/Projects/High-D/highd-dataset-v1.0/data/{r:02}_recordingMeta.csv").iloc[0] 
        upper = np.array(meta.upperLaneMarkings.split(';'), dtype=float)
        lower = np.array(meta.lowerLaneMarkings.split(';'), dtype=float)
        upper_centers = (upper[1:] + upper[:-1])/2
        lower_centers = (lower[1:] + lower[:-1])/2
        laneCenters[r-1,0,:len(upper_centers)] = upper_centers
        laneCenters[r-1,1,:len(lower_centers)] = lower_centers

    # Let's put important data into a tensor
    vehicle_mean_speed = data.groupby(['ego_id','recording_id']).ego_xVel.transform('mean')
    vehicle_abs_speed = vehicle_mean_speed.abs()    # get vehicle moving directions +/-
    data['dir'] = vehicle_mean_speed / vehicle_abs_speed 
    # use _dir as a directional index
    _dir = np.clip(data.dir.values, a_min=0, a_max=1)
    # get lane deviation
    track_lanes_centers = laneCenters[data.recording_id.values.astype('int')-1, _dir.astype('int') ]
    devs = data.ego_y.values[:,np.newaxis] - track_lanes_centers
    min_abs_indices = np.argmin(np.abs(devs), axis=1)
    lane_deviates = devs[np.arange(devs.shape[0]), min_abs_indices]
    # get spacing
    _x =  (data.foe_x - data.ego_x).values * data.dir.values 
    _y =  (data.foe_y - data.ego_y).values * data.dir.values 
    # get ego speed
    ego_xVel = np.abs(data.ego_xVel.values) 
    ego_yVel = data.ego_yVel.values * data.dir.values
    # get foe speed
    foe_xVel = np.abs(data.foe_xVel.values) 
    foe_yVel = data.foe_yVel.values * data.dir.values
    # get ego accel
    ego_xAccel = data.ego_xAccel.values * data.dir.values
    ego_yAccel = data.ego_yAccel.values * data.dir.values
    # get foe accel
    foe_xAccel = data.foe_xAccel.values
    foe_yAccel = data.foe_yAccel.values * data.dir.values
    # get ego id 
    _ego = data.ego_id.values
    # get foe id
    _foe = data.foe_id.values
    # get ego class
    ego_class = data.ego_truck.values
    # get foe class
    foe_class = data.foe_truck.values
    # get ego length, width
    ego_length = data.ego_length.values
    ego_width = data.ego_width.values
    # get foe length, width
    foe_length = data.foe_length.values
    foe_width = data.foe_length.values
    # get record
    _record = data.recording_id.values
    #
    # get frame
    _frame = data.frame.values
    data = np.array(
    [
        _x, _y, ego_xVel, ego_yVel, foe_xVel, foe_yVel, ego_xAccel, ego_yAccel, foe_xAccel, foe_yAccel, lane_deviates, ego_length, ego_width, foe_length, foe_width, _ego, _foe, _record, _frame
    ]
    ).T
    np.nan_to_num(data, 0)
    data = torch.tensor(data, device=device, dtype=torch.float)

    _ego = data[:,-4]
    _foe = data[:,-3]
    _record = data[:,-2]
    _frames = data[:,-1]

    n_features = data.shape[1]
    n_sequance = int(sequence/5)
    
    if start_stop:
        if surrounding:
            _Data = torch.torch.full((len(Tran), n_sequance, max_surround, n_features), float('nan'), device='cuda')
            for m in trange(len(Tran)):
                ego, foe, record, start, end, _from, to = Tran[m]
                _start = torch.max(end - (n_sequance-1)*5, start)
                mask = ((_frames.unsqueeze(1) >= (_start)) & (_frames.unsqueeze(1) <= end) & (_ego.unsqueeze(1)==ego) & (_record.unsqueeze(1)==record)   ).any(dim=1)
                surr = data[mask]
                unique_foes = torch.unique(surr[:, -3])
                
                for f, foeid in enumerate(unique_foes):
                    if f >= max_surround:
                        break 
                    fseq = surr[surr[:, -3] == foeid]
                    indices =  (-(end-fseq[:,-1])/ 5 ).to(torch.int) - 1  
                    _Data[m, indices, f, :] = fseq.view(1, len(indices), n_features)
        else:
            _Data = torch.torch.full((len(Tran), n_sequance, n_features), float('nan'), device='cuda')
            for m in trange(len(Tran)):
                ego, foe, record, start, end, _from, to = Tran[m]
                _start = torch.max(end - (n_sequance-1)*5, start)
                
                mask = ((_frames.unsqueeze(1) >= (_start)) & (_frames.unsqueeze(1) <= end) & (_ego.unsqueeze(1)==ego) &  (_foe.unsqueeze(1)==foe) & (_record.unsqueeze(1)==record)   ).any(dim=1)
                indices = (-(end-_frames[mask])/ 5 ).to(torch.int) - 1
                _Data[m][indices] = data[mask]
    else:
        if surrounding:
            _Data = torch.torch.full((len(Tran), n_sequance, max_surround, n_features), float('nan'), device='cuda')
            for m in trange(len(Tran)):
                ego, foe, record, start, end, _from, to = Tran[m]
                mask = ((_frames.unsqueeze(1) > (start - sequence)) & (_frames.unsqueeze(1) <= start) & (_ego.unsqueeze(1)==ego) & (_record.unsqueeze(1)==record)   ).any(dim=1)
                surr = data[mask]
                unique_foes = torch.unique(surr[:, -3])
                for f, foeid in enumerate(unique_foes):
                    if f >= max_surround:
                        break 
                    fseq = surr[surr[:, -3] == foeid]
                    indices =  (-(start-fseq[:,-1])/ 5 ).to(torch.int) - 1  
                    _Data[m, indices, f, :] = fseq.view(1, len(indices), n_features)
        else:
            _Data = torch.torch.full((len(Tran), n_sequance, n_features), float('nan'), device='cuda')
            for m in trange(len(Tran)):
                ego, foe, record, start, end, _from, to = Tran[m]
                mask = ((_frames.unsqueeze(1) > (start - sequence)) & (_frames.unsqueeze(1) <= start) & (_ego.unsqueeze(1)==ego) &  (_foe.unsqueeze(1)==foe) & (_record.unsqueeze(1)==record)   ).any(dim=1)
                indices = (-(start-_frames[mask])/ 5 ).to(torch.int) - 1
                _Data[m][indices] = data[mask]
    return _Data

def get_start_stop_input_transitions(data, _transition, window=15, max_frame=250, sequence=250, from_state=2, to_state=1, device='cuda', surrounding=False, max_surround=10):
    # get intermediate input
    _input = get_processed_input_sequence(data, _transition, sequence, from_state, to_state, device, surrounding, max_surround, start_stop=True)
    valid_mask = (_transition.end - _transition.start) < 250
    transition = _transition[valid_mask]
    _input = _input[valid_mask]
    # process the input data
    ChunkList = []
    print ( "-- Generating start stop data -- ")
    for i in trange(_input.shape[0]):
        row = _input[i]
        dur = transition.iloc[i]
        transition_mask = ~torch.isnan(row).any(dim=1)
        transition_data = row[transition_mask]
        transition_frames = transition_data[:,-1]
        group_ids = ((transition_frames - transition_frames[0]) // window).long()
        max_g = max(group_ids)
        group_counts = torch.bincount(group_ids)
        for g in group_ids.unique():
            chunk = transition_data[group_ids == g]
            chunkdata = chunk[0]
            # tenure
            tenure = torch.tensor([g * window], device=chunkdata.device)
            # transition duration
            past = torch.tensor([window], device=chunkdata.device) if g < max_g else torch.tensor([group_counts[-1]*5], device=chunkdata.device)
            # from 
            _from = torch.tensor([dur['from']], device=chunkdata.device) 
            # to
            _to = _from if g < max_g else torch.tensor([dur['to']], device=chunkdata.device)

            new_row = torch.cat([chunkdata, tenure, past, _from, _to])
            ChunkList.append(new_row)
    Chunks = torch.vstack(ChunkList)
    return Chunks





if __name__ == '__main__':
    Args = add_arguments()

    data_path = Args.raw_data_path
    coef1 = Args.ttc1_coef
    coef2 = Args.ttc2_coef
    thres1 = Args.ttc1_thres
    thres2 = Args.ttc2_thres
    bubble1 = Args.ttc1_bubble
    bubble2 = Args.ttc2_bubble
    batchsize = Args.batch_size
    save_path = Args.to_data_path
    start_stop_window = Args.start_stop_window
    mode = Args.ttc_mode
    visible_only = Args.visible
    sequence = Args.sequence
    surround = False if Args.pair else True
    name = f"2d-ttc-2-levels-{mode}-t{thres1}-t{thres2}-c{coef1}-c{coef2}-b{bubble1}-b{bubble2}-{'surround' if surround else 'pair'}-{'start-stop' if start_stop_window else ''}"

    print (f"---- Reading raw data ----")
    full_data = pd.concat([ pd.read_csv(f"{data_path}/{r:02}_spacings_2.csv",index_col=0) for r in trange(1,61)])
    # get vehicle moving direcions
    vehicle_mean_speed = full_data.groupby(['ego_id','recording_id']).ego_xVel.transform('mean')
    vehicle_abs_speed = vehicle_mean_speed.abs()    # get vehicle moving directions +/-
    dirs = vehicle_mean_speed / vehicle_abs_speed 

    # if foe vehicle is in front of the ego vehicle, it is visible to the ego vehicle
    if visible_only:
        visible = (full_data.foe_x - full_data.ego_x).values * dirs >= 0
        full_data = full_data[visible]

    ego_pos = torch.tensor(full_data[['ego_x','ego_y']].values, device='cuda')
    foe_pos = torch.tensor(full_data[['foe_x','foe_y']].values, device='cuda')
    ego_dim = torch.tensor(full_data[['ego_length','ego_width']].values, device='cuda')
    foe_dim = torch.tensor(full_data[['foe_length','foe_width']].values, device='cuda')
    ego_vel = torch.tensor(full_data[['ego_xVel','ego_yVel']].values, device='cuda')
    foe_vel = torch.tensor(full_data[['foe_xVel','foe_yVel']].values, device='cuda')
    ego_accel = torch.tensor(full_data[['ego_xAccel','ego_yAccel']].values, device='cuda')
    foe_accel = torch.tensor(full_data[['foe_xAccel','foe_yAccel']].values, device='cuda')

    print (f"---- Calculating TTC. Levels: 2. Mode: {mode} ----")

    if mode == 'circle':
        _ttcs1 = get_2D_ttc_circles_cuda_batched(
            ego_pos, ego_vel, ego_dim,
            foe_pos, foe_vel, foe_dim,
            batch_size=batchsize, buffer_coef=coef1, bubble=bubble1
        )

        _ttcs2 = get_2D_ttc_circles_cuda_batched(
            ego_pos, ego_vel, ego_dim,
            foe_pos, foe_vel, foe_dim,
            batch_size=batchsize, buffer_coef=coef2, bubble=bubble2
        )   
    elif mode == 'rectangle':
        _ttcs1 = rectangle_modified_time_to_collision_batched(
            ego_pos, ego_vel, ego_accel, ego_dim,
            foe_pos, foe_vel, foe_accel, foe_dim,
            batch_size=batchsize, bubble=bubble1, T=10, dt=0.1,
        )

        _ttcs2 = rectangle_modified_time_to_collision_batched(
            ego_pos, ego_vel, ego_accel, ego_dim,
            foe_pos, foe_vel, foe_accel, foe_dim,
            batch_size=batchsize, bubble=bubble2, T=10, dt=0.1
        )  

    data = full_data # shallow copy 
    data['ttc1'] = _ttcs1.cpu()
    data['ttc2'] = _ttcs2.cpu()

    print (f"---- Calculating state transition durations ----")

    tran = compute_transitions_vectorized_mode_1(  torch.tensor(data.values[:,(27,28,26,29,30,31)].astype(np.float32)), 
                                                        device='cuda', 
                                                        thres1=thres1,
                                                        thres2=thres2
                                                    )
    Tran = pd.DataFrame(tran.cpu(), columns=['ego','foe','record','start','end','from','to'])
    
    print (f"---- Calculating input sequences ----")
    if start_stop_window:
        Chunks = get_start_stop_input_transitions(full_data, Tran, window=start_stop_window, sequence=sequence, from_state=None, to_state=None, surrounding=surround)
        columns = ['xSpace', 'ySpace', 'ego_xVel', 'ego_yVel', 'foe_xVel', 'foe_yVel', 'ego_xAccel', 'ego_yAccel', 'foe_xAccel', 'foe_yAccel', 'lane_deviates', 'ego_length', 'ego_width', 'foe_length', 'foe_width', 'ego_id', 'foe_id', 'recording_id', 'frame', 'tenure', 'duration', 'from', 'to']
        start_stop_df = pd.DataFrame(Chunks.cpu(), columns=columns)
        start_stop_df.to_csv(f"{save_path}{name}-transition.csv")
    else:
        input_sequence = get_processed_input_sequence(full_data, Tran, sequence=sequence, from_state=None, to_state=None, surrounding=surround)
        print (f"---- Saving data ----")
        Tran.to_csv(f"{save_path}{name}-transition.csv")
        torch.save(input_sequence, f"{save_path}{name}-input-seq-{sequence}")

    print (f"---- Complete! ----")