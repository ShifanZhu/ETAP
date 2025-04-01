import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.representations import VoxelGrid
from src.utils import Visualizer

def make_grid(height, width, stride=1):
    x = np.arange(0, width, stride)
    y = np.arange(0, height, stride)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten()], axis=1)

def read_binary_mask(path):
    img = np.array(Image.open(path).convert('L'))
    mask = (img != 255).astype(np.uint8)
    return mask

def interpolate_wheel_points(query_points, center_point, timestamps, valleys, t_start, target_timestamps=None):
    """
    Interpolate point positions on a rotating wheel.
    
    Args:
        query_points: numpy array of shape [N, 2] containing initial (x, y) positions at t_start
        center_point: tuple or array (x, y) representing wheel center
        timestamps: array of all timestamps used for valley detection
        valleys: indices of valleys in timestamps array (when wheel rotates exactly 1/3)
        t_start: initial timestamp
        target_timestamps: optional array of timestamps at which to calculate positions
        
    Returns:
        numpy array of shape [len(target_timestamps), N, 2] containing interpolated positions
    """
    # Convert everything to numpy arrays and ensure correct shapes
    query_points = np.array(query_points)
    center_point = np.array(center_point)
    
    # Use target_timestamps if provided, otherwise use original timestamps
    target_timestamps = timestamps if target_timestamps is None else target_timestamps
    
    N = len(query_points)
    n_timestamps = len(target_timestamps)
    
    # Initialize output array
    new_positions = np.zeros((n_timestamps, N, 2))
    
    # Calculate initial angles for each point
    initial_vectors = query_points - center_point
    initial_angles = np.arctan2(initial_vectors[:, 1], initial_vectors[:, 0])
    radii = np.linalg.norm(initial_vectors, axis=1)
    
    # Calculate angular velocity between each pair of valleys using original timestamps
    valley_times = timestamps[valleys]
    angular_velocity = -2 * np.pi / 3  # negative for clockwise rotation
    
    # For each target timestamp, calculate the angle and new position
    for i, t in enumerate(target_timestamps):
        # Find the appropriate valley interval using original valley times
        if t < valley_times[0]:
            # Before first valley - interpolate from start
            delta_t = t - timestamps[0]
            fraction = delta_t / (valley_times[0] - timestamps[0])
            angle_change = fraction * angular_velocity
        elif t > valley_times[-1]:
            # After last valley - extrapolate from last interval
            delta_t = t - valley_times[-1]
            last_interval = valley_times[-1] - valley_times[-2]
            angle_change = angular_velocity * (len(valleys) - 1 + delta_t / last_interval)
        else:
            # Between valleys - find appropriate interval and interpolate
            valley_idx = np.searchsorted(valley_times, t) - 1
            valley_idx = max(0, valley_idx)
            delta_t = t - valley_times[valley_idx]
            interval = valley_times[valley_idx + 1] - valley_times[valley_idx]
            fraction = delta_t / interval
            angle_change = angular_velocity * (valley_idx + fraction)
            
        # Calculate new angles and positions
        new_angles = initial_angles + angle_change
        
        # Convert polar coordinates back to Cartesian
        new_positions[i, :, 0] = center_point[0] + radii * np.cos(new_angles)
        new_positions[i, :, 1] = center_point[1] + radii * np.sin(new_angles)

    return new_positions

if __name__ == '__main__':
    data_dir = Path('data/e2d2/231025_110210_fidget5_high_exposure')
    output_dir = Path('output/e2d2_gt/231025_110210_fidget5_high_exposure')

    data_path = data_dir / 'seq.h5'
    sequence_name = '231025_110210_fidget5_high_exposure'
    t_start = 3.3961115
    duration = 0.5
    t_delta = 0.001
    timestamps = np.arange(t_start, t_start + duration, t_delta)
    timestamps = (timestamps * 1e6).astype(int)
    N_events = 20000
    image_shape = (480, 640)
    converter = VoxelGrid(image_shape, num_bins=1)
    threshold_l2_norm = 300
    mask_path = data_dir / 'mask00004.png'
    query_stride = 40
    center_point = np.array([369, 229])
    t_delta_gt = 0.0033
    timestamps_gt = np.arange(t_start, t_start + duration, t_delta_gt)

    output_dir = output_dir / f'{str(int(1e6 * t_delta_gt)).zfill(8)}'
    os.makedirs(output_dir, exist_ok=True)
    #os.makedirs(output_dir / 'histograms', exist_ok=True)

    # List to store L2 norms
    l2_norms = []
    first_frame = None

    with h5py.File(data_path, 'r') as f:
        ev_idx = np.searchsorted(f['t'][:], timestamps) - 1
        ev_idx_start = ev_idx - N_events // 2
        ev_idx_end = ev_idx + N_events // 2

        vid = []

        for i_start, i_end, t in tqdm(zip(ev_idx_start, ev_idx_end, timestamps), total=len(timestamps)):
            events = np.stack([f['y'][i_start:i_end],
                             f['x'][i_start:i_end],
                             f['t'][i_start:i_end],
                             f['p'][i_start:i_end]], axis=-1)

            repr = converter(events)
            repr = repr[0]

            if first_frame is None:
                first_frame = repr.copy()

            l2_norm = np.sqrt(np.sum((repr - first_frame) ** 2))
            l2_norms.append(l2_norm)

            if repr.max() != repr.min():
                repr_norm = ((repr - repr.min()) / (repr.max() - repr.min()) * 255).astype(np.uint8)
            else:
                repr_norm = np.zeros_like(repr, dtype=np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(repr_norm)

            # Add timestamp to image
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()

            timestamp_sec = t / 1e6  # Convert to seconds for display
            draw.text((10, 10), f"{timestamp_sec:.3f}s", font=font, fill=255)

            frame_with_timestamp = np.array(img)
            vid.append(frame_with_timestamp)
 
            #filename = f"{t:012d}.png"
            #filepath = output_dir / 'histograms' / filename
            #img.save(filepath)

    video = np.stack(vid, axis=0)  # Shape: [T, H, W]
    video = np.float32(video)
    video = np.stack([video, video, video], axis=1)  # Shape: [T, 3, H, W]
    video = 255 * (video - video.min()) / (video.max() - video.min())

    # Convert to numpy array and invert the signal
    l2_norms = np.array(l2_norms)
    inverted_signal = -l2_norms

    # Find valleys (peaks in the inverted signal)
    valleys, _ = find_peaks(inverted_signal,
                          height=(-threshold_l2_norm, 0),
                          distance=10,
                          prominence=10)

    valleys = np.concatenate([np.zeros(1, dtype=valleys.dtype), valleys])

    # Plot detected minima with scientific formatting
    plt.figure(figsize=(12, 6))
    timestamps_sec = timestamps / 1e6
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    plt.plot(timestamps_sec, l2_norms, label='L2 Norm')
    plt.plot(timestamps_sec[valleys], l2_norms[valleys], 'r*', label='Detected Valleys')
    plt.axhline(y=threshold_l2_norm, color='r', linestyle='--', 
                label=f'Threshold ({threshold_l2_norm})')
    plt.xlabel('Time (s)')
    plt.ylabel('L2 Norm')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_dir / 'l2_norms.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # Make queries
    height, width = image_shape
    query_xy = make_grid(height, width, stride=query_stride)
    num_queries = query_xy.shape[0]

    if mask_path is not None:
        segm_mask = read_binary_mask(mask_path)
        query_x = query_xy[:, 0]
        query_y = query_xy[:, 1]
        segm_mask = segm_mask[query_y, query_x]
        query_xy = query_xy[segm_mask == 1]

    # Interpolate positions for visualization
    new_positions = interpolate_wheel_points(query_xy, center_point, timestamps, valleys, t_start)

    # Calculate positions at GT timestamps
    timestamps_gt_us = (timestamps_gt * 1e6).astype(int)  # Convert to same units as timestamps
    new_positions_gt = interpolate_wheel_points(query_xy, center_point, timestamps, valleys, t_start, 
                                              target_timestamps=timestamps_gt_us)
    np.save(output_dir / 'gt_positions.npy', new_positions_gt)
    np.save(output_dir / 'gt_timestamps.npy', timestamps_gt_us)
    np.save(output_dir / 'queries.npy', query_xy)

    viz = Visualizer(save_dir=str(output_dir),
                     pad_value=0,
                     linewidth=2,
                     tracks_leave_trace=-1,
                     show_first_frame=0)
    rgbs = viz.visualize(
        torch.from_numpy(video[None]),
        torch.from_numpy(new_positions[None])
    )

    print('Done.')