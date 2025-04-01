# This script calculates the ground truth point tracks for EVIMO datasets.
import argparse
import multiprocessing
import os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.linalg import logm
from tqdm import tqdm
import h5py
import multiprocessing.pool as mpp
import yaml
import pandas as pd

def istarmap(self, func, iterable, chunksize=1):
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap

def interpolate_pose(t, pose):
    right_i = np.searchsorted(pose[:, 0], t)
    if right_i==pose.shape[0]:
        return None
    if right_i==0:
        return None

    left_t  = pose[right_i-1, 0]
    right_t = pose[right_i,   0]

    alpha = (t - left_t) / (right_t - left_t)
    if alpha>1:
        return None
    elif alpha < 0:
        return None

    left_position  = pose[right_i - 1, 1:4]
    right_position = pose[right_i,     1:4]

    position_interp = alpha * (right_position - left_position) + left_position

    left_right_rot_stack = Rotation.from_quat((pose[right_i - 1, 4:8],
                                        pose[right_i,     4:8]))

    slerp = Slerp((0, 1), left_right_rot_stack)
    R_interp = slerp(alpha)

    return np.array([t,] + list(position_interp) + list(R_interp.as_quat()))

def apply_transform(T_cb, T_ba):
    R_ba = Rotation.from_quat(T_ba[4:8])
    t_ba = T_ba[1:4]

    R_cb = Rotation.from_quat(T_cb[4:8])
    t_cb = T_cb[1:4]

    R_ca = R_cb * R_ba
    t_ca = R_cb.as_matrix() @ t_ba + t_cb
    return np.array([T_ba[0],] + list(t_ca) + list(R_ca.as_quat()))

def inv_transform(T_ba):
    R_ba = Rotation.from_quat(T_ba[4:8])
    t_ba = T_ba[1:4]

    R_ab = R_ba.inv()
    t_ab = -R_ba.inv().as_matrix() @ t_ba

    return np.array([T_ba[0],] + list(t_ab) + list(R_ab.as_quat()))

def project_points_radtan(points,
                          fx, fy, cx, cy,
                          k1, k2, p1, p2):
    x_ = np.divide(points[:, :, 0], points[:, :, 2], out=np.zeros_like(points[:, :, 0]), where=points[:, :, 2]!=0)
    y_ = np.divide(points[:, :, 1], points[:, :, 2], out=np.zeros_like(points[:, :, 1]), where=points[:, :, 2]!=0)

    r2 = np.square(x_) + np.square(y_)
    r4 = np.square(r2)

    dist = (1.0 + k1 * r2 + k2 * r4)

    x__ = x_ * dist + 2.0 * p1 * x_ * y_ + p2 * (r2 + 2.0 * x_ * x_)
    y__ = y_ * dist + 2.0 * p2 * x_ * y_ + p1 * (r2 + 2.0 * y_ * y_)

    u = fx * x__ + cx
    v = fy * y__ + cy

    return u, v

def get_all_poses(meta):
    vicon_pose_samples = len(meta['full_trajectory'])

    poses = {}
    key_i = {}
    for key in meta['full_trajectory'][0].keys():
        if key == 'id' or key == 'ts' or key == 'gt_frame':
            continue
        poses[key] = np.zeros((vicon_pose_samples, 1+3+4))
        key_i[key] = 0

    for all_pose in meta['full_trajectory']:
        for key in poses.keys():
            if key == 'id' or key == 'ts' or key == 'gt_frame':
                continue

            if key in all_pose:
                i = key_i[key]
                poses[key][i, 0] = all_pose['ts']
                poses[key][i, 1] = all_pose[key]['pos']['t']['x']
                poses[key][i, 2] = all_pose[key]['pos']['t']['y']
                poses[key][i, 3] = all_pose[key]['pos']['t']['z']
                poses[key][i, 4] = all_pose[key]['pos']['q']['x']
                poses[key][i, 5] = all_pose[key]['pos']['q']['y']
                poses[key][i, 6] = all_pose[key]['pos']['q']['z']
                poses[key][i, 7] = all_pose[key]['pos']['q']['w']
                key_i[key] += 1

    for key in poses.keys():
        poses[key] = poses[key][:key_i[key], :]

    return poses

def get_intrinsics(meta):
    meta_meta = meta['meta']
    K = np.array(((meta_meta['fx'],          0, meta_meta['cx']),
                  (         0, meta_meta['fy'], meta_meta['cy']),
                  (         0,          0,          1)))

    dist_coeffs = np.array((meta_meta['k1'],
                            meta_meta['k2'],
                            meta_meta['p1'],
                            meta_meta['p2']))

    return K, dist_coeffs

def load_data(file):
    meta  = np.load(Path(file) / 'dataset_info.npz', allow_pickle=True)['meta'].item()
    depth = np.load(Path(file) / 'dataset_depth.npz')
    mask  = np.load(Path(file) / 'dataset_mask.npz')
    return meta, depth, mask

def convert(file, overwrite=False, max_m_per_s=9.0, max_norm_deg_per_s=6.25*360):
    cv2.setNumThreads(1)

    h5_tracks_file_name = Path(file) / 'dataset_tracks.h5'

    if not overwrite and h5_tracks_file_name.exists():
        print(f'skipping {file} because {h5_tracks_file_name} exists')
        return

    # Load data
    meta, depth, mask = load_data(file)
    all_poses = get_all_poses(meta)
    K, dist_coeffs = get_intrinsics(meta)

    # Get depth shape for map initialization
    first_depth_key = 'depth_' + str(0).rjust(10, '0')
    depth_shape = depth[first_depth_key].shape

    # Initialize undistortion maps
    map1, map2 = cv2.initInverseRectificationMap(
        K,
        dist_coeffs,
        np.eye(3),
        np.eye(3),
        (depth_shape[1], depth_shape[0]),
        cv2.CV_32FC1)

    # Get info from initial frame
    initial_frame_idx = 1 # For frame 0 there are sometimes not previous poses so we start from frame 1
    first_frame_id = meta['frames'][initial_frame_idx]['id']  # Use second frame
    first_frame_info = meta['frames'][initial_frame_idx]
    first_depth_key = 'depth_' + str(initial_frame_idx).rjust(10, '0')  # Use depth from second frame
    first_mask_key = 'mask_' + str(initial_frame_idx).rjust(10, '0')    # Use mask from second frame

    depth_frame = depth[first_depth_key].astype(np.float32) / 1000.0  # convert to meters
    mask_frame = mask[first_mask_key]

    # Initialize points to track (e.g., all points with valid depth)
    valid_points = np.where(depth_frame > 0)  # [y, x]
    valid_points = (valid_points[0][::1000], valid_points[1][::1000])  # for debugging subsample a little bit

    initial_points = np.stack([valid_points[1], valid_points[0]], axis=1)  # nx2 array of x,y coords
    initial_depths = depth_frame[valid_points]
    initial_masks = mask_frame[valid_points]

    # Initialize storage for tracks
    num_frames = len(meta['frames'])
    num_points = len(initial_points)
    tracks = np.full((num_frames, num_points, 2), np.nan, dtype=np.float32)  # store x,y coordinates
    occlusions = np.ones((num_frames, num_points), dtype=bool)  # True means occluded
    times = np.full(num_frames, np.nan, dtype=np.float32)

    # Store initial positions at index 0
    tracks[initial_frame_idx] = initial_points
    occlusions[initial_frame_idx] = False
    times[initial_frame_idx] = meta['frames'][initial_frame_idx]['ts']

    # Unproject initial points to 3D
    Z_m = initial_depths
    X_m = map1[valid_points] * Z_m
    Y_m = map2[valid_points] * Z_m
    initial_XYZ = np.stack((X_m, Y_m, Z_m), axis=1)

    depth_keys = sorted([k for k in depth.files if k.startswith('depth_')])
    frame_numbers = [int(k.split('_')[1]) for k in depth_keys]

    # Get initial poses for each object
    initial_time = first_frame_info['cam']['ts']
    initial_poses = {}
    for key in all_poses:
        if key != 'cam':
            initial_poses[key] = interpolate_pose(initial_time, all_poses[key])

    # Track through all frames starting from the third frame
    found_gap = False
    for frame_idx, frame_info in enumerate(meta['frames'][initial_frame_idx + 1:]):
        frame_idx += (initial_frame_idx + 1)  # Start from index 1 since we used frame 1 as our initial frame

        if frame_numbers[frame_idx] != frame_numbers[frame_idx - 1] + 1:
            print(f"Gap found in depth maps between frames {frame_numbers[frame_idx-1]} and {frame_numbers[frame_idx]}")
            found_gap = True
            break

        times[frame_idx] = frame_info['ts']
        frame_id = frame_info['id'] - first_frame_id
        current_depth_key = 'depth_' + str(frame_id).rjust(10, '0')
        current_mask_key = 'mask_' + str(frame_id).rjust(10, '0')

        current_depth = depth[current_depth_key].astype(np.float32) / 1000.0
        current_mask = mask[current_mask_key]

        # Get current poses for each object
        frame_time = frame_info['cam']['ts']
        frame_poses = {}
        for key in all_poses:
            if key != 'cam':
                frame_poses[key] = interpolate_pose(frame_time, all_poses[key])
        
        # Transform and project points
        for point_idx in range(num_points):
            object_id = initial_masks[point_idx] // 1000
            object_key = str(object_id)
            
            if object_key not in frame_poses or object_key not in initial_poses:
                continue
            
            # Get initial and current object poses
            T_c1o = initial_poses[object_key]
            T_c2o = frame_poses[object_key]
            
            if T_c1o is None or T_c2o is None:
                continue

            # Calculate relative transform from initial camera frame to current camera frame
            T_c2c1 = apply_transform(T_c2o, inv_transform(T_c1o))
            
            # Check velocity to detect potential tracking loss
            dt = frame_time - initial_time
            if dt > 0:
                v = np.linalg.norm(T_c2c1[1:4]) / dt
                R_matrix = Rotation.from_quat(T_c2c1[4:8]).as_matrix()
                w_matrix = logm(R_matrix) / dt
                w = np.array([-w_matrix[1,2], w_matrix[0, 2], -w_matrix[0, 1]])
                w_deg = np.linalg.norm(w) * 180 / np.pi
                
                if v > max_m_per_s or w_deg > max_norm_deg_per_s:
                    continue

            # Transform point using relative transform
            point_3d = initial_XYZ[point_idx]
            R = Rotation.from_quat(T_c2c1[4:8]).as_matrix()
            t = T_c2c1[1:4]
            transformed_point = (R @ point_3d) + t
            
            # Project to image plane
            px, py = project_points_radtan(transformed_point[None, None],
                                         K[0, 0], K[1,1], K[0, 2], K[1, 2],
                                         *dist_coeffs)
            px = px[0, 0]
            py = py[0, 0]
            
            is_occluded = False
            
            # Check if point projects outside image bounds
            if px < 0 or px >= current_depth.shape[1] or py < 0 or py >= current_depth.shape[0]:
                is_occluded = True
            else:
                px_int, py_int = int(px), int(py)
                
                # Check if point is occluded using depth test
                actual_depth = current_depth[py_int, px_int]
                if actual_depth > 0 and abs(actual_depth - transformed_point[2]) > 0.03:  # 3cm threshold
                    is_occluded = True

                # Check if point projects onto the correct object
                if current_mask[py_int, px_int] != initial_masks[point_idx]:
                    is_occluded = True

            tracks[frame_idx, point_idx] = [px, py]
            occlusions[frame_idx, point_idx] = is_occluded

    # Find the last valid timestamp if we found a gap
    if found_gap:
        valid_indices = ~np.isnan(times)
        last_valid_idx = np.where(valid_indices)[0][-1]
    else:
        last_valid_idx = num_frames - 1

    times = times[initial_frame_idx:last_valid_idx + 1]
    tracks = tracks[initial_frame_idx:last_valid_idx + 1]
    occlusions = occlusions[initial_frame_idx:last_valid_idx + 1]

    with h5py.File(h5_tracks_file_name, 'w') as f:
        f.create_dataset('tracks', data=tracks, compression='gzip', compression_opts=9)
        f.create_dataset('occlusions', data=occlusions, compression='gzip', compression_opts=9)
        f.create_dataset('initial_points', data=initial_points, compression='gzip', compression_opts=9)
        f.create_dataset('initial_masks', data=initial_masks, compression='gzip', compression_opts=9)
        f.create_dataset('times', data=times, compression='gzip', compression_opts=9)

def process_with_error_handling(args):
    try:
        convert(*args)
        return True
    except Exception as e:
        print(f"Error processing {args[0]}: {str(e)}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Calculates optical flow from EVIMO datasets.')
    parser.add_argument('--dt', dest='dt', type=float, default=0.01,
                        help='dt for flow approximation')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', 
                        help='Overwrite existing output files')
    parser.add_argument('--max_m_per_s', dest='max_m_per_s', type=float, default=9.0, 
                        help='Maximum meters per second of linear velocity')
    parser.add_argument('--max_norm_deg_per_s', dest='max_norm_deg_per_s', type=float, default=6.25*360, 
                        help='Maximum normed degrees per second of angular velocity')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for dataset paths')
    parser.add_argument('--config', type=str, default='config/misc/evimo2/val_samples.csv',
                        help='YAML config file with dataset paths')
    parser.add_argument('--split', type=str, default='val_samples',
                        help='Dataset split to process from YAML')

    args = parser.parse_args()
    files = list(pd.read_csv(args.config)['name'])

    p_args_list = [[
        Path(args.data_root) / f,
        args.overwrite, 
        args.max_m_per_s, 
        args.max_norm_deg_per_s
    ] for f in files]

    with Pool(multiprocessing.cpu_count()) as p:
        results = list(tqdm(
            p.imap_unordered(process_with_error_handling, p_args_list),
            total=len(p_args_list),
            desc='Sequences'
        ))

    successes = sum(results)
    print(f"Completed {successes}/{len(p_args_list)} sequences")