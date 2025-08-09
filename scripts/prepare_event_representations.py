#!/usr/bin/env python3
# Unified script for preprocessing of events from different event camera datasets
# Supported datasets: EDS, EC, EventKubric, EVIMO2, E2D2, PennAviary
# Example usage:
# python scripts/prepare_event_representations.py --config config/event_representations/eds_config.yaml --dataset eds

import argparse
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm
import h5py
import hdf5plugin
import cv2
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.representations import EventRepresentationFactory
from src.utils import SUPPORTED_SEQUENCES_FEATURE_TRACKING

def propagate_config(config):
    """Add image shape to event representation config based on common settings."""
    height, width = config['common']['height'], config['common']['width']
    config['event_representation']['image_shape'] = (height, width)
    return config

def kalibr_calib2camera_matrix(cam_key, calibration):
    """Convert calibration from Kalibr format to camera matrix."""
    intrinsics = calibration[cam_key]['intrinsics']
    fx, fy, cx, cy = intrinsics
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def homography_from_calibration(K_0, K_1, H_1_0):
    """Calculate homography from calibration matrices."""
    # Mapping from cam1 to cam0
    return K_0 @ H_1_0 @ np.linalg.inv(K_1)

def homography_from_kalibr(calibration):
    """Calculate homography from Kalibr calibration."""
    T = np.array(calibration['cam1']['T_cn_cnm1'])
    K_0 = kalibr_calib2camera_matrix('cam0', calibration)
    K_1 = kalibr_calib2camera_matrix('cam1', calibration)
    H = np.eye(3)  # Simplified assumption
    return homography_from_calibration(K_0, K_1, H)

def undistort_voxel_representation(ev_repr, camera_matrix, distortion_coeffs):
    """Undistort voxel representation."""
    channels, height, width = ev_repr.shape
    ev_repr = ev_repr.astype(np.float32)
    undistorted_repr = np.zeros_like(ev_repr)

    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, camera_matrix, (width, height), cv2.CV_32FC1)

    for i in range(channels):
        undistorted_repr[i] = cv2.remap(ev_repr[i], map1, map2, cv2.INTER_CUBIC)

    return undistorted_repr

def warp_voxel_representation(ev_repr, homography):
    """Warp voxel representation with homography."""
    channels, height, width = ev_repr.shape
    ev_repr = ev_repr.astype(np.float32)
    warped_repr = np.zeros_like(ev_repr)
    for i in range(channels):
        warped_repr[i] = cv2.warpPerspective(ev_repr[i], homography, (width, height))
    return warped_repr

def calculate_frame_times(num_frames=96, total_time=2.0):
    """Calculate frame times for EventKubric dataset."""
    time_step = total_time / (num_frames - 1)
    frame_times = np.arange(num_frames) * time_step
    return frame_times

def invert_events(events, t_min, t_max):
    """
    Invert the polarity and time of events.
    Handles both [0, 1] and [-1, 1] polarity ranges.
    """
    inverted_events = events.copy()

    # Invert time
    inverted_events[:, 2] = t_max - (inverted_events[:, 2] - t_min)

    # Invert polarity
    polarity = inverted_events[:, 3]
    polarity_min, polarity_max = polarity.min(), polarity.max()

    if np.isclose(polarity_min, 0) and np.isclose(polarity_max, 1):
        # [0, 1] case
        inverted_events[:, 3] = 1 - polarity
    elif np.isclose(polarity_min, -1) and np.isclose(polarity_max, 1):
        # [-1, 1] case
        inverted_events[:, 3] = -polarity
    else:
        raise ValueError(f"Unexpected polarity range: [{polarity_min}, {polarity_max}]. "
                         f"Expected [0, 1] or [-1, 1].")

    return inverted_events[::-1]

# Dataset-specific processing functions

def process_eds_subsequence(subsequence_name, config, args, save_name, camera_matrix, distortion_coeffs, homography, converter):
    """Process a subsequence for the EDS dataset."""
    dataset_path = config['common']['dataset_path']
    repr_name = config['event_representation']['representation_name']
    num_events = config['common']['num_events'] if repr_name == 'event_stack' else None
    t_delta_s = config['common'].get('t_delta') if repr_name == 'voxel_grid' else None

    seq_root = os.path.join(dataset_path, subsequence_name)
    out_path = os.path.join(seq_root, 'events', save_name)
    os.makedirs(out_path, exist_ok=True)

    gt_path = os.path.join('config/misc/eds/gt_tracks', f'{subsequence_name}.gt.txt')
    gt_tracks = np.genfromtxt(gt_path)  # [id, t, x, y]
    timestamps_us = (1e6 * np.unique(gt_tracks[:, 1])).astype(np.int64)

    if args.use_rectified:
        events_path = os.path.join(seq_root, 'events_corrected.h5')
    else:
        events_path = os.path.join(seq_root, 'events.h5')
    with h5py.File(events_path, 'r') as f:
        x = f['x']
        y = f['y']
        t = f['t'][:]
        p = f['p']
        ev_indices = np.searchsorted(t, timestamps_us)
        for time, ev_index in tqdm(zip(timestamps_us, ev_indices), total=len(timestamps_us),
                                  desc=subsequence_name):
            if repr_name == 'event_stack':
                i_start = max(ev_index - num_events, 0)
            elif repr_name == 'voxel_grid':
                t_delta = t_delta_s * 1e6  # Convert to microseconds
                t_start = time - t_delta
                i_start = np.searchsorted(t, t_start)
            else:
                raise ValueError(f'Unknown representation name: {repr_name}')
            
            events = np.stack([y[i_start:ev_index], x[i_start:ev_index],
                              t[i_start:ev_index], p[i_start:ev_index]], axis=1)
            ev_repr = converter(events)

            if not args.use_rectified:
                ev_repr = undistort_voxel_representation(ev_repr, camera_matrix,
                                                       distortion_coeffs)
                ev_repr = warp_voxel_representation(ev_repr, homography)

            np.save(os.path.join(out_path, f'{time}.npy'), ev_repr)
    
    print(f'Done processing {subsequence_name}.')


def process_cear_subsequence(subsequence_name, config, args, save_name, camera_matrix, distortion_coeffs, homography, converter):
    """Process a subsequence for the CEAR dataset."""
    dataset_path = config['common']['dataset_path']
    repr_name = config['event_representation']['representation_name']
    num_events = config['common']['num_events'] if repr_name == 'event_stack' else None
    t_delta_s = config['common'].get('t_delta') if repr_name == 'voxel_grid' else None

    seq_root = os.path.join(dataset_path, subsequence_name)
    out_path = os.path.join(seq_root, 'events', save_name)
    os.makedirs(out_path, exist_ok=True)
    print("dataset_path", dataset_path)
    print("seq_root:", seq_root)
    print("out_path:", out_path)


    gt_path = os.path.join('config/misc/cear/gt_tracks', f'{subsequence_name}.gt.txt')
    gt_tracks = np.genfromtxt(gt_path)  # [id, t, x, y]
    timestamps_us = (1e6 * np.unique(gt_tracks[:, 1])).astype(np.int64)

    if args.use_rectified:
        events_path = os.path.join(seq_root, 'events_corrected.h5')
    else:
        events_path = os.path.join(seq_root, 'events.h5')
    print("events_path", events_path)
    with h5py.File(events_path, 'r') as f:
        x = f['x']
        y = f['y']
        t = f['t'][:]
        p = f['p']
        ev_indices = np.searchsorted(t, timestamps_us)
        for time, ev_index in tqdm(zip(timestamps_us, ev_indices), total=len(timestamps_us),
                                  desc=subsequence_name):
            if repr_name == 'event_stack':
                i_start = max(ev_index - num_events, 0)
            elif repr_name == 'voxel_grid':
                t_delta = t_delta_s * 1e6  # Convert to microseconds
                t_start = time - t_delta
                i_start = np.searchsorted(t, t_start)
            else:
                raise ValueError(f'Unknown representation name: {repr_name}')
            
            events = np.stack([y[i_start:ev_index], x[i_start:ev_index],
                              t[i_start:ev_index], p[i_start:ev_index]], axis=1)
            ev_repr = converter(events)

            if not args.use_rectified:
                ev_repr = undistort_voxel_representation(ev_repr, camera_matrix,
                                                       distortion_coeffs)
                ev_repr = warp_voxel_representation(ev_repr, homography)

            np.save(os.path.join(out_path, f'{time}.npy'), ev_repr)
    
    print(f'Done processing {subsequence_name}.')

def process_ec_subsequence(subsequence_name, config, converter):
    """Process a subsequence for the EC dataset."""
    dataset_path = config['common']['dataset_path']
    rep_name = config['event_representation']['representation_name']
    num_events = config['common']['num_events'] if rep_name == 'event_stack' else None
    height, width = config['common']['height'], config['common']['width']
    t_delta_s = config['common'].get('t_delta') if rep_name == 'voxel_grid' else None
    save_name = f'{rep_name}_{config["common"]["save_prefix"]}'

    seq_root = os.path.join(dataset_path, subsequence_name)
    out_path = os.path.join(seq_root, 'events', save_name)
    os.makedirs(out_path, exist_ok=True)

    gt_path = os.path.join('config/misc/ec/gt_tracks', f'{subsequence_name}.gt.txt')
    gt_tracks = np.genfromtxt(gt_path)  # [id, t, x, y]

    timestamps = np.unique(gt_tracks[:, 1])
    timestamps_us = (1e6 * timestamps).astype(np.int64)

    events_txt_path = os.path.join(seq_root, 'events.txt')
    
    print(f"Loading events from {events_txt_path}")
    events_data = np.loadtxt(events_txt_path)

    t = events_data[:, 0]
    x = events_data[:, 1]
    y = events_data[:, 2]
    p = events_data[:, 3]

    intrinsics = np.genfromtxt(os.path.join(seq_root, "calib.txt"))
    camera_matrix = intrinsics[:4]
    distortion_coeffs = intrinsics[4:]
    camera_matrix = np.array(
        [
            [camera_matrix[0], 0, camera_matrix[2]],
            [0, camera_matrix[1], camera_matrix[3]],
            [0, 0, 1],
        ]
    )

    ev_indices = np.searchsorted(t, timestamps)

    for time, ev_index in tqdm(zip(timestamps_us, ev_indices), total=len(timestamps),
                              desc=subsequence_name):
        if rep_name == 'event_stack':
            i_start = max(ev_index - num_events, 0)                        
        elif rep_name == 'voxel_grid':
            t_delta = t_delta_s * 1e6
            t_start = time - t_delta
            i_start = np.searchsorted(t, t_start / 1e6)
        else:
            raise ValueError(f'Unknown representation name: {rep_name}')
        
        events = np.stack([y[i_start:ev_index], x[i_start:ev_index],
                          t[i_start:ev_index], p[i_start:ev_index]], axis=1)
        boundary_mask = (events[:, 0] >= 0) & (events[:, 0] < height) & \
                      (events[:, 1] >= 0) & (events[:, 1] < width)
        events = events[boundary_mask]
        ev_repr = converter(events)

        ev_repr = undistort_voxel_representation(ev_repr, camera_matrix, distortion_coeffs)
        np.save(os.path.join(out_path, f'{time}.npy'), ev_repr)
    
    print(f'Done processing {subsequence_name}.')

def process_event_kubric_example(converter, dataset_path, example_path, save_name, num_events, seq_len, rep_name, rep_config,
                                create_time_inverted, debug):
    """Process a single example in the EventKubric dataset."""
    samples_save_dir = os.path.join(dataset_path, example_path, 'event_representations')
    os.makedirs(samples_save_dir, exist_ok=True)
    sample_save_path = os.path.join(samples_save_dir, f'{save_name}.h5')
    sample_save_path_inverted = sample_save_path.replace('.h5', '_inverted.h5')

    if os.path.exists(sample_save_path) and (not create_time_inverted or os.path.exists(sample_save_path_inverted)):
        return

    if debug:
        debug_dir = os.path.join(dataset_path, example_path, f'debug_visualizations_{save_name}')
        os.makedirs(debug_dir, exist_ok=True)

    event_root = os.path.join(dataset_path, example_path, 'events')
    event_paths = sorted(os.listdir(event_root))
    events = []
    for event_path in event_paths:
        event_mini_batch = np.load(os.path.join(event_root, event_path))
        events.append(np.stack([
            event_mini_batch['y'],
            event_mini_batch['x'],
            event_mini_batch['t'],
            event_mini_batch['p'],
        ], axis=1))
    events = np.concatenate(events, axis=0)

    t_min, t_max = 0.0, 2.0
    t_min_ns, t_max_ns = int(t_min * 1e9), int(t_max * 1e9)

    # Original timestamps
    timestamps = calculate_frame_times(num_frames=96, total_time=t_max)  # Hardcoded for event_kubric_2
    frame_indices = np.arange(3, 93, 3)[3:-3]  # 24 frames with enough padding

    timestamps = timestamps[frame_indices]
    timestamps = 1e9 * timestamps  # Convert to nanoseconds

    def process_events(events_to_process, timestamps_to_use, suffix='', centered_channels=False):
        ev_indices = np.searchsorted(events_to_process[:, 2], timestamps_to_use)
        representations = []
        if rep_name == 'event_stack':
            for ev_i, t_mid in zip(ev_indices, timestamps_to_use):
                if centered_channels:
                    i_mid = ev_i
                    i_start = max(i_mid - num_events // 2, 0)
                    i_end = min(i_mid + num_events // 2, len(events_to_process))
                    repr = converter(events_to_process[i_start:i_end], t_mid)
                else:
                    i_end = ev_i
                    if i_end == 0:
                        repr = np.zeros(((rep_config['num_stacks'],) + rep_config['image_shape']),
                                        dtype=np.float32)
                    else:
                        i_start = max(i_end - num_events, 0)
                        repr = converter(events_to_process[i_start:i_end], t_mid=None)
                representations.append(repr)
        else:
            raise NotImplementedError

        representations = np.stack(representations)
        
        # Save H5 file
        h5_path = sample_save_path if suffix == '' else sample_save_path_inverted
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('representations', data=representations, compression='gzip')
        
        # Save PNG for each timestep if debug flag is set
        if debug:
            for i in range(representations.shape[0]):
                img = representations[i].sum(axis=0)
                img = (img - img.min()) / (img.max() - img.min()) * 255
                img = img.astype(np.uint8)
                
                img_path = os.path.join(debug_dir, f'rep_{i:03d}{suffix}.png')
                Image.fromarray(img).save(img_path)

        return representations

    # Process original events
    centered_channels = rep_config['centered_channels'] if 'centered_channels' in rep_config else False
    process_events(events, timestamps, centered_channels=centered_channels)

    # Process time-inverted events if flag is set
    if create_time_inverted:
        events_inverted = invert_events(events, t_min_ns, t_max_ns)        
        timestamps_inverted = (t_max_ns - (timestamps - t_min_ns))[::-1]  # Invert timestamps
        process_events(events_inverted, timestamps_inverted, suffix='_inverted', centered_channels=centered_channels)

def process_evimo_example(converter, sample_path, num_events, save_name, repr_name):
    """Process a single example in the EVIMO2 dataset."""
    sample_name = sample_path.name
    out_path = sample_path / 'event_representations'
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Load events
    xy = np.load(sample_path / 'dataset_events_xy.npy', mmap_mode='r')
    p = np.load(sample_path / 'dataset_events_p.npy', mmap_mode='r')
    t = np.load(sample_path / 'dataset_events_t.npy', mmap_mode='r')
    
    # Load GT timestamps
    gt_path = sample_path / 'dataset_tracks.h5'
    with h5py.File(gt_path, 'r') as f:
        times = f['times'][:]
    
    # Create event representations
    event_indices = np.searchsorted(t, times)
    all_representations = []
    for i, ev_index in tqdm(enumerate(event_indices), desc=sample_name, total=len(times)):
        if repr_name == 'event_stack':
            assert num_events > 0, 'Number of events must be specified for event stack representation'
            i_start = max(ev_index - num_events, 0)
        elif repr_name == 'e2vid':
            i_start = event_indices[i-1] if i > 0 else 0
            i_start = max(i_start, 0)
        else:
            raise ValueError(f'Unknown representation name: {repr_name}')
        
        events = np.stack([xy[i_start:ev_index, 1],
                          xy[i_start:ev_index, 0],
                          t[i_start:ev_index],
                          p[i_start:ev_index]], axis=1)
        ev_repr = converter(events)
        all_representations.append(ev_repr)
    
    all_representations = np.stack(all_representations, axis=0)
    h5_path = out_path / f'{save_name}.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('timestamps', data=times)
        f.create_dataset('representations', data=all_representations)
    
    print(f'Done processing sample {sample_path}')

def process_e2d2_example(converter, sequence_path, config, args):
    """Process a single sequence for the E2D2 dataset."""
    sequence_path = Path(sequence_path)
    sequence_name = sequence_path.name
    num_events = config['common']['num_events']
    
    rep_name = config['event_representation']['representation_name']
    save_prefix = config['common']['save_prefix']
    save_name = f"{rep_name}_{save_prefix}"

    ev_repr_dir = sequence_path / 'event_representations' / save_name
    ev_repr_dir.mkdir(parents=True, exist_ok=True)

    h5_path = sequence_path / 'seq.h5'
    if not h5_path.exists():
        print(f"Warning: seq.h5 not found in {sequence_path}, skipping...")
        return

    gt_timestamps_path = sequence_path / 'gt_timestamps.npy'
    if gt_timestamps_path.exists():
        # Use ground truth timestamps if available
        timestamps = np.load(gt_timestamps_path)
        print(f"Loaded {len(timestamps)} timestamps from gt_timestamps.npy")
    else:
        # Generate timestamps based on start_time, duration, and step_time
        raise NotImplementedError("Timestamp generation not implemented yet")

    with h5py.File(h5_path, 'r') as f:
        indices_end = np.searchsorted(f['t'], timestamps) - 1
        indices_start = np.maximum(indices_end - num_events, 0)

        for idx, (i_start, i_end, timestamp) in enumerate(
            tqdm(zip(indices_start, indices_end, timestamps), 
                 desc=f"{sequence_name}: creating event representations",
                 total=len(timestamps))):

            output_path = ev_repr_dir / f"{int(timestamp):020d}.npy"
            if output_path.exists() and not args.force:
                continue
            
            events = np.stack([
                f['y'][i_start:i_end],
                f['x'][i_start:i_end],
                f['t'][i_start:i_end],
                f['p'][i_start:i_end]
            ], axis=-1)
            
            ev_repr = converter(events, t_mid=None)
            np.save(output_path, ev_repr)
    
    print(f"Completed processing {sequence_name}")

def process_eds_dataset(config, args):
    """Process the entire EDS dataset."""
    rep_config = config['event_representation']
    rep_name = rep_config['representation_name']
    save_name = f'{rep_name}_{config["common"]["save_prefix"]}'
    converter = EventRepresentationFactory.create(rep_config)

    if not args.use_rectified:
        calib_path = 'config/misc/eds/calib.yaml'

        with open(calib_path, "r") as f:
            calib_data = yaml.safe_load(f)
            homography = homography_from_kalibr(calib_data)
            camera_matrix = kalibr_calib2camera_matrix('cam1', calib_data)
            distortion_coeffs = np.array(calib_data['cam1']['distortion_coeffs'])
    else:
        calib_data = camera_matrix = distortion_coeffs = homography = None

    if args.parallel:
        with mp.Pool(processes=args.num_workers) as pool:
            subsequence_args = [
                (subsequence_name, config, args, save_name, camera_matrix, distortion_coeffs, homography, converter)
                for subsequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']
            ]

            pool.starmap(process_eds_subsequence, subsequence_args)
    else:
        for subsequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']:
            process_eds_subsequence(subsequence_name, config, args, save_name, camera_matrix, 
                                  distortion_coeffs, homography, converter)


def process_cear_dataset(config, args):
    """Process the entire CEAR dataset."""
    rep_config = config['event_representation']
    rep_name = rep_config['representation_name']
    save_name = f'{rep_name}_{config["common"]["save_prefix"]}'
    converter = EventRepresentationFactory.create(rep_config)

    if not args.use_rectified:
        calib_path = 'config/misc/cear/calib.yaml'

        with open(calib_path, "r") as f:
            calib_data = yaml.safe_load(f)
            homography = homography_from_kalibr(calib_data)
            camera_matrix = kalibr_calib2camera_matrix('cam1', calib_data)
            distortion_coeffs = np.array(calib_data['cam1']['distortion_coeffs'])
    else:
        calib_data = camera_matrix = distortion_coeffs = homography = None

    if args.parallel:
        with mp.Pool(processes=args.num_workers) as pool:
            subsequence_args = [
                (subsequence_name, config, args, save_name, camera_matrix, distortion_coeffs, homography, converter)
                for subsequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['cear']
            ]

            pool.starmap(process_cear_subsequence, subsequence_args)
    else:
        for subsequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['cear']:
            process_cear_subsequence(subsequence_name, config, args, save_name, camera_matrix, 
                                  distortion_coeffs, homography, converter)

def process_ec_dataset(config, args):
    """Process the entire EC dataset."""
    rep_config = config['event_representation']
    converter = EventRepresentationFactory.create(rep_config)

    if args.parallel:
        with mp.Pool(processes=args.num_workers) as pool:
            subsequence_args = [
                (subsequence_name, config, converter)
                for subsequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']
            ]

            pool.starmap(process_ec_subsequence, subsequence_args)
    else:
        for subsequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']:
            process_ec_subsequence(subsequence_name, config, converter)

def process_event_kubric_dataset(config, args):
    """Process the EventKubric dataset."""
    rep_config = config['event_representation']
    rep_name = rep_config['representation_name']
    save_name = f'{rep_name}_{config["common"]["save_prefix"]}'
    create_time_inverted = config['common']['create_time_inverted']
    
    assert ('cready' in save_name) == create_time_inverted

    converter = EventRepresentationFactory.create(rep_config)
    dataset_path = config['common']['dataset_path']
    num_events = config['common']['num_events']
    seq_len = config['common']['sequence_length']

    print(f"Dataset path: {dataset_path}")
    print(f"Save name: {save_name}")
    print(f"Number of events: {num_events}")
    print(f"Sequence length: {seq_len}")
    print(f"Time inversion: {create_time_inverted}")

    dataset = sorted(os.listdir(dataset_path))

    process_func = partial(
        process_event_kubric_example,
        converter,
        dataset_path,
        save_name=save_name,
        num_events=num_events,
        seq_len=seq_len,
        rep_name=rep_name,
        rep_config=rep_config,
        create_time_inverted=create_time_inverted,
        debug=args.debug,
    )

    if args.parallel:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_func, example_path) for example_path in dataset]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")
    else:
        for example_path in tqdm(dataset):
            try:
                process_func(example_path)
            except Exception as e:
                print(f"Error processing file {example_path}: {e}")

def process_evimo_dataset(config, args):
    """Process the EVIMO2 dataset."""
    num_events = config['common'].get('num_events', -1)
    repr_name = config['event_representation']['representation_name']
    save_name = f'{repr_name}_{config["common"]["save_prefix"]}'
    
    sample_paths = list(pd.read_csv(config['common']['sample_config'])['name'])
    sample_paths = [Path(config['common']['data_path']) / sample_path for sample_path in sample_paths]
    
    rep_config = config['event_representation']
    converter = EventRepresentationFactory.create(rep_config)
    
    if args.parallel:
        with mp.Pool(processes=args.num_workers) as pool:
            pool.starmap(process_evimo_example, 
                        [(converter, sample_path, num_events, save_name, repr_name) 
                        for sample_path in sample_paths])
    else:
        for sample_path in sample_paths:
            process_evimo_example(converter, sample_path, num_events, save_name, repr_name)

def process_e2d2_dataset(config, args):
    """Process the E2D2 dataset sequences."""    
    rep_config = config['event_representation']
    converter = EventRepresentationFactory.create(rep_config)
    data_root = Path(config['common']['data_root'])
    
    if 'sequences' in config['common'] and config['common']['sequences']:
        sequences = [data_root / seq for seq in config['common']['sequences']]
    else:
        sequences = [d for d in data_root.iterdir() if d.is_dir()]
    
    if not sequences:
        print(f"No sequences found in {data_root}")
        return
    
    print(f"Processing {len(sequences)} sequences for E2D2 dataset")

    if args.parallel:
        with mp.Pool(processes=args.num_workers) as pool:
            pool.starmap(process_e2d2_example, 
                        [(converter, sequence_path, config, args) 
                        for sequence_path in sequences])
    else:
        for sequence_path in sequences:
            process_e2d2_example(converter, sequence_path, config, args)

def main():
    """Main function to parse arguments and process datasets."""
    parser = argparse.ArgumentParser(description='Unified script for preprocessing event camera data')
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to the config file')
    parser.add_argument('--dataset', type=str, required=True, 
                      choices=['eds', 'ec', 'cear', 'event_kubric', 'evimo2', 'e2d2'],
                      help='Which dataset to process')
    parser.add_argument('--use_rectified', action='store_true', 
                      help='Use rectified events (EDS dataset only)')
    parser.add_argument('--parallel', action='store_true',
                      help='Use parallel processing')
    parser.add_argument('--num_workers', type=int, default=32,
                      help='Number of worker processes for parallel processing')
    parser.add_argument('--debug', action='store_true',
                      help='Save debug visualizations (EventKubric dataset only)')
    parser.add_argument('--force', action='store_true',
                      help='Force recomputation of existing files')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if 'common' not in config:
        config['common'] = {}
    if 'save_prefix' not in config['common']:
        config['common']['save_prefix'] = 'default'

    if 'event_representation' in config:
        config = propagate_config(config)

    if args.dataset == 'eds':
        process_eds_dataset(config, args)
    elif args.dataset == 'ec':
        process_ec_dataset(config, args)
    elif args.dataset == 'event_kubric':
        process_event_kubric_dataset(config, args)
    elif args.dataset == 'evimo2':
        process_evimo_dataset(config, args)
    elif args.dataset == 'e2d2':
        process_e2d2_dataset(config, args)
    elif args.dataset == 'cear':
        process_cear_dataset(config, args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print('Processing completed successfully.')

if __name__ == "__main__":
    main()