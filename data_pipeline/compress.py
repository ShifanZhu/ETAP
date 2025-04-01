import os
import json
import h5py
import numpy as np
import cv2
import glob
import argparse
import multiprocessing
from functools import partial
import hdf5plugin
import yaml
from tqdm import tqdm


def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")

    # Convert BGR to RGB for 3-channel color images
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert BGRA to RGBA for 4-channel images
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    return img


def calculate_original_size(sample_path):
    total_size = 0
    for file_name in os.listdir(sample_path):
        if file_name != 'data.hdf5':  # Exclude the HDF5 file if it exists
            file_path = os.path.join(sample_path, file_name)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def process_sample(sample_path, verbose, subsample_factor, compression_level=9):
    output_file = os.path.join(sample_path, 'data.hdf5')
    sample_path = os.path.join(sample_path, "raw")

    # Skip if HDF5 file already exists
    if os.path.exists(output_file):
        if verbose:
            print(f"Skipping {sample_path}: HDF5 file already exists")
        return None

    try:
        original_size = calculate_original_size(sample_path)

        with h5py.File(output_file, 'w') as hf:
            for data_type in ['backward_flow', 'depth', 'forward_flow', 'normal', 'object_coordinates', 'rgba',
                              'segmentation']:
                file_pattern = f"{data_type}_*.{'tiff' if data_type == 'depth' else 'png'}"
                files = sorted(glob.glob(os.path.join(sample_path, file_pattern)))

                if not files:
                    print(f"Warning: No {data_type} files found in {sample_path}")
                    continue

                if data_type != 'rgba':
                    # Subsample for all data types except 'rgba'
                    files = files[::subsample_factor]

                images = [load_image(f) for f in files]
                dataset = np.stack(images)

                hf.create_dataset(data_type, data=dataset, compression="gzip", compression_opts=compression_level)

            # Store subsampling indices
            if len(files) > 0:
                total_frames = len(glob.glob(os.path.join(sample_path, 'rgba_*.png')))
                subsample_indices = np.arange(0, total_frames, subsample_factor)
                hf.create_dataset('subsample_indices', data=subsample_indices, compression="gzip",
                                  compression_opts=compression_level)

            # Add events and metadata as attributes
            with open(os.path.join(sample_path, 'events.json'), 'r') as f:
                events = json.load(f)
            hf.attrs['events'] = json.dumps(events)

            with open(os.path.join(sample_path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            hf.attrs['metadata'] = json.dumps(metadata)

            # Check for and add additional metadata from metadata.yaml
            yaml_path = os.path.join(sample_path, 'metadata.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    additional_metadata = yaml.safe_load(f)
                hf.attrs['additional_metadata'] = json.dumps(additional_metadata)

        if verbose:
            compressed_size = os.path.getsize(output_file)
            compression_factor = original_size / compressed_size
            return {
                'sample': os.path.basename(sample_path),
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_factor': compression_factor
            }
        return None
    except Exception as e:
        return {'sample': os.path.basename(sample_path), 'error': str(e)}


def process_dataset(dataset_root, verbose, num_processes, subsample_factor, compression_level=9):
    sample_paths = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) if
                    os.path.isdir(os.path.join(dataset_root, d))]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(partial(process_sample, verbose=verbose, subsample_factor=subsample_factor,
                                              compression_level=compression_level), sample_paths),
                            total=len(sample_paths), desc="Processing samples"))

    failed_samples = [result['sample'] for result in results if result and 'error' in result]

    if verbose:
        for result in results:
            if result and 'error' not in result:
                print(f"Sample: {result['sample']}")
                print(f"Original size: {result['original_size'] / 1024:.2f} KB")
                print(f"Compressed size: {result['compressed_size'] / 1024:.2f} KB")
                print(f"Compression factor: {result['compression_factor']:.2f}x")
                print("--------------------")

    # Print and log failed samples
    if failed_samples:
        print("Failed samples:")
        for sample in failed_samples:
            print(sample)

        log_file = os.path.join(dataset_root, 'log.txt')
        with open(log_file, 'w') as f:
            f.write("Failed samples:\n")
            for sample in failed_samples:
                f.write(f"{sample}\n")
        print(f"Failed samples have been logged in {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dataset to HDF5 format using gzip compression with multiprocessing and subsampling")
    parser.add_argument("dataset_root", help="Path to the dataset root")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print compression information for each sample")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count(),
                        help="Number of parallel jobs to run (default: number of CPU cores)")
    parser.add_argument("-s", "--subsample", type=int, default=1,
                        help="Subsample factor for non-rgba data (default: 1)")
    parser.add_argument("-c", "--compression", type=int, default=4, choices=range(0, 10),
                        help="Gzip compression level (0-9, default: 9)")
    args = parser.parse_args()

    process_dataset(args.dataset_root, args.verbose, args.jobs, args.subsample, args.compression)
    print("Conversion completed.")
