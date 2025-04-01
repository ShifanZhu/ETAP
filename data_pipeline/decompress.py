import os
import json
import h5py
import numpy as np
import cv2
import argparse

def save_image(data, file_path, data_type):
    if file_path.endswith('.tiff') or file_path.endswith('.tif'):
        cv2.imwrite(file_path, data)
    else:
        if data_type in ['backward_flow', 'forward_flow'] or data.dtype == np.uint16:
            cv2.imwrite(file_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            #cv2.imwrite(file_path, data)
        elif len(data.shape) == 2:  # Grayscale image
            cv2.imwrite(file_path, data)
        elif len(data.shape) == 3:
            if data.shape[2] == 3:  # RGB image
                cv2.imwrite(file_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            elif data.shape[2] == 4:  # RGBA image
                cv2.imwrite(file_path, cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA))
            else:
                raise ValueError(f"Unsupported image shape: {data.shape}")
        else:
            raise ValueError(f"Unsupported image shape: {data.shape}")

def extract_sample(hdf5_path, output_folder):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as hf:
        # Extract datasets
        for data_type in ['backward_flow', 'depth', 'forward_flow', 'normal', 'object_coordinates', 'rgba', 'segmentation']:
            if data_type in hf:
                data = hf[data_type][:]
                if len(data.shape) == 3:  # Single image or multiple 2D frames
                    for i, frame in enumerate(data):
                        file_name = f"{data_type}_{i:06d}.{'tiff' if data_type == 'depth' else 'png'}"
                        file_path = os.path.join(output_folder, file_name)
                        save_image(frame, file_path, data_type)
                elif len(data.shape) == 4:  # Multiple color frames
                    for i, frame in enumerate(data):
                        file_name = f"{data_type}_{i:06d}.{'tiff' if data_type == 'depth' else 'png'}"
                        file_path = os.path.join(output_folder, file_name)
                        save_image(frame, file_path, data_type)
                else:
                    print(f"Warning: Unexpected shape for {data_type}: {data.shape}")
            else:
                print(f"Warning: {data_type} not found in HDF5 file")

        # Extract metadata and events
        if 'metadata' in hf.attrs:
            metadata = json.loads(hf.attrs['metadata'])
            with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

        if 'events' in hf.attrs:
            events = json.loads(hf.attrs['events'])
            with open(os.path.join(output_folder, 'events.json'), 'w') as f:
                json.dump(events, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sample data from HDF5 file to original file structure")
    parser.add_argument("hdf5_path", help="Path to the input HDF5 file")
    parser.add_argument("output_folder", help="Path to the output folder for extracted files")
    args = parser.parse_args()

    extract_sample(args.hdf5_path, args.output_folder)
    print("Extraction completed.")
