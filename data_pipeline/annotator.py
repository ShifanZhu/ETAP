import os
import argparse
import shutil
import json
import tensorflow as tf
import sys
import numpy as np

sys.path.append(os.path.join("kubric", "challenges"))

from PIL import Image
from point_tracking.dataset import add_tracks

from movi.movi_f import subsample_nearest_neighbor
from movi.movi_f import subsample_avg
from movi.movi_f import read_png
from movi.movi_f import read_tiff
from movi.movi_f import convert_float_to_uint16

def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--num_frames", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tracks_to_sample", type=int, default=2048)
    args = vars(ap.parse_args())

    _scene_dir       = args["scene_dir"]
    _width, _height  = args["resolution"], args["resolution"]
    _num_frames      = args["num_frames"]
    _target_size     = (_height, _width)
    _output_dir      = args["output_dir"]
    _tracks_to_sample= args["tracks_to_sample"]
    layers = ("rgba", "segmentation", "depth", "normal", "object_coordinates")

    # Load simulation output
    with tf.io.gfile.GFile(os.path.join(_scene_dir, 'metadata.json'), "r") as fp:
        metadata = json.load(fp)
    paths = {
        key: [os.path.join(_scene_dir, (f"{key}_{f:05d}.png")) for f in range (_num_frames)]
        for key in layers if key != "depth"
    }

    # Gather relevant data for point tracking annotation
    result = {}
    result["normal"] = tf.convert_to_tensor([subsample_nearest_neighbor(read_png(frame_path), _target_size) for frame_path in paths["normal"]], dtype=float)
    result["object_coordinates"] = tf.convert_to_tensor([subsample_nearest_neighbor(read_png(frame_path), _target_size) for frame_path in paths["object_coordinates"]])
    result["segmentations"] = tf.convert_to_tensor([subsample_nearest_neighbor(read_png(frame_path), _target_size) for frame_path in paths["segmentation"]])
    result["video"] = tf.convert_to_tensor([subsample_avg(read_png(frame_path), _target_size)[..., :3] for frame_path in paths["rgba"]])
    result["metadata"] = {}

    depth_paths = [os.path.join(_scene_dir, f"depth_{f:05d}.tiff") for f in range(_num_frames)]
    depth_frames = np.array([subsample_nearest_neighbor(read_tiff(frame_path), _target_size) for frame_path in depth_paths])
    depth_min, depth_max = np.min(depth_frames), np.max(depth_frames)

    result["depth"] = convert_float_to_uint16(depth_frames, depth_min, depth_max)
    result["metadata"]["depth_range"] = [depth_min, depth_max]

    result["instances"] = {}
    result["instances"]["bboxes_3d"] = tf.convert_to_tensor([np.array(obj["bboxes_3d"], np.float32) for obj in metadata["instances"]])
    result["instances"]["quaternions"] = tf.convert_to_tensor([np.array(obj["quaternions"], np.float32) for obj in metadata["instances"]])
    result["camera"] = {}

    result["camera"]["focal_length"] = metadata["camera"]["focal_length"]
    result["camera"]["sensor_width"] = metadata["camera"]["sensor_width"]
    result["camera"]["positions"]    = np.array(metadata["camera"]["positions"], np.float32)
    result["camera"]["quaternions"]  = np.array(metadata["camera"]["quaternions"], np.float32)

    # Annotate using add_tracks
    point_tracking = add_tracks(result, train_size=_target_size, random_crop=False, tracks_to_sample=_tracks_to_sample)
    video = point_tracking["video"].numpy()
    target_points = point_tracking["target_points"].numpy()
    occluded = point_tracking["occluded"].numpy()

    # Save annotations
    annotations = {"target_points": target_points, "occluded":occluded}
    np.save(os.path.join(_output_dir, "annotations.npy"), annotations)

if __name__ == "__main__":
    main()