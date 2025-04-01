import sys
import argparse
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
import src.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description='Compute TAPVid metrics for point tracking')
    parser.add_argument('--gt_dir', type=str, default='data/e2d2/231025_110210_fidget5_high_exposure',
                        help='Directory containing ground truth files')
    parser.add_argument('--pred_dir', type=str, default='output/inference/tap_e2d2',
                        help='Directory containing prediction files')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    error_threshold_range = 2 * np.array([1, 2, 4, 8, 16])

    gt_tracks_path = gt_dir / 'gt_positions.npy'
    gt_tracks = np.load(gt_tracks_path)

    query_path = gt_dir / 'queries.npy'
    query_points = np.load(query_path)
    query_t = np.zeros((query_points.shape[0], 1)) # All query points are at t = 0
    query_points = np.concatenate([query_t, query_points], axis=1)

    pred_path = Path(pred_dir) / '231025_110210_fidget5_high_exposure.npz'
    pred = np.load(pred_path)
    coords_predicted = pred['coords_predicted']
    vis_logits = pred['vis_logits']
    vis_predicted = vis_logits > 0.8

    gt_tracks_formatted = np.expand_dims(np.transpose(gt_tracks, (1, 0, 2)), axis=0)
    coords_predicted_formatted = np.expand_dims(np.transpose(coords_predicted, (1, 0, 2)), axis=0)

    # Create occlusion masks (assuming no occlusions in gt, using vis_predicted for predictions)
    num_points, num_frames = gt_tracks.shape[1], gt_tracks.shape[0]
    occluded_gt = np.zeros((1, num_points, num_frames), dtype=bool)
    occluded_pred = np.expand_dims(~np.transpose(vis_predicted, (1, 0)), axis=0) # Assuming vis_predicted is visibility, not occlusion

    tap_metrics = utils.compute_tapvid_metrics(
        query_points=np.expand_dims(query_points, axis=0), # Add batch dimension
        gt_occluded=occluded_gt,
        gt_tracks=gt_tracks_formatted,
        pred_occluded=occluded_pred,
        pred_tracks=coords_predicted_formatted,
        query_mode='first',
        thresholds=error_threshold_range
    )
    print("TAPVid Metrics:")
    for k, v in tap_metrics.items():
        print(f"{k}: {v}")
