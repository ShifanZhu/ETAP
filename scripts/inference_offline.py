import argparse
from pathlib import Path
import sys

import yaml
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.data.modules import DataModuleFactory
from src.model.etap.model import Etap
from src.utils import Visualizer, compute_tapvid_metrics


def normalize_and_expand_channels(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input must be a torch tensor")

    *batch_dims, height, width = image.shape

    if len(image.shape) < 2:
        raise ValueError("Input tensor must have at least shape (..., height, width)")

    image_flat = image.view(-1, height, width)

    min_val = image_flat.min()
    max_val = image_flat.max()
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    image_normalized = (image_flat - min_val) / range_val * 255
    image_rgb_flat = image_normalized.to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1)
    image_rgb = image_rgb_flat.view(*batch_dims, 3, height, width)

    return image_rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/exe/test_event_kubric/debug.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Load and process config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config['common'].get('output_dir', 'output/inference')
    output_dir = Path(output_dir) / config['common']['exp_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config['common'].get('checkpoint')

    data_module = DataModuleFactory.create(config['data'])
    data_module.prepare_data()
    test_set = data_module.test_dataset

    model_config = config['model']
    model_config['model_resolution'] = (512, 512)
    tracker = Etap(**model_config)
    weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    tracker.load_state_dict(weights)
    tracker = tracker.to(args.device)
    tracker.eval()

    viz = Visualizer(save_dir=output_dir, tracks_leave_trace=-1)

    seq_names = []
    sequence_metrics = []

    for i, sample in tqdm(enumerate(test_set), total=len(test_set), desc="Testing"):
        voxels = sample.voxels[None].to(args.device)
        trajs_g = sample.trajectory[None].to(args.device)
        vis_g = sample.visibility[None].to(args.device)
        queries = sample.query_points[None].to(args.device)
        B, T, C, H, W = voxels.shape
        _, _, N, D = trajs_g.shape

        with torch.no_grad():
            result = tracker(voxels, queries, iters=6)
            predictions, visibility = result['coords_predicted'], result['vis_predicted']

        if visibility.dtype != torch.bool:
            visibility = visibility > 0.8

        # Visualization
        projection = voxels.sum(2).cpu()
        lower_bound = torch.tensor(np.percentile(projection.cpu().numpy(), 2))
        upper_bound = torch.tensor(np.percentile(projection.cpu().numpy(), 98))
        projection_clipped = torch.clamp(projection, lower_bound, upper_bound)
        video = normalize_and_expand_channels(projection_clipped)

        viz.visualize(
            video=video,
            tracks=predictions,
            visibility=visibility,
            filename=f"pred_{sample.seq_name}",
        )
        viz.visualize(
            video=video,
            tracks=trajs_g,
            visibility=vis_g,
            filename=f"gt_{sample.seq_name}",
        )

        # Calculate metrics for this sequence
        queries_np = queries.cpu().numpy()
        trajs_g_np = trajs_g.cpu().numpy().transpose(0, 2, 1, 3)
        gt_occluded_np = ~vis_g.cpu().numpy().transpose(0, 2, 1)
        trajs_pred_np = predictions.cpu().numpy().transpose(0, 2, 1, 3)
        pred_occluded_np = ~visibility.cpu().numpy().transpose(0, 2, 1)

        seq_metrics = compute_tapvid_metrics(
            query_points=queries_np,
            gt_occluded=gt_occluded_np,
            gt_tracks=trajs_g_np,
            pred_occluded=pred_occluded_np,
            pred_tracks=trajs_pred_np,
            query_mode="first"
        )
        
        seq_names.append(sample.seq_name)
        sequence_metrics.append(seq_metrics)

    data = []
    for seq_name, seq_metrics in zip(seq_names, sequence_metrics):
        row = {'sequence': seq_name}
        row.update(seq_metrics)
        data.append(row)

    avg_metrics = {}
    metric_keys = sequence_metrics[0].keys()
    for key in metric_keys:
        values = [metrics[key] for metrics in sequence_metrics]
        avg_metrics[key] = np.mean(values)
    
    avg_row = {'sequence': 'average'}
    avg_row.update(avg_metrics)
    data.append(avg_row)
    
    df = pd.DataFrame(data)
    csv_path = output_dir / 'sequence_metrics.csv'
    df.to_csv(csv_path, index=False)

    print(f'Metrics saved to: {csv_path}')
    print('Done.')

if __name__ == '__main__':
    main()