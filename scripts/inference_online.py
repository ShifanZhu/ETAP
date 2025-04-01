import argparse
import subprocess
from pathlib import Path
import sys
import numpy as np

import yaml
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.modules import DataModuleFactory
from src.model.etap.model import Etap
from src.utils import Visualizer, normalize_and_expand_channels, make_grid

torch.set_float32_matmul_precision('high')


def write_points_to_file(points, timestamps, filepath):
    """Write tracking points to a file."""
    T, N, _ = points.shape

    with open(filepath, 'w') as f:
        for t in range(T):
            for n in range(N):
                x, y = points[t, n]
                f.write(f"{n} {timestamps[t]:.9f} {x:.9f} {y:.9f}\n")


def get_git_commit_hash():
    """Get the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error obtaining git commit hash: {e.output.decode().strip()}")
        return "unknown"


def normalize_voxels(voxels):
    """Perform channelwise std-mean normalization on voxels."""
    mask = voxels != 0
    mean = voxels.sum(dim=(0, 2, 3), keepdim=True) / mask.sum(dim=(0, 2, 3), keepdim=True)
    var = ((voxels - mean)**2 * mask).sum(dim=(0, 2, 3), keepdim=True) / mask.sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(var + 1e-8)
    return torch.where(mask, (voxels - mean) / std, voxels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/exe/inference_online/feature_tracking.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'output' / 'inference' / config['common']['exp_name']
    save_dir.mkdir(parents=True, exist_ok=True)

    config['runtime_info'] = {
        'command': ' '.join(sys.argv),
        'git_commit': get_git_commit_hash()
    }
    config_save_path = save_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    add_support_points = config['common'].get('add_support_points', False)

    if add_support_points:
        support_point_stride = config['common'].get('support_point_stride', 20)
        height, width = config['common']['height'], config['common']['width']

    device = torch.device(args.device)
    data_module = DataModuleFactory.create(config['data'])
    data_module.prepare_data()

    model_config = config['model']
    model_config['model_resolution'] = (512, 512)
    checkpoint_path = Path(config['common']['ckp_path'])

    tracker = Etap(**model_config)
    weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    tracker.load_state_dict(weights)
    tracker = tracker.to(device)
    tracker.eval()

    viz = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1,
                     tracks_leave_trace=-1, show_first_frame=5)

    for dataset in data_module.test_datasets:
        sequence_name = dataset.subsequence_name
        tracker.init_video_online_processing()
        timestamps_s = None
        
        original_queries = dataset.query_points.to(device)
        
        if add_support_points:
            support_query_xy = torch.from_numpy(make_grid(height, width, stride=support_point_stride)).float().to(device)
            support_num_queries = support_query_xy.shape[0]
            
            support_query_t = torch.zeros(support_num_queries, dtype=torch.int64, device=device)
            support_queries = torch.cat([support_query_t[:, None], support_query_xy], dim=1)
            
            queries = torch.cat([original_queries, support_queries])
            print(f"Added {support_num_queries} support points to {original_queries.shape[0]} original queries")
        else:
            queries = original_queries
            support_num_queries = 0

        event_visus = None

        for sample, start_idx in tqdm(dataset, desc=f'Predicting {sequence_name}'):
            assert start_idx == tracker.online_ind
            voxels = sample.voxels.to(device)
            step = voxels.shape[0] // 2

            if timestamps_s is None:
                timestamps_s = sample.timestamps
            else:
                timestamps_s = torch.cat([timestamps_s, sample.timestamps[-step:]])

            voxels = normalize_voxels(voxels)

            with torch.no_grad():
                results = tracker(
                    video=voxels[None], 
                    queries=queries[None],
                    is_online=True, 
                    iters=6
                )

            coords_predicted = results['coords_predicted'].clone()
            vis_logits = results['vis_predicted']
            
            # Remove support points
            if support_num_queries > 0:
                coords_predicted = coords_predicted[:, :, :-support_num_queries]
                vis_logits = vis_logits[:, :, :-support_num_queries]

            event_visu = normalize_and_expand_channels(voxels.sum(dim=1))
            event_visus = torch.cat([event_visus, event_visu[-step:]]) if event_visus is not None else event_visu        

        # Save predictions
        output_file = save_dir / f'{sequence_name}.npz'
        np.savez(
            output_file,
            coords_predicted=coords_predicted[0].cpu().numpy(),
            vis_logits=vis_logits[0].cpu().numpy(),
            timestamps=timestamps_s.cpu().numpy()
        )

        # Save predictions in feature tracking format
        output_file = save_dir / f'{sequence_name}.txt'
        write_points_to_file(
            coords_predicted.cpu().numpy()[0], 
            timestamps_s,
            output_file
        )

        viz.visualize(
            event_visus[None], 
            coords_predicted, 
            filename=sequence_name
        )

    print('Done.')

if __name__ == '__main__':
    main()