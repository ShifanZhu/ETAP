#!/usr/bin/env python3
"""
ETAP online tracking with LCM I/O (single-file, robust)

Changes vs previous:
- Filters out zero rows in queries (and skips if nothing left)
- Accumulates only frames inside [start_us, end_us]
- Skips save/visualize cleanly if nothing processed
- Aligns per-chunk timestamps with model outputs (min_len)
- Optional window margin to avoid boundary misses
"""

import argparse
import subprocess
from pathlib import Path
import sys
import numpy as np
import yaml
import torch
from tqdm import tqdm
import cv2
import datetime
import time
import threading
from typing import Optional, List

# -----------------------------
# Make sure we can import msgs
# -----------------------------
# Expect generated files under ./lcm_gen/msgs or scripts/lcm_types/msgs
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root / "lcm_gen"))
sys.path.append(str(repo_root / "scripts" / "lcm_types"))

# Your repo modules
sys.path.append(str(repo_root))

from src.data.modules import DataModuleFactory
from src.model.etap.model import Etap
from src.utils import Visualizer, normalize_and_expand_channels, make_grid

# LCM
import lcm
try:
    from msgs import TrackingCommand, TrackingUpdate
except Exception:
    from lcm_types.msgs import TrackingCommand, TrackingUpdate  # type: ignore

torch.set_float32_matmul_precision('high')


def write_points_to_file(points, timestamps, filepath):
    """Write tracking points to a file.

    points: [T, N, 2] (numpy)
    timestamps: [T] seconds (numpy)
    """
    T, N, _ = points.shape
    with open(filepath, 'w') as f:
        for t in range(T):
            for n in range(N):
                x, y = points[t, n]
                f.write(f"{n} {float(timestamps[t]):.9f} {x:.9f} {y:.9f}\n")


def get_git_commit_hash():
    """Get the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error obtaining git commit hash: {e.output.decode().strip()}")
        return "unknown"


def normalize_voxels(voxels):
    """Channelwise std-mean normalization on voxels, ignoring zeros."""
    mask = voxels != 0
    denom = mask.sum(dim=(0, 2, 3), keepdim=True).clamp_min(1)
    mean = voxels.sum(dim=(0, 2, 3), keepdim=True) / denom
    var = ((voxels - mean) ** 2 * mask).sum(dim=(0, 2, 3), keepdim=True) / denom
    std = torch.sqrt(var + 1e-8)
    return torch.where(mask, (voxels - mean) / std, voxels)


def to_us(ts_s: float) -> int:
    return int(round(float(ts_s) * 1e6))


def to_s(ts_us: int) -> float:
    return float(ts_us) * 1e-6


class LcmBridge:
    """Background LCM I/O: subscribe to TrackingCommand, publish TrackingUpdate."""
    def __init__(self, cmd_topic: str, upd_topic: str):
        self.lc = lcm.LCM()
        self.cmd_topic = cmd_topic
        self.upd_topic = upd_topic

        self._lock = threading.Lock()
        self._pending_cmd: Optional[TrackingCommand] = None
        self._shutdown = False

        self.lc.subscribe(self.cmd_topic, self._on_cmd)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        self._next_id = 10_000_000  # allocator for any "new" features you might add

    def _on_cmd(self, channel: str, data: bytes):
        msg = TrackingCommand.decode(data)
        with self._lock:
            self._pending_cmd = msg

    def _spin(self):
        while not self._shutdown:
            try:
                self.lc.handle_timeout(50)
            except Exception:
                pass

    def get_pending_request(self) -> Optional[TrackingCommand]:
        with self._lock:
            cmd = self._pending_cmd
            self._pending_cmd = None
            return cmd

    def publish_update(self, ts_us: int, ids: List[int], xs: List[float], ys: List[float]):
        msg = TrackingUpdate()
        msg.timestamp_us = int(ts_us)
        msg.feature_ids = list(map(int, ids))
        msg.feature_x = list(map(float, xs))
        msg.feature_y = list(map(float, ys))
        self.lc.publish(self.upd_topic, msg.encode())

    def allocate_ids(self, n: int) -> List[int]:
        out = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        return out

    def close(self):
        self._shutdown = True
        self._thread.join(timeout=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/exe/inference_online/feature_tracking_cear.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    # LCM overrides (optional – can also be set in config['common'])
    parser.add_argument('--lcm_cmd_topic', type=str, default=None)
    parser.add_argument('--lcm_upd_topic', type=str, default=None)
    # Window margin (ns) to catch boundary frames
    parser.add_argument('--window_margin_ns', type=int, default=1_000_000)  # 1 ms
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

    # --- LCM integration ---
    lcm_cmd_topic = args.lcm_cmd_topic or config['common'].get('lcm_cmd_topic', 'TRACKING_COMMAND')
    lcm_upd_topic = args.lcm_upd_topic or config['common'].get('lcm_upd_topic', 'TRACKING_UPDATE')
    vis_thresh = float(config['common'].get('visibility_threshold', 0.0))
    lcm_bridge = LcmBridge(lcm_cmd_topic, lcm_upd_topic)

    try:
        for dataset in data_module.test_datasets:
            print("dataset", dataset)
            sequence_name = dataset.subsequence_name

            # ---- Wait for a command for this sequence ----
            print(f"[{sequence_name}] Waiting for LCM command on topic '{lcm_cmd_topic}'...")
            cmd = None
            while cmd is None:
                cmd = lcm_bridge.get_pending_request()
                time.sleep(0.01)

            print(f"Received command for '{sequence_name}': "
                  f"[{cmd.start_time_us} .. {cmd.end_time_us}], {len(cmd.feature_ids)} seeds")

            # Validate seed arrays
            if len(cmd.feature_ids) != len(cmd.feature_x) or len(cmd.feature_ids) != len(cmd.feature_y):
                raise ValueError("LCM TrackingCommand arrays must be the same length")

            # Build queries tensor from seeds: [N, 3] = [t_idx, x, y], t_idx=0 at start
            seed_ids = list(map(int, cmd.feature_ids))
            seed_xy = torch.tensor(np.stack([cmd.feature_x, cmd.feature_y], axis=1),
                                   dtype=torch.float32, device=device)  # [N,2]
            seed_t = torch.zeros((seed_xy.shape[0], 1), dtype=torch.int64, device=device)
            original_queries = torch.cat([seed_t, seed_xy], dim=1)  # [N,3]
            print("original_queries:", original_queries)

            # Optionally add support points (get unique IDs for them)
            if add_support_points:
                support_query_xy = torch.from_numpy(
                    make_grid(height, width, stride=support_point_stride)
                ).float().to(device)
                support_num_queries = support_query_xy.shape[0]
                support_ids = lcm_bridge.allocate_ids(support_num_queries)
                support_query_t = torch.zeros(support_num_queries, dtype=torch.int64, device=device)
                support_queries = torch.cat([support_query_t[:, None], support_query_xy], dim=1)
                queries = torch.cat([original_queries, support_queries], dim=0)
                all_ids = seed_ids + support_ids
            else:
                queries = original_queries
                support_num_queries = 0
                all_ids = seed_ids

            # Remove all-zero rows from queries (and keep IDs aligned)
            nonzero_mask = (queries != 0).any(dim=1)
            queries = queries[nonzero_mask]
            all_ids = [id_ for keep, id_ in zip(nonzero_mask.tolist(), all_ids) if keep]

            if queries.numel() == 0 or queries.shape[0] == 0:
                print(f"[WARN] No nonzero queries for '{sequence_name}'. Skipping.")
                continue

            print("queries (filtered):", queries)

            # Online tracking init
            tracker.init_video_online_processing()
            event_visus = None

            # Accumulators for outputs in the requested time window
            processed_any = False
            coords_all = []  # list of [Ti, N, 2] CPU tensors
            vis_all = []     # list of [Ti, N] CPU tensors
            ts_all = []      # list of [Ti] CPU tensors (seconds)

            # Expand window edges slightly to avoid boundary misses
            start_us = int(cmd.start_time_us) - int(args.window_margin_ns)
            end_us = int(cmd.end_time_us) + int(args.window_margin_ns)

            # Iterate dataset
            for sample, start_idx in tqdm(dataset, desc=f'Predicting {sequence_name}'):
                # assert start_idx == tracker.online_ind
                voxels = sample.voxels.to(device)
                step = voxels.shape[0] // 2  # as in your original code

                # Chunk timestamps (seconds → us) for publish/window check
                ts_chunk_s = sample.timestamps[-step:]                 # [step]
                ts_chunk_us = (ts_chunk_s * 1e6).round().long()        # [step]

                # Skip chunk if fully outside requested window
                if (ts_chunk_us[-1] < start_us) or (ts_chunk_us[0] > end_us):
                    # print(f"[{sequence_name}] Skipping chunk: "
                    #     f"ts_s [{ts_chunk_s[0]:.6f} .. {ts_chunk_s[-1]:.6f}] outside [{to_s(start_us):.6f} .. {to_s(end_us):.6f}]")
                    continue

                voxels = normalize_voxels(voxels)

                with torch.no_grad():
                    results = tracker(
                        video=voxels[None],
                        queries=queries[None],
                        is_online=True,
                        iters=6
                    )

                # results: [B, T, Q, ...]
                coords_predicted = results['coords_predicted'].clone()  # [1, T, Q, 2]
                vis_logits = results['vis_predicted']                   # [1, T, Q]
                # print("predicted coords:", coords_predicted)

                # Remove support points from what we keep/publish (keep only seeds)
                if support_num_queries > 0:
                    coords_predicted = coords_predicted[:, :, :-support_num_queries]
                    vis_logits = vis_logits[:, :, :-support_num_queries]

                coords_predicted = coords_predicted[0]  # [T, N, 2]
                vis_logits = vis_logits[0]             # [T, N]

                # Align lengths (some models may output T different from step)
                Tcur = coords_predicted.shape[0]
                min_len = min(Tcur, ts_chunk_s.shape[0], ts_chunk_us.shape[0])
                if min_len <= 0:
                    continue
                coords_predicted = coords_predicted[:min_len]
                vis_logits = vis_logits[:min_len]
                ts_chunk_s = ts_chunk_s[:min_len]
                ts_chunk_us = ts_chunk_us[:min_len]

                # Select frames within window for saving
                frame_mask = (ts_chunk_us >= start_us) & (ts_chunk_us <= end_us)
                if frame_mask.any():
                    keep_idx = torch.nonzero(frame_mask, as_tuple=False).squeeze(-1)
                    coords_keep = coords_predicted[keep_idx].cpu()  # [Ti, N, 2]
                    vis_keep = vis_logits[keep_idx].cpu()           # [Ti, N]
                    ts_keep_s = ts_chunk_s[keep_idx].cpu()          # [Ti]
                    coords_all.append(coords_keep)
                    vis_all.append(vis_keep)
                    ts_all.append(ts_keep_s)
                    processed_any = True

                # Publish per-frame within window (optionally gate by visibility)
                for ti in range(min_len):
                    ts_us = int(ts_chunk_us[ti].item())
                    if ts_us < start_us or ts_us > end_us:
                        continue
                    xy = coords_predicted[ti].detach().cpu().numpy()  # [N,2]
                    vis = vis_logits[ti].detach().cpu().numpy()       # [N]

                    if vis_thresh > 0.0:
                        mask = vis >= vis_thresh
                        if not mask.any():
                            continue
                        ids_out = [all_ids[i] for i, m in enumerate(mask) if m]
                        xs_out = [float(xy[i, 0]) for i, m in enumerate(mask) if m]
                        ys_out = [float(xy[i, 1]) for i, m in enumerate(mask) if m]
                    else:
                        ids_out = all_ids
                        xs_out = xy[:, 0].tolist()
                        ys_out = xy[:, 1].tolist()
                        # Ensure all lists are exactly 50 items
                        num_features = 50
                        out_len = len(ids_out)
                        # Ensure all lists are exactly 50 items
                        num_features = 50
                        out_len = len(ids_out)
                        # Pad or truncate each list independently to ensure length 50
                        if out_len < num_features:
                          pad_count = num_features - out_len
                          ids_out += [0] * pad_count
                        else:
                          ids_out = ids_out[:num_features]
                        xs_out = (xs_out + [0.0] * num_features)[:num_features]
                        ys_out = (ys_out + [0.0] * num_features)[:num_features]
                    lcm_bridge.publish_update(ts_us, ids_out, xs_out, ys_out)

                # Optional visualization buffer (not strictly windowed)
                event_visu = normalize_and_expand_channels(voxels.sum(dim=1))
                event_visus = torch.cat([event_visus, event_visu[-step:]]) if event_visus is not None else event_visu

            # ---- Save predictions only if something was processed ----
            if not processed_any:
                print(f"[WARN] No frames processed for {sequence_name} in time window "
                      f"[{start_us} .. {end_us}]. Skipping save/vis.")
                continue

            coords_out = torch.cat(coords_all, dim=0)   # [Tsel, N, 2]
            vis_out = torch.cat(vis_all, dim=0)         # [Tsel, N]
            ts_out = torch.cat(ts_all, dim=0)           # [Tsel] seconds

            output_npz = save_dir / f'{sequence_name}.npz'
            np.savez(
                output_npz,
                coords_predicted=coords_out.numpy(),
                vis_logits=vis_out.numpy(),
                timestamps=ts_out.numpy()
            )

            # Save text version
            output_txt = save_dir / f'{sequence_name}.txt'
            write_points_to_file(
                coords_out.numpy(),
                ts_out.numpy(),
                output_txt
            )

            # Visualization (best-effort: uses accumulated event_visus buffer)
            viz.visualize(
                event_visus[None] if event_visus is not None else None,
                coords_out[None],
                filename=sequence_name
            )

        print('Done.')

    finally:
        lcm_bridge.close()


if __name__ == '__main__':
    main()
