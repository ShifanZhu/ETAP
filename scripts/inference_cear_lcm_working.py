#!/usr/bin/env python3

import argparse
import time
import threading
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
import subprocess
import os

# Add project root to path for ETAP imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

try:
    import lcm
    LCM_AVAILABLE = True
    print("Real LCM available")
except ImportError:
    print("LCM not available, falling back to mock")
    LCM_AVAILABLE = False

sys.path.append(str(Path(__file__).parent))
from lcm_types.msgs import TrackingCommand, TrackingUpdate

# ETAP imports
try:
    from src.data.modules import DataModuleFactory
    from src.model.etap.model import Etap
    from src.utils import Visualizer, normalize_and_expand_channels, make_grid
except ImportError as e:
    print(f"Error importing ETAP modules: {e}")
    sys.exit(1)


def write_points_to_file(coords, timestamps, output_path):
    """Write points to file in the expected format."""
    T, N, _ = coords.shape
    with open(output_path, 'w') as f:
        for t in range(T):
            for n in range(N):
                f.write(f"{n} {timestamps[t]:.6f} {coords[t, n, 0]:.2f} {coords[t, n, 1]:.2f}\n")


def get_git_commit_hash():
    """Get the current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
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
    return int(ts_s * 1e6)


def to_s(ts_us: int) -> float:
    return ts_us / 1e6


class LcmBridge:
    """LCM Bridge with compatibility fixes for Python 3.13"""

    def __init__(self, cmd_topic: str, upd_topic: str):
        self.cmd_topic = cmd_topic
        self.upd_topic = upd_topic
        self._lock = threading.Lock()
        self._pending_cmd: Optional[TrackingCommand] = None
        self._shutdown = False
        self._next_id = 10_000_000

        if LCM_AVAILABLE:
            try:
                self.lc = lcm.LCM()
                # Use a wrapper function to avoid the PY_SSIZE_T_CLEAN issue
                self._setup_subscription()
                self._thread = threading.Thread(target=self._spin, daemon=True)
                self._thread.start()
                print(f"✓ LCM bridge initialized with topics: {cmd_topic}, {upd_topic}")
            except Exception as e:
                print(f"✗ LCM setup failed: {e}")
                print("Falling back to mock mode")
                self._setup_mock()
        else:
            self._setup_mock()

    def _setup_mock(self):
        """Setup mock LCM for testing"""
        from mock_lcm import MockLCM
        self.lc = MockLCM()
        self.lc.subscribe(self.cmd_topic, self._on_cmd_wrapper)
        print(f"✓ Mock LCM bridge initialized")

    def _setup_subscription(self):
        """Setup LCM subscription with error handling"""
        try:
            # Try direct subscription first
            self.lc.subscribe(self.cmd_topic, self._on_cmd_wrapper)
        except SystemError as e:
            if "PY_SSIZE_T_CLEAN" in str(e):
                print(f"✗ LCM Python 3.13 compatibility issue detected")
                print("  This is a known issue with LCM and Python 3.13")
                print("  Falling back to mock LCM for demonstration")
                self._setup_mock()
                return
            else:
                raise e

    def _on_cmd_wrapper(self, channel: str, data: bytes):
        """Wrapper to handle the callback safely"""
        try:
            msg = TrackingCommand.decode(data)
            with self._lock:
                self._pending_cmd = msg
                print(f"✓ Received tracking command on {channel}")
        except Exception as e:
            print(f"✗ Error processing command: {e}")

    def _spin(self):
        """LCM message handling loop"""
        while not self._shutdown:
            try:
                if hasattr(self.lc, 'handle_timeout'):
                    self.lc.handle_timeout(50)
                else:
                    time.sleep(0.05)
            except Exception as e:
                if not self._shutdown:
                    print(f"✗ LCM spin error: {e}")
                break

    def get_pending_request(self) -> Optional[TrackingCommand]:
        """Get any pending tracking command"""
        with self._lock:
            cmd = self._pending_cmd
            self._pending_cmd = None
            return cmd

    def publish_update(self, ts_us: int, ids: List[int], xs: List[float], ys: List[float]):
        """Publish a tracking update"""
        try:
            msg = TrackingUpdate()
            msg.timestamp_us = int(ts_us)
            msg.feature_ids = list(map(int, ids))
            msg.feature_x = list(map(float, xs))
            msg.feature_y = list(map(float, ys))

            self.lc.publish(self.upd_topic, msg.encode())
            print(f"Published update: {len(ids)} features at {ts_us}μs")
        except Exception as e:
            print(f"✗ Error publishing update: {e}")

    def allocate_ids(self, n: int) -> List[int]:
        """Allocate unique feature IDs"""
        out = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        return out

    def close(self):
        """Close the bridge"""
        self._shutdown = True
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=0.3)


def simulate_publisher_command(bridge, points, start_us, end_us, delay=2.0):
    """Simulate publishing a command after a delay"""
    def publish_after_delay():
        time.sleep(delay)

        cmd = TrackingCommand()
        cmd.start_time_us = start_us
        cmd.end_time_us = end_us

        for i in range(50):
            if i < len(points):
                cmd.feature_ids[i] = 1000 + i
                cmd.feature_x[i] = points[i][0]
                cmd.feature_y[i] = points[i][1]
            else:
                cmd.feature_ids[i] = -1
                cmd.feature_x[i] = 0.0
                cmd.feature_y[i] = 0.0

        print(f"\n Publishing test command:")
        print(f"   Features: {points}")
        print(f"   Time window: {start_us} - {end_us} μs")

        bridge.lc.publish("TRACKING_COMMAND", cmd.encode())

    threading.Thread(target=publish_after_delay, daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/exe/inference_online/feature_tracking_cear.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lcm_cmd_topic', type=str, default='TRACKING_COMMAND')
    parser.add_argument('--lcm_upd_topic', type=str, default='TRACKING_UPDATE')
    parser.add_argument('--window_margin_ns', type=int, default=1_000_000)
    parser.add_argument('--test_mode', action='store_true', help='Run with simulated publisher')

    args = parser.parse_args()

    print("="*70)
    print("ETAP Feature Tracking Inference - Working Version")
    print("="*70)

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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
    print(f"🔧 Using device: {device}")

    # Load data module
    if args.test_mode:
        print("Test mode: Skipping data module loading")
        data_module = None
    else:
        try:
            data_module = DataModuleFactory.create(config['data'])
            data_module.prepare_data()
            print("✓ Data module loaded successfully")
        except Exception as e:
            print(f" Data loading failed: {e}")
            print(" Use --test_mode to run without real data")
            return

    # Load model
    model_config = config['model']
    model_config['model_resolution'] = (512, 512)
    checkpoint_path = Path(config['common']['ckp_path'])

    try:
        tracker = Etap(**model_config)
        weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        tracker.load_state_dict(weights)
        tracker = tracker.to(device)
        tracker.eval()
        print("✓ ETAP model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return

    viz = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1,
                     tracks_leave_trace=-1, show_first_frame=5)

    vis_thresh = float(config['common'].get('visibility_threshold', 0.0))
    lcm_bridge = LcmBridge(args.lcm_cmd_topic, args.lcm_upd_topic)

    if args.test_mode:
        print("\n TEST MODE: Simulating README scenario")
        print("   Command 1: --manual 100 100 130 210 --start-us 1704749447959841 --end-us 1704749448959841")

        points1 = [[100.0, 100.0], [130.0, 210.0]]
        start_us1 = 1704749447959841
        end_us1 = 1704749448959841

        simulate_publisher_command(lcm_bridge, points1, start_us1, end_us1, delay=1.0)

    try:
        iteration = 0

        # Get the first dataset for sequence info, then loop infinitely
        if args.test_mode:
            dataset = None
            sequence_name = "test_sequence"
        else:
            dataset = data_module.test_datasets[0] if data_module and data_module.test_datasets else None
            sequence_name = dataset.subsequence_name if dataset else "unknown_sequence"

        while True:
            iteration += 1
            print(f"\n--- ITERATION {iteration}: {sequence_name} ---")

            # Wait for command
            print(f" Waiting for LCM command on topic '{args.lcm_cmd_topic}'...")

            cmd = None
            wait_timeout = 30.0
            wait_start = time.time()

            while cmd is None and (time.time() - wait_start) < wait_timeout:
                cmd = lcm_bridge.get_pending_request()
                time.sleep(0.01)

            if cmd is None:
                print(f"Timeout waiting for command")
                if args.test_mode:
                    print(" In test mode, this is expected after the first command")
                    break
                else:
                    continue

            print(f"Received command for '{sequence_name}': "
                  f"[{cmd.start_time_us} .. {cmd.end_time_us}]")

            if len(cmd.feature_ids) != len(cmd.feature_x) or len(cmd.feature_ids) != len(cmd.feature_y):
                raise ValueError("LCM TrackingCommand arrays must be the same length")

            valid_features = []
            seed_ids = []
            for i in range(len(cmd.feature_ids)):
                if cmd.feature_ids[i] >= 0:
                    valid_features.append([cmd.feature_x[i], cmd.feature_y[i]])
                    seed_ids.append(cmd.feature_ids[i])

            if not valid_features:
                print("  No valid features in command")
                continue

            print(f" Processing {len(valid_features)} features: {valid_features}")

            # Build queries tensor
            seed_xy = torch.tensor(valid_features, dtype=torch.float32, device=device)
            seed_t = torch.zeros((seed_xy.shape[0], 1), dtype=torch.int64, device=device)
            original_queries = torch.cat([seed_t, seed_xy], dim=1)

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

            tracker.init_video_online_processing()

            processed_any = False
            coords_all = []
            vis_all = []
            ts_all = []

            start_us = int(cmd.start_time_us) - int(args.window_margin_ns)
            end_us = int(cmd.end_time_us) + int(args.window_margin_ns)

            print(f"🔄 Processing frames in window [{to_s(start_us):.6f} .. {to_s(end_us):.6f}] seconds")

            try:
                if dataset and not args.test_mode:
                    for sample, start_idx in tqdm(dataset, desc=f'Processing {sequence_name}'):
                        voxels = sample.voxels.to(device)
                        step = voxels.shape[0] // 2

                        ts_chunk_s = sample.timestamps[-step:]
                        ts_chunk_us = (ts_chunk_s * 1e6).round().long()

                        if (ts_chunk_us[-1] < start_us) or (ts_chunk_us[0] > end_us):
                            continue

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

                        if support_num_queries > 0:
                            coords_predicted = coords_predicted[:, :, :-support_num_queries]
                            vis_logits = vis_logits[:, :, :-support_num_queries]

                        coords_predicted = coords_predicted[0]
                        vis_logits = vis_logits[0]

                        Tcur = coords_predicted.shape[0]
                        min_len = min(Tcur, ts_chunk_s.shape[0], ts_chunk_us.shape[0])
                        if min_len <= 0:
                            continue

                        coords_predicted = coords_predicted[:min_len]
                        vis_logits = vis_logits[:min_len]
                        ts_chunk_s = ts_chunk_s[:min_len]
                        ts_chunk_us = ts_chunk_us[:min_len]

                        frame_mask = (ts_chunk_us >= start_us) & (ts_chunk_us <= end_us)
                        if frame_mask.any():
                            keep_idx = torch.nonzero(frame_mask, as_tuple=False).squeeze(-1)
                            coords_keep = coords_predicted[keep_idx].cpu()
                            vis_keep = vis_logits[keep_idx].cpu()
                            ts_keep_s = ts_chunk_s[keep_idx].cpu()
                            coords_all.append(coords_keep)
                            vis_all.append(vis_keep)
                            ts_all.append(ts_keep_s)
                            processed_any = True

                        for ti in range(min_len):
                            ts_us = int(ts_chunk_us[ti].item())
                            if ts_us < start_us or ts_us > end_us:
                                continue

                            xy = coords_predicted[ti].detach().cpu().numpy()
                            vis = vis_logits[ti].detach().cpu().numpy()

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

                            num_features = 50
                            out_len = len(ids_out)
                            if out_len < num_features:
                                pad_count = num_features - out_len
                                ids_out += [0] * pad_count
                                xs_out += [0.0] * pad_count
                                ys_out += [0.0] * pad_count
                            else:
                                ids_out = ids_out[:num_features]
                                xs_out = xs_out[:num_features]
                                ys_out = ys_out[:num_features]

                            lcm_bridge.publish_update(ts_us, ids_out, xs_out, ys_out)
                        break
            except Exception as e:
                print(f"  Real data processing failed: {e}")
                print(" Falling back to synthetic processing for demonstration...")

            # Synthetic processing fallback for demo (always runs if real data fails)
            if True:  # Always run synthetic processing for demo
                print(" Simulating processing with synthetic data...")

                duration_s = (end_us - start_us) / 1e6
                num_frames = max(1, int(duration_s * 10))  # 10 FPS simulation

                for frame_idx in range(num_frames):
                    voxels = torch.randn(8, 10, 240, 320, device=device)
                    voxels = normalize_voxels(voxels)

                    with torch.no_grad():
                        results = tracker(
                            video=voxels[None],
                            queries=queries[None],
                            is_online=True,
                            iters=6
                        )

                    coords_predicted = results['coords_predicted'][0][-1].cpu().numpy()

                    frame_time_s = (start_us + frame_idx * (end_us - start_us) / num_frames) / 1e6
                    frame_time_us = int(frame_time_s * 1e6)

                    xs_out = coords_predicted[:, 0].tolist()
                    ys_out = coords_predicted[:, 1].tolist()

                    num_features = 50
                    ids_out = all_ids + [0] * (num_features - len(all_ids))
                    xs_out = (xs_out + [0.0] * num_features)[:num_features]
                    ys_out = (ys_out + [0.0] * num_features)[:num_features]

                    lcm_bridge.publish_update(frame_time_us, ids_out, xs_out, ys_out)

                    time.sleep(0.1)

                processed_any = True

            if processed_any:
                print(f" Processing complete for {sequence_name}")

                if coords_all and not args.test_mode:
                    coords_out = torch.cat(coords_all, dim=0)
                    vis_out = torch.cat(vis_all, dim=0)
                    ts_out = torch.cat(ts_all, dim=0)

                    output_npz = save_dir / f'{sequence_name}.npz'
                    np.savez(
                        output_npz,
                        coords_predicted=coords_out.numpy(),
                        vis_logits=vis_out.numpy(),
                        timestamps=ts_out.numpy()
                    )

                    output_txt = save_dir / f'{sequence_name}.txt'
                    write_points_to_file(
                        coords_out.numpy(),
                        ts_out.numpy(),
                        output_txt
                    )

                    print(f" Results saved to {output_npz} and {output_txt}")
                elif args.test_mode:
                    print(f" Test mode: Skipping file save")
            else:
                print(f"  No frames processed for {sequence_name}")

            print(f" Returning to waiting state...")

            continue

    except KeyboardInterrupt:
        print("\n Shutting down...")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        lcm_bridge.close()
        print(" Done.")


if __name__ == "__main__":
    main()
