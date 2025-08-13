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
import types
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
try:
    import lcm
    LCM_AVAILABLE = True
    print("Real LCM available")
except ImportError:
    print("LCM not available, falling back to mock")
    LCM_AVAILABLE = False
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
                print(f"✓ Received tracking message on {channel}")
        except Exception as e:
            print(f"✗ Error processing message: {e}")

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
        """Get any pending tracking message"""
        with self._lock:
            msg = self._pending_cmd
            self._pending_cmd = None
            return msg

    def publish_update(self, ts_us: int, ids: List[int], xs: List[float], ys: List[float]):
        """Publish a tracking update"""
        try:
            msg = TrackingUpdate()
            msg.timestamp_us = int(ts_us)
            msg.feature_ids = list(map(int, ids))
            msg.feature_x = list(map(float, xs))
            msg.feature_y = list(map(float, ys))

            self.lc.publish(self.upd_topic, msg.encode())
            print(f"Published update: {len(ids)} features at {ts_us}μs at topic '{self.upd_topic}'")
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
    """Simulate publishing a message after a delay"""
    def publish_after_delay():
        time.sleep(delay)

        msg = TrackingCommand()
        msg.start_time_us = start_us
        msg.end_time_us = end_us

        for i in range(50):
            if i < len(points):
                msg.feature_ids[i] = 1000 + i
                msg.feature_x[i] = points[i][0]
                msg.feature_y[i] = points[i][1]
            else:
                msg.feature_ids[i] = -1
                msg.feature_x[i] = 0.0
                msg.feature_y[i] = 0.0

        print(f"\n Publishing test message:")
        print(f"   Features: {points}")
        print(f"   Time window: {start_us} - {end_us} μs")

        bridge.lc.publish("TRACKING_COMMAND", msg.encode())

    threading.Thread(target=publish_after_delay, daemon=True).start()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/exe/inference_online/feature_tracking_cear.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    # LCM overrides (optional – can also be set in config['common'])
    parser.add_argument('--lcm_cmd_topic', type=str, default=None)
    parser.add_argument('--lcm_upd_topic', type=str, default=None)
    # Window margin (us) to catch boundary frames
    parser.add_argument('--window_margin_us', type=int, default=1_000)  # 1 ms
    parser.add_argument('--test_mode', action='store_true', help='Run with simulated publisher')

    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_root = Path(__file__).parent.parent
    save_dir = project_root / 'output' / 'inference' / config['common']['exp_name']
    save_dir.mkdir(parents=True, exist_ok=True)

    config['runtime_info'] = {
        'message': ' '.join(sys.argv),
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

    data_module = DataModuleFactory.create(config['data'])
    data_module.prepare_data()

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

    # --- LCM integration ---
    lcm_cmd_topic = args.lcm_cmd_topic or config['common'].get('lcm_cmd_topic', 'TRACKING_COMMAND')
    lcm_upd_topic = args.lcm_upd_topic or config['common'].get('lcm_upd_topic', 'TRACKING_UPDATE')
    vis_thresh = float(config['common'].get('visibility_threshold', 0.0))
    lcm_bridge = LcmBridge(lcm_cmd_topic, lcm_upd_topic)
    print(f"🔧 LCM topics: cmd='{lcm_cmd_topic}', upd='{lcm_upd_topic}'")

    if args.test_mode:
        print("\n TEST MODE: Simulating README scenario")
        print("   Command 1: --manual 100 100 130 210 --start-us 1704749447959841 --end-us 1704749448959841")

        points1 = [[100.0, 100.0], [130.0, 210.0]]
        start_us1 = 1704749447959841
        end_us1 = 1704749448959841

        simulate_publisher_command(lcm_bridge, points1, start_us1, end_us1, delay=1.0)

    DEBUG_PRINTS = True  # flip to True for verbose prints
    print(f"Current time0: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        iteration = 0

        # Initialize dataset/sequence info once
        if args.test_mode:
            dataset = None
            sequence_name = "test_sequence"
        else:
            dataset = data_module.test_datasets[0] if (data_module and data_module.test_datasets) else None
            sequence_name = dataset.subsequence_name if dataset else "unknown_sequence"

        if (dataset is None):
            print("  Error: Dataset is None. Skipping real data processing.")

        while True:
            iteration += 1
            print(f"\n--- ITERATION {iteration}: {sequence_name} ---")

            # 1) Wait for LCM message
            print(f" Waiting for LCM message on topic '{lcm_cmd_topic}'...")
            msg = None
            wait_timeout = 30.0
            wait_start = time.time()

            while msg is None and (time.time() - wait_start) < wait_timeout:
                msg = lcm_bridge.get_pending_request()
                time.sleep(0.01)

            if msg is None:
                print("Timeout waiting for message")
                if args.test_mode:
                    print(" In test mode, this is expected after the first message. Exiting loop.")
                    break
                else:
                    # Go back to waiting for the next message
                    continue

            print(
                f"Received message for '{sequence_name}': "
                f"[{msg.start_time_us*1e-6} - {msg.end_time_us*1e-6}] s, "
            )
            print(f"Current time1: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if (len(msg.feature_ids) != len(msg.feature_x)) or (len(msg.feature_ids) != len(msg.feature_y)):
                raise ValueError("LCM TrackingCommand arrays must be the same length")

            # 2) Filter valid features
            valid_features = []
            seed_ids = []
            for i in range(len(msg.feature_ids)):
                if msg.feature_ids[i] >= 0:
                    valid_features.append([msg.feature_x[i], msg.feature_y[i]])
                    seed_ids.append(msg.feature_ids[i])

            if not valid_features:
                print("  No valid features in message")
                continue

            # 3) Build queries tensor
            seed_xy = torch.tensor(valid_features, dtype=torch.float32, device=device)
            seed_t = torch.zeros((seed_xy.shape[0], 1), dtype=torch.int64, device=device)
            original_queries = torch.cat([seed_t, seed_xy], dim=1)
            # if DEBUG_PRINTS:
            #     print("id: ", seed_ids)
            #     print("original queries:", original_queries)

            # Optional support points
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
            coords_all, vis_all, ts_all = [], [], []

            # Expand window by margin
            start_us = int(msg.start_time_us) - int(args.window_margin_us)
            end_us   = int(msg.end_time_us)   + int(args.window_margin_us)
            if DEBUG_PRINTS:
                print(f"Window expanded to: [{start_us * 1e-6} - {end_us * 1e-6}] s")
                print(f"🔄 Processing frames in window [{to_s(start_us):.6f} - {to_s(end_us):.6f}] seconds")

            # 4) MAIN PROCESSING
            # try:
            if dataset and not args.test_mode:
                # Real dataset branch
                print(" Processing real dataset...")
            sample_all_voxels = []
            sample_all_timestamps = []








            print(f"Current time2: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

            # Convert once
            start_s = start_us * 1e-6
            end_s   = end_us   * 1e-6

            sample_all_voxels = []
            sample_all_timestamps = []
            append_vox = sample_all_voxels.append
            append_ts  = sample_all_timestamps.append

            frames_collected = 0
            last_ts_s: Optional[float] = None  # track last appended timestamp (sec)

            # (Drop tqdm here to minimize overhead in the hot loop)
            for sample, start_idx in dataset:
                ts = sample.timestamps  # sorted 1D
                vox = sample.voxels

                # ---- Torch path (preferred) ----
                if torch.is_tensor(ts):
                    # 1D CPU tensor view, no detach/copy unless needed
                    ts_cpu = ts.cpu() if ts.device.type != "cpu" else ts
                    ts_cpu = ts_cpu.reshape(-1)

                    # Fast reject
                    ts0 = float(ts_cpu[0]); ts1 = float(ts_cpu[-1])
                    if ts1 < start_s:
                        continue
                    if ts0 > end_s:
                        break  # OK only if samples are globally time-ordered

                    # Window slice
                    i0 = int(torch.searchsorted(ts_cpu, torch.tensor(start_s, dtype=ts_cpu.dtype), right=False))
                    i1 = int(torch.searchsorted(ts_cpu, torch.tensor(end_s,   dtype=ts_cpu.dtype), right=True))

                    if last_ts_s is not None:
                        # Skip anything <= last appended timestamp to avoid duplicates/overlap
                        i_skip = int(torch.searchsorted(ts_cpu, torch.tensor(last_ts_s, dtype=ts_cpu.dtype), right=True))
                        if i_skip > i0:
                            i0 = i_skip

                    if i1 <= i0:
                        continue

                    append_vox(vox[i0:i1])
                    append_ts(ts_cpu[i0:i1])
                    frames_collected += (i1 - i0)
                    last_ts_s = float(ts_cpu[i1 - 1])  # update cursor

                # ---- NumPy / list fallback ----
                else:
                    ts_np = np.asarray(ts).reshape(-1)
                    ts0 = float(ts_np[0]); ts1 = float(ts_np[-1])

                    if ts1 < start_s:
                        continue
                    if ts0 > end_s:
                        break

                    i0 = int(np.searchsorted(ts_np, start_s, side='left'))
                    i1 = int(np.searchsorted(ts_np, end_s,   side='right'))

                    if last_ts_s is not None:
                        i_skip = int(np.searchsorted(ts_np, last_ts_s, side='right'))
                        if i_skip > i0:
                            i0 = i_skip

                    if i1 <= i0:
                        continue

                    append_vox(vox[i0:i1])
                    append_ts(torch.from_numpy(ts_np[i0:i1]))  # store as torch for easy cat
                    frames_collected += (i1 - i0)
                    last_ts_s = float(ts_np[i1 - 1])

            print(f"Collected {frames_collected} frames across {len(sample_all_voxels)} samples in the window.")

            # ---- Combine, then sort + de-dup (robust against any residual overlap or out-of-order samples) ----
            if sample_all_voxels:
                combined_voxels = torch.cat(sample_all_voxels, dim=0)
                combined_timestamps = torch.cat(sample_all_timestamps, dim=0)  # seconds (torch 1D)

                # Sort by timestamp (microsecond precision), then drop exact duplicates
                ts_us = (combined_timestamps * 1e6).round().to(torch.long)
                order = torch.argsort(ts_us)
                ts_us = ts_us.index_select(0, order)
                combined_voxels = combined_voxels.index_select(0, order)

                keep = torch.ones_like(ts_us, dtype=torch.bool)
                keep[1:] = ts_us[1:] != ts_us[:-1]  # drop duplicates, keep first occurrence

                ts_us = ts_us[keep]
                combined_voxels = combined_voxels[keep]
                combined_timestamps = ts_us.to(torch.float64) * 1e-6  # back to seconds

                # (Optional) assert strictly increasing
                # assert torch.all(ts_us[1:] > ts_us[:-1])








            print(f"Current time3: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

            # Combine after loop
            if sample_all_voxels:
                combined_voxels = torch.cat(sample_all_voxels, dim=0)
                combined_timestamps = torch.cat(sample_all_timestamps, dim=0)

                if DEBUG_PRINTS:
                    print("Combined timestamps:", [f"{ts:.6f}" for ts in combined_timestamps])
                    print(f"📦 Aggregated {combined_voxels.shape[0]} frames "
                          f"in range [{combined_timestamps[0]:.6f} - {combined_timestamps[-1]:.6f}] s")

                max_len = 8  # maximum frames per voxel chunk
                # print("original queries:", original_queries)

                num_frames = combined_voxels.shape[0]
                for chunk_start in range(0, num_frames, max_len):
                    chunk_end = min(chunk_start + max_len, num_frames)

                    voxel_chunk = combined_voxels[chunk_start:chunk_end]          # [T_chunk, ...]
                    voxel_chunk = voxel_chunk.to(device)
                    ts_chunk    = combined_timestamps[chunk_start:chunk_end]      # [T_chunk]

                    if DEBUG_PRINTS:
                        print(f"▶ Processing chunk {chunk_start//max_len + 1}: "
                              f"{voxel_chunk.shape[0]} frames "
                              f"[{ts_chunk[0]:.6f} .. {ts_chunk[-1]:.6f}] s")

                    print(f"Current time4: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    
                    # Normalize and send to tracker
                    voxel_chunk = normalize_voxels(voxel_chunk)
                    print(f"Current time5: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    with torch.no_grad():
                        results = tracker(
                            video=voxel_chunk[None],   # [1, T_chunk, ...]
                            queries=queries[None],     # [1, N_total, 3]
                            is_online=True,
                            iters=6
                        )
                    print(f"Current time6: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

                    # ---- Update queries for next chunk ----
                    cp = results['coords_predicted']   # possibly [1, T, N_total, 2] or [T, N_total, 2]
                    if cp.dim() == 4:
                        cp = cp[0]                     # -> [T, N_total, 2]
                    elif cp.dim() != 3:
                        raise RuntimeError(f"Unexpected coords_predicted shape: {tuple(cp.shape)}")

                    # If you used support queries earlier, exclude them from the update
                    seed_N = len(seed_ids) if 'seed_ids' in locals() else cp.shape[1]
                    last_xy = cp[-1, :seed_N, :].to(torch.float32)   # [N_seed, 2]

                    # Build new seed queries: [N_seed, 3] (t=0)
                    new_seed_queries = torch.cat(
                        [
                            torch.zeros((last_xy.shape[0], 1), dtype=torch.float32, device=last_xy.device),
                            last_xy
                        ],
                        dim=1
                    )  # [N_seed, 3]
                    queries = new_seed_queries
                    # print("Updated queries:", queries)


                ids_out = seed_ids
                xs_out = last_xy[:, 0].detach().cpu().numpy().tolist()
                ys_out = last_xy[:, 1].detach().cpu().numpy().tolist()
                num_features = 50
                out_len = len(ids_out)
                if out_len < num_features:
                  pad_count = num_features - out_len
                  ids_out += [0] * pad_count
                  xs_out  += [0.0] * pad_count
                  ys_out  += [0.0] * pad_count
                else:
                  ids_out = ids_out[:num_features]
                  xs_out  = xs_out[:num_features]
                  ys_out  = ys_out[:num_features]
                ts_us = combined_timestamps[-1].item()*1e6
                # print("ids_out:", ids_out)
                # print("xs_out:", xs_out)
                # print("ys_out:", ys_out)
                # print(f"Publishing update for time {to_s(ts_us):.6f} s")
                

                lcm_bridge.publish_update(ts_us, ids_out, xs_out, ys_out)
                
                continue

    finally:
        lcm_bridge.close()
        print(" Done.")




if __name__ == '__main__':
    main()
