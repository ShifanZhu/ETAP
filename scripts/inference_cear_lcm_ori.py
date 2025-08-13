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

def normalize_item(item):
    """Return (sample_with_attrs, start_idx) where sample has .voxels and .timestamps."""
    if item is None:
        return None, None

    start_idx = None

    # Unpack tuples
    if isinstance(item, tuple):
        if len(item) == 2:
            a, b = item

            # Pattern A: (sample, start_idx)
            if hasattr(a, "voxels") or (isinstance(a, dict) and "voxels" in a):
                sample, start_idx = a, b

            # Pattern B: (voxels, timestamps)
            elif torch.is_tensor(a) and (torch.is_tensor(b) or isinstance(b, (np.ndarray, list))):
                sample = types.SimpleNamespace(voxels=a, timestamps=b)

            # Pattern C: (meta, sample) or (start_idx, sample)
            elif hasattr(b, "voxels") or (isinstance(b, dict) and "voxels" in b):
                sample, start_idx = b, a

            # Pattern D: (dict_sample, anything)
            elif isinstance(a, dict):
                sample, start_idx = a, b

            else:
                return None, None

        elif len(item) == 1:
            return normalize_item(item[0])
        else:
            return None, None
    else:
        sample = item

    # Convert dict samples to attribute access
    if isinstance(sample, dict):
        vox = sample.get("voxels")
        ts  = sample.get("timestamps")
        sample = types.SimpleNamespace(voxels=vox, timestamps=ts)

    # Final sanity
    if not hasattr(sample, "voxels") or not hasattr(sample, "timestamps"):
        return None, None

    return sample, start_idx

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
                f"[{msg.start_time_us*1e-6} .. {msg.end_time_us*1e-6}] s, "
            )

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

            # if DEBUG_PRINTS:
            #     print(f" Processing {len(valid_features)} features: {valid_features}")

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
                print(f"Window expanded to: [{start_us * 1e-6} .. {end_us * 1e-6}] s")
            print(f"🔄 Processing frames in window [{to_s(start_us):.6f} .. {to_s(end_us):.6f}] seconds")

            # 4) MAIN PROCESSING
            # try:
            if dataset and not args.test_mode:
                # Real dataset branch
                print(" Processing real dataset...")
                for sample, start_idx in tqdm(dataset, desc=f'Processing {sequence_name}'):
                # for sample in dataset:
                    # print(f"Processing sample starting at index {start_idx}, sample: {sample}")

                    voxels = sample.voxels.to(device) # data
                    step = voxels.shape[0] // 2 # 4 = 8 // 2


                    # voxels = sample.voxels
                    # ts     = sample.timestamps
                    # if voxels is None or ts is None:
                    #     print(f"  ⚠ dataset[{start_idx}] missing voxels/timestamps — skipping")
                    #     continue

                    # # --- your existing processing from here ---
                    # voxels = voxels.to(device)
                    # step = max(1, voxels.shape[0] // 2) # 4

                    # ts_arr = torch.as_tensor(ts, dtype=torch.float32) if not torch.is_tensor(ts) else ts.to(torch.float32)
                    # if ts_arr.numel() < step:
                    #     if DEBUG_PRINTS:
                    #         print(f"  ⚠ dataset[{start_idx}] not enough timestamps ({ts_arr.numel()} < {step}) — skipping")
                    #     continue
                      
                      
                    ts_chunk_s = sample.timestamps[-step:]
                    ts_chunk_us = (ts_chunk_s * 1e6).round().long()
                    # print(f"Processing chunk with timestamps: {ts_chunk_us.tolist()}")

                    # Skip chunks outside window
                    if (ts_chunk_us[-1] < start_us) or (ts_chunk_us[0] > end_us):
                        continue

                    if DEBUG_PRINTS:
                        print("sample timestamps (s):", [f"{v:.6f}" for v in sample.timestamps.tolist()])
                        print(f"  Processing chunk with timestamps (s): {[ts_chunk_s.tolist()]}")
                        # print(f"  Processing chunk with timestamps (s): {[to_s(ts) for ts in ts_chunk_us.tolist()]}")

                    voxels = normalize_voxels(voxels)
                    # print("queries before:", queries)

                    with torch.no_grad():
                        results = tracker(
                            video=voxels[None],
                            queries=queries[None],
                            is_online=True,
                            iters=6
                        )
                    coords_predicted = results['coords_predicted']
                    vis_logits = results['vis_predicted']
                    # Update queries with predicted results for next chunk
                    # build [N,3] queries (time column set to 0)
                    cp = coords_predicted
                    if cp.dim() == 4:
                        cp = cp[0]            # -> [T, N, 2]
                    elif cp.dim() != 3:
                        raise RuntimeError(f"Unexpected coords_predicted shape: {tuple(cp.shape)}")

                    # (optional) drop support queries; keep only seed queries
                    seed_N = len(seed_ids) if 'seed_ids' in locals() else cp.shape[1]
                    last_xy = cp[-1, :seed_N, :]         # [N, 2]
                    queries = torch.cat([
                        torch.zeros((last_xy.shape[0], 1), dtype=torch.float32, device=last_xy.device),  # [N,1]
                        last_xy                                                                            # [N,2]
                    ], dim=1)  # -> [N,3]
                    # print("queries after:", queries)
                    print("id: ", seed_ids)
                    print("predicted:", last_xy)
                    
                    ids_out = seed_ids
                    xs_out = last_xy[:, 0].detach().cpu().numpy().tolist()
                    ys_out = last_xy[:, 1].detach().cpu().numpy().tolist()
                    ts_us = int(ts_chunk_us[-1].item())
                    print(f"Publishing update for time {to_s(ts_us):.6f} s")
                    

                    lcm_bridge.publish_update(ts_us, ids_out, xs_out, ys_out)
                    break
                

                    # Remove support queries if used
                    if support_num_queries > 0:
                        coords_predicted = coords_predicted[:, :, :-support_num_queries]
                        vis_logits       = vis_logits[:, :, :-support_num_queries]

                    coords_predicted = coords_predicted[0]  # [T, N, 2]
                    vis_logits       = vis_logits[0]        # [T, N]

                    Tcur = coords_predicted.shape[0]
                    min_len = min(Tcur, ts_chunk_s.shape[0], ts_chunk_us.shape[0])
                    print("min_len:", min_len)
                    if min_len <= 0:
                        continue

                    # if DEBUG_PRINTS:
                    #     print("coords_predicted.shape:", coords_predicted.shape)

                    coords_predicted = coords_predicted[:min_len]
                    vis_logits       = vis_logits[:min_len]
                    ts_chunk_s       = ts_chunk_s[:min_len]
                    ts_chunk_us      = ts_chunk_us[:min_len]

                    # Keep frames inside the window for saving
                    frame_mask = (ts_chunk_us >= start_us) & (ts_chunk_us <= end_us)
                    if frame_mask.any():
                        keep_idx   = torch.nonzero(frame_mask, as_tuple=False).squeeze(-1)
                        coords_all.append(coords_predicted[keep_idx].cpu())
                        vis_all.append(vis_logits[keep_idx].cpu())
                        ts_all.append(ts_chunk_s[keep_idx].cpu())
                        processed_any = True

                    # Stream results via LCM
                    for ti in range(min_len):
                        ts_us = int(ts_chunk_us[ti].item())
                        if ts_us < start_us or ts_us > end_us:
                            continue

                        xy  = coords_predicted[ti].detach().cpu().numpy()
                        vis = vis_logits[ti].detach().cpu().numpy()

                        if vis_thresh > 0.0:
                            mask = vis >= vis_thresh
                            if not mask.any():
                                continue
                            ids_out = [all_ids[i] for i, m in enumerate(mask) if m]
                            xs_out  = [float(xy[i, 0]) for i, m in enumerate(mask) if m]
                            ys_out  = [float(xy[i, 1]) for i, m in enumerate(mask) if m]
                        else:
                            ids_out = all_ids
                            xs_out  = xy[:, 0].tolist()
                            ys_out  = xy[:, 1].tolist()

                        # Pad/clip to fixed length
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

                        if DEBUG_PRINTS:
                            # print(f"  Frame {ti}: ts={ts_us}, ids={ids_out}, xs={xs_out}, ys={ys_out}")
                            # print("ids_out:", ids_out)
                            # print("xs_out:", xs_out)
                            # print("ys_out:", ys_out)
                            print(f"id: {ids_out[0]}, x: {xs_out[0]:.4f}, y: {ys_out[0]:.4f}, at time: {to_s(ts_us):.6f}")

                        # Publish update
                        print(f"Publishing update for frame at {to_s(ts_us):.6f} s")
                        # lcm_bridge.publish_update(ts_us, ids_out, xs_out, ys_out)

            else:
                # Synthetic fallback / demo branch
                print(" Simulating processing with synthetic data...")

                duration_s = max(0.0, (end_us - start_us) / 1e6)
                num_frames = max(1, int(duration_s * 10))  # ~10 FPS simulation

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

                    frame_time_us = int(start_us + frame_idx * (end_us - start_us) / max(1, num_frames))

                    xs_out = coords_predicted[:, 0].tolist()
                    ys_out = coords_predicted[:, 1].tolist()

                    num_features = 50
                    ids_out = (all_ids + [0] * num_features)[:num_features]
                    xs_out  = (xs_out  + [0.0] * num_features)[:num_features]
                    ys_out  = (ys_out  + [0.0] * num_features)[:num_features]

                    lcm_bridge.publish_update(frame_time_us, ids_out, xs_out, ys_out)
                    time.sleep(0.1)

                processed_any = True

            # except Exception as e:
            #     # If the real-data branch throws, we fall back to synthetic here for robustness
                # print(f"  Real data processing failed: {e}")
            #     print(" Falling back to synthetic processing for demonstration...")

            #     duration_s = max(0.0, (end_us - start_us) / 1e6)
            #     num_frames = max(1, int(duration_s * 10))

            #     for frame_idx in range(num_frames):
            #         voxels = torch.randn(8, 10, 240, 320, device=device)
            #         voxels = normalize_voxels(voxels)

            #         with torch.no_grad():
            #             results = tracker(
            #                 video=voxels[None],
            #                 queries=queries[None],
            #                 is_online=True,
            #                 iters=6
            #             )

            #         coords_predicted = results['coords_predicted'][0][-1].cpu().numpy()
            #         frame_time_us = int(start_us + frame_idx * (end_us - start_us) / max(1, num_frames))

            #         xs_out = coords_predicted[:, 0].tolist()
            #         ys_out = coords_predicted[:, 1].tolist()

            #         num_features = 50
            #         ids_out = (all_ids + [0] * num_features)[:num_features]
            #         xs_out  = (xs_out  + [0.0] * num_features)[:num_features]
            #         ys_out  = (ys_out  + [0.0] * num_features)[:num_features]

            #         lcm_bridge.publish_update(frame_time_us, ids_out, xs_out, ys_out)
            #         time.sleep(0.1)

                processed_any = True

            # 5) Post-processing / save
            if processed_any:
                print(f" Processing complete for {sequence_name}")

                if coords_all and not args.test_mode:
                    coords_out = torch.cat(coords_all, dim=0)
                    vis_out    = torch.cat(vis_all,   dim=0)
                    ts_out     = torch.cat(ts_all,    dim=0)

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
                    print(" Test mode: Skipping file save")
            else:
                print(f"  No frames processed for {sequence_name}")

            print(" Returning to waiting state...")

        # end while

    except KeyboardInterrupt:
        print("\n Shutting down...")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        lcm_bridge.close()
        print(" Done.")




if __name__ == '__main__':
    main()
