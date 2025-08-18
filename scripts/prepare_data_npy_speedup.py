#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Your repo imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.modules import DataModuleFactory

# -----------------------------
# Utils
# -----------------------------
def get_git_commit_hash():
    try:
        import subprocess
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.STDOUT
        ).decode().strip()
        return commit_hash
    except Exception:
        return "unknown"

def save_cache_npy(cache_dir: Path, name: str, start_us: int, end_us: int,
                   margin_us: int, vox: torch.Tensor, ts_s: torch.Tensor,
                   write_meta_json: bool = True):
    """
    Save window as:
      <start_us>_vox.npy      # [T, C, H, W], dtype from vox
      <start_us>_ts.npy       # [T], int64 microseconds
      <start_us>_meta.json    # small metadata (optional)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    vox_path = cache_dir / f"{start_us}_vox.npy"
    ts_path  = cache_dir / f"{start_us}_ts.npy"
    meta_path = cache_dir / f"{start_us}_meta.json"

    vox_np = vox.contiguous().cpu().numpy()  # [T,C,H,W]
    ts_us_np = (ts_s * 1e6).round().to(torch.long).cpu().numpy()  # [T] int64 µs

    np.save(vox_path, vox_np)
    np.save(ts_path, ts_us_np)

    if write_meta_json:
        meta = {
            "sequence": str(name),
            "start_us": int(start_us),
            "end_us": int(end_us),
            "margin_us": int(margin_us),
            "frames": int(vox_np.shape[0]),
            "shape": list(vox_np.shape),
            "dtype": str(vox_np.dtype),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "git": get_git_commit_hash(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    print(f"💾 Saved:")
    print(f"   vox -> {vox_path}  (shape={vox_np.shape}, dtype={vox_np.dtype})")
    print(f"   ts  -> {ts_path}   (len={ts_us_np.shape[0]})")
    if write_meta_json:
        print(f"   meta-> {meta_path}")

# -----------------------------
# Streaming exporter
# -----------------------------
def stream_export_half_open(dataset,
                            gt_times_us: np.ndarray,
                            out_dir: Path,
                            sequence_name: Optional[str] = None,
                            debug: bool = False,
                            write_meta_json: bool = True):
    """
    Export consecutive GT windows [t_i, t_{i+1}) in a single pass over `dataset`.
    - Skips missing sample files.
    - Skips intervals already on disk (<start>_vox.npy & <start>_ts.npy).
    - Deduplicates at microsecond precision on flush (sort+drop duplicate timestamps).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(np.unique(gt_times_us), dtype=np.int64)
    if gt.size < 2:
        if debug:
            print("Not enough GT timestamps to export.")
        return

    # State for current interval and buffers
    k = 0
    start_us, end_us = int(gt[k]), int(gt[k+1])
    vox_buf, ts_buf = [], []

    def _maybe_flush():
        """Flush current interval buffers to disk and advance interval pointer."""
        nonlocal vox_buf, ts_buf, start_us, end_us, k

        if not vox_buf:
            return

        vox_cat = torch.cat(vox_buf, dim=0)            # [T, C, H, W]
        ts_cat  = torch.cat(ts_buf,  dim=0)            # [T] int64 µs

        # Sort+dedup by µs to be robust across sample overlaps
        order = torch.argsort(ts_cat)
        ts_sorted = ts_cat.index_select(0, order)
        vox_sorted = vox_cat.index_select(0, order)
        keep = torch.ones_like(ts_sorted, dtype=torch.bool)
        keep[1:] = ts_sorted[1:] != ts_sorted[:-1]
        ts_sorted = ts_sorted[keep]
        vox_sorted = vox_sorted[keep]

        # Convert timestamps to seconds (torch float64) for save helper
        ts_sec = ts_sorted.to(torch.float64) * 1e-6

        vox_file = out_dir / f"{start_us}_vox.npy"
        ts_file  = out_dir / f"{start_us}_ts.npy"
        if vox_file.exists() and ts_file.exists():
            if debug:
                print(f"↪︎ Interval [{start_us/1e6:.6f}, {end_us/1e6:.6f}) exists; skipping save.")
        else:
            save_cache_npy(
                cache_dir=out_dir,
                name=(sequence_name or str(start_us)),
                start_us=start_us,
                end_us=end_us,
                margin_us=0,
                vox=vox_sorted,
                ts_s=ts_sec,
                write_meta_json=write_meta_json,
            )

        vox_buf.clear()
        ts_buf.clear()

    # Iterate dataset exactly once
    for idx in range(len(dataset)):
        try:
            sample, _ = dataset[idx]
        except FileNotFoundError as e:
            if debug:
                print(f"⚠️ Missing file at idx {idx}: {getattr(e, 'filename', e)}")
            continue
        except Exception as e:
            if debug:
                print(f"⚠️ Skipping idx {idx} due to error: {e}")
            continue

        ts = sample.timestamps  # [T_sample] seconds (sorted)
        vox = sample.voxels     # [T_sample, C, H, W]

        # Convert timestamps to int64 µs
        if torch.is_tensor(ts):
            ts_np = ts.detach().cpu().reshape(-1).numpy()
        else:
            ts_np = np.asarray(ts).reshape(-1)
        ts_us_np = np.rint(ts_np * 1e6).astype(np.int64)

        # If this whole sample is before the current interval, skip
        if ts_us_np[-1] <= start_us:
            continue

        # Jump forward in GT intervals until we overlap this sample
        while ts_us_np[0] >= end_us:
            _maybe_flush()
            k += 1
            if k >= gt.size - 1:
                if debug: print("✅ Finished all intervals.")
                return
            start_us, end_us = int(gt[k]), int(gt[k+1])

        # Consume as many intervals as this sample covers
        while True:
            # Slice overlap of this sample with [start_us, end_us)
            i0 = int(np.searchsorted(ts_us_np, start_us, side="left"))
            i1 = int(np.searchsorted(ts_us_np, end_us,   side="left"))  # end exclusive
            if i1 > i0:
                if torch.is_tensor(vox):
                    vox_slice = vox[i0:i1]
                else:
                    vox_slice = torch.from_numpy(vox[i0:i1])
                ts_slice = torch.from_numpy(ts_us_np[i0:i1])  # int64 µs
                vox_buf.append(vox_slice)
                ts_buf.append(ts_slice)

            # If sample ends before interval ends, move to next dataset sample
            if ts_us_np[-1] < end_us:
                break

            # Otherwise we reached/passed the end of the interval: flush and advance
            _maybe_flush()
            k += 1
            if k >= gt.size - 1:
                if debug: print("✅ Finished all intervals.")
                return
            start_us, end_us = int(gt[k]), int(gt[k+1])
            # Loop again: the same sample may contain frames for multiple intervals

    # End of dataset: flush whatever remains for the last partial interval
    _maybe_flush()
    if debug:
        print("✅ Export complete (end of dataset).")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to your YAML config (same you use for DataModuleFactory)')
    parser.add_argument('--debug', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build datamodule & dataset (test split)
    data_module = DataModuleFactory.create(config['data'])
    data_module.prepare_data()
    dataset = data_module.test_datasets[0] if data_module.test_datasets else None
    if dataset is None:
        print("✗ No test dataset found from DataModuleFactory.")
        return

    sequence_name = getattr(dataset, "subsequence_name", "unknown_sequence")
    if args.debug:
        print(f"🎞  Using sequence: {sequence_name}")

    # Derive output directory from your config pattern
    data_root = Path(config['data']['data_root'])
    dataset_name = config['data']['dataset_name']
    preprocessed_name = config['data']['preprocessed_name']
    # Adjust preprocessed_name like your previous code did
    if isinstance(preprocessed_name, (list, tuple)):
        preprocessed_name = list(preprocessed_name[:-1]) + [2]
    elif isinstance(preprocessed_name, str):
        parts = preprocessed_name.split('_')
        if len(parts) > 1:
            preprocessed_name = '_'.join(parts[:-1] + ['2'])
        else:
            preprocessed_name = '2'

    out_dir = data_root / dataset_name / sequence_name / 'events' / str(preprocessed_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        print(f"📂 Output dir: {out_dir}")

    # Load GT times for this sequence (seconds -> µs)
    gt_path = Path('config/misc/cear/gt_tracks') / f'{sequence_name}.gt.txt'
    if not gt_path.exists():
        print(f"✗ GT file not found: {gt_path}")
        return
    gt_tracks = np.genfromtxt(str(gt_path))  # [id, t, x, y] with t in seconds
    gt_times_us = np.rint(np.unique(gt_tracks[:, 1].astype(np.float64)) * 1e6).astype(np.int64)

    if args.debug:
        print(f"GT intervals: {len(gt_times_us)-1} windows")
        if len(gt_times_us) > 5:
            print("First few GT µs:", gt_times_us[:5].tolist())

    # Stream export once and exit
    stream_export_half_open(
        dataset=dataset,
        gt_times_us=gt_times_us,
        out_dir=out_dir,
        sequence_name=sequence_name,
        debug=args.debug,
        write_meta_json=True,
    )

if __name__ == '__main__':
    main()
