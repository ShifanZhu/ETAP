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
import re
from pathlib import Path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import SUPPORTED_SEQUENCES_FEATURE_TRACKING
import os, json, re
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed



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

def _parse_range_from_name(name: str, seq: str):
    """
    Try to parse [start_us, end_us] from filename. Supports either:
      <seq>_<start>_<end>_m*.pt
      <seq>_t<start>_to_t<end>.pt
    Returns (start_us, end_us) or None.
    """
    pat1 = re.compile(rf"^{re.escape(seq)}_(\d+)_(\d+)_m\d+\.pt$")
    m = pat1.match(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    pat2 = re.compile(rf"^{re.escape(seq)}_t(\d+)_to_t(\d+)\.pt$")
    m = pat2.match(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def load_cached_window_ori(cache_root: Path, sequence_name: str, start_us: int, end_us: int):
    """
    Load precomputed caches that overlap [start_us, end_us], merge them,
    then slice strictly to the requested window. Returns (voxels, timestamps_sec)
    or (None, None) if nothing found.
    """
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return None, None

    candidates = []
    for p in cache_root.glob("*.pt"):
        rng = _parse_range_from_name(p.name, sequence_name)
        if rng is None:
            # Fallback: try reading meta (slower, but robust)
            try:
                blob = torch.load(p, map_location="cpu")
                meta = blob.get("meta", {})
                s, e = int(meta.get("start_us", -1)), int(meta.get("end_us", -1))
                if s >= 0 and e >= 0:
                    rng = (s, e)
                else:
                    continue
            except Exception:
                continue
        s, e = rng
        # keep only files that overlap our requested window
        if e <= start_us or s >= end_us:
            continue
        candidates.append((s, e, p))

    if not candidates:
        return None, None

    # Sort by start time to make merging deterministic
    candidates.sort(key=lambda x: x[0])

    vox_list, ts_us_list = [], []
    for s, e, p in candidates:
        blob = torch.load(p, map_location="cpu")
        vox = blob["voxels"]                      # [T, C, H, W]
        ts_us = blob.get("timestamps_us", None)   # [T], int64 microseconds
        if ts_us is None:
            # Back-compat: if saved as seconds
            ts_s = blob.get("timestamps", None)
            if ts_s is None:
                continue
            ts_us = (ts_s * 1e6).round().to(torch.long)

        # Slice to requested window (inclusive on both ends to match your search)
        mask = (ts_us >= start_us) & (ts_us <= end_us)
        if not mask.any():
            continue

        vox_list.append(vox[mask])
        ts_us_list.append(ts_us[mask])

    if not vox_list:
        return None, None

    vox = torch.cat(vox_list, dim=0)
    ts_us = torch.cat(ts_us_list, dim=0)

    # Sort + de-dup by microsecond (protect against boundary overlaps)
    order = torch.argsort(ts_us)
    ts_us = ts_us.index_select(0, order)
    vox   = vox.index_select(0, order)
    keep = torch.ones_like(ts_us, dtype=torch.bool)
    keep[1:] = ts_us[1:] != ts_us[:-1]
    ts_us = ts_us[keep]
    vox   = vox[keep]

    ts_s = ts_us.to(torch.float64) * 1e-6
    return vox, ts_s


# Patterns we might have used when saving
def _parse_range_from_name_any(name: str):
    # pattern A: <seq>_<start>_<end>_m<margin>.pt
    m = re.search(r'_(\d+)_(\d+)_m\d+\.pt$', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # pattern B: <seq>_t<start>_to_t<end>.pt
    m = re.search(r'_t(\d+)_to_t(\d+)\.pt$', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None



import re, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parse either "<seq>_<start>_<end>_m*.pt"  OR  "<seq>_t<start>_to_t<end>.pt"
_PAT_A = re.compile(r'_(\d+)_(\d+)_m\d+\.pt$')
_PAT_B = re.compile(r'_t(\d+)_to_t(\d+)\.pt$')

def _parse_range_from_name_fast(name: str):
    m = _PAT_A.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _PAT_B.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def load_cached_window_fast(cache_root: Path, sequence_name: str, start_us: int, end_us: int):
    """
    Faster loader:
      - Recursively scans *.pt under cache_root (handles subfolders)
      - Filters by filename-encoded ranges first (no torch.load)
      - Falls back to meta read only when needed
      - Loads overlapping files in parallel and slices via searchsorted
    Returns (voxels[T,C,H,W], timestamps_sec[T]) or (None, None).
    """
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return None, None
      
    # print("Debug cache scan:")
    # for p in (Path(cache_root) / sequence_name if (Path(cache_root)/sequence_name).exists() else Path(cache_root)).rglob("*.pt"):
    #     print("  ", p.name)


    # Prefer searching in "<root>/<sequence_name>" if it exists; else whole root.
    search_roots = []
    seq_dir = cache_root / sequence_name
    if seq_dir.exists():
        search_roots.append(seq_dir)
    search_roots.append(cache_root)

    # 1) Gather overlapping candidates by filename (FAST)
    candidates = []           # (start_us_file, end_us_file, Path)
    unknown_meta_files = []   # paths where we couldn't parse range from name

    for root in search_roots:
        for p in root.rglob("*.pt"):
            rng = _parse_range_from_name_fast(p.name)
            if rng is not None:
                s, e = rng
                # keep only files that overlap [start_us, end_us]
                if not (e <= start_us or s >= end_us):
                    # If sequence_name is encoded in path/name, prefer those;
                    # else accept (still correct, just broader).
                    if sequence_name in p.name or sequence_name in p.parts:
                        candidates.append((s, e, p))
                    else:
                        # keep as lower priority; add later only if needed
                        unknown_meta_files.append(p)
            else:
                unknown_meta_files.append(p)

    # If we found nothing with the name filter, try meta for a few (avoid O(N) loads)
    if not candidates and unknown_meta_files:
        # Meta probe limit keeps things from getting slow; raise if needed
        PROBE_LIMIT = 50
        for p in unknown_meta_files[:PROBE_LIMIT]:
            try:
                blob = torch.load(p, map_location="cpu")
                meta = blob.get("meta", {})
                s = int(meta.get("start_us", -1))
                e = int(meta.get("end_us", -1))
                seq_meta = str(meta.get("sequence", "")) if "sequence" in meta else ""
                if s >= 0 and e >= 0 and not (e <= start_us or s >= end_us):
                    if not sequence_name or not seq_meta or seq_meta == sequence_name:
                        candidates.append((s, e, p))
            except Exception:
                pass  # ignore unreadable files

    if not candidates:
        return None, None

    # Sort by start time for deterministic merging
    candidates.sort(key=lambda x: x[0])

    # 2) Load overlapping slices in parallel (I/O bound -> threads help)
    def _load_slice(path: Path):
        try:
            blob = torch.load(path, map_location="cpu")
            ts_us = blob.get("timestamps_us", None)
            if ts_us is None:
                ts_s = blob.get("timestamps", None)
                if ts_s is None:
                    return None
                ts_us = (ts_s * 1e6).round().to(torch.long)

            # Compute slice via searchsorted (inclusive end, to match your original mask)
            start_t = torch.tensor(start_us, dtype=ts_us.dtype)
            end_t   = torch.tensor(end_us,   dtype=ts_us.dtype)
            i0 = int(torch.searchsorted(ts_us, start_t, right=False))
            i1 = int(torch.searchsorted(ts_us, end_t,   right=True))
            if i1 <= i0:
                return None

            vox = blob["voxels"]
            return vox[i0:i1], ts_us[i0:i1]
        except Exception as e:
            print(f"✗ Failed to load {path}: {e}")
            return None

    max_workers = min(8, (os.cpu_count() or 4))
    vox_chunks, ts_chunks = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_slice, p) for _, _, p in candidates]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            v, t = res
            vox_chunks.append(v)
            ts_chunks.append(t)

    if not vox_chunks:
        return None, None

    # 3) Merge, sort, de-dup
    vox = torch.cat(vox_chunks, dim=0)
    ts_us = torch.cat(ts_chunks, dim=0)
    order = torch.argsort(ts_us)
    ts_us = ts_us.index_select(0, order)
    vox   = vox.index_select(0, order)
    keep = torch.ones_like(ts_us, dtype=torch.bool)
    keep[1:] = ts_us[1:] != ts_us[:-1]
    ts_us = ts_us[keep]
    vox   = vox[keep]

    ts_s = ts_us.to(torch.float64) * 1e-6
    return vox, ts_s

def load_cached_window_fast_npy(cache_root: Path, sequence_name: str, start_us: int, end_us: int):
    """
    Faster loader for .npy caches:
      - Recursively scans *_ts.npy under cache_root (handles subfolders)
      - Uses mmap to peek timestamps and find overlap via searchsorted
      - Loads only the overlapping voxel slices (copy) in parallel
    Returns (voxels[T,C,H,W], timestamps_sec[T]) or (None, None).
    """
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return None, None

    # Prefer "<root>/<sequence_name>" if present; else whole root
    search_roots = []
    seq_dir = cache_root / sequence_name
    if seq_dir.exists():
        search_roots.append(seq_dir)
    search_roots.append(cache_root)

    # Gather candidate (vox_path, ts_path, i0, i1)
    candidates = []
    for root in search_roots:
        for ts_path in root.rglob("*_ts.npy"):
            vox_path = ts_path.with_name(ts_path.name.replace("_ts.npy", "_vox.npy"))
            if not vox_path.exists():
                continue

            # mmap timestamps; cheap to read first/last and do searchsorted
            ts = np.load(ts_path, mmap_mode="r")
            if ts.size == 0:
                continue

            # quick reject (assumes ts sorted ascending)
            if ts[-1] < start_us or ts[0] > end_us:
                continue

            i0 = int(np.searchsorted(ts, start_us, side="left"))
            i1 = int(np.searchsorted(ts, end_us,   side="right"))
            if i1 <= i0:
                continue

            candidates.append((vox_path, ts_path, i0, i1))

    if not candidates:
        return None, None

    # Load overlapping slices in parallel.
    # We copy slices (small) to avoid memmap lifetime issues across threads.
    def _load_slice(vox_path: Path, ts_path: Path, i0: int, i1: int):
        try:
            ts_mm = np.load(ts_path, mmap_mode="r")
            ts_slice = np.array(ts_mm[i0:i1], dtype=np.int64, copy=True)  # [T], int64

            vox_mm = np.load(vox_path, mmap_mode="r")                     # [T,C,H,W]
            vox_slice = np.array(vox_mm[i0:i1], copy=True)                # keep original dtype

            v = torch.from_numpy(vox_slice)                               # CPU tensor
            t = torch.from_numpy(ts_slice)                                # int64 µs
            return v, t
        except Exception as e:
            print(f"✗ Failed to load {vox_path.name.replace('_vox.npy','')}*: {e}")
            return None

    max_workers = min(8, (os.cpu_count() or 4))
    vox_chunks, ts_chunks = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_slice, vp, tp, i0, i1) for (vp, tp, i0, i1) in candidates]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            v, t = res
            vox_chunks.append(v)
            ts_chunks.append(t)

    if not vox_chunks:
        return None, None

    # Merge, sort, de-dup (in case multiple files overlapped)
    vox = torch.cat(vox_chunks, dim=0)
    ts_us = torch.cat(ts_chunks, dim=0)

    order = torch.argsort(ts_us)
    ts_us = ts_us.index_select(0, order)
    vox   = vox.index_select(0, order)

    keep = torch.ones_like(ts_us, dtype=torch.bool)
    keep[1:] = ts_us[1:] != ts_us[:-1]
    ts_us = ts_us[keep]
    vox   = vox[keep]

    ts_s = ts_us.to(torch.float64) * 1e-6
    return vox, ts_s



# Precompiled patterns (fast)
_RANGE1 = None
_RANGE2 = None
def _parse_range_from_name(name: str, seq: str):
    global _RANGE1, _RANGE2
    if _RANGE1 is None:
        _RANGE1 = re.compile(rf"^{re.escape(seq)}_(\d+)_(\d+)_m\d+\.pt$")
        _RANGE2 = re.compile(rf"^{re.escape(seq)}_t(\d+)_to_t(\d+)\.pt$")
    m = _RANGE1.match(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _RANGE2.match(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def _index_path(cache_root: Path, sequence_name: str) -> Path:
    return Path(cache_root) / f"{sequence_name}.index.json"

def _build_index(cache_root: Path, sequence_name: str):
    """Scan cache_root for *.pt belonging to this sequence and build a sorted index."""
    cache_root = Path(cache_root)
    entries = []
    for p in cache_root.glob(f"{sequence_name}*.pt"):
        rng = _parse_range_from_name(p.name, sequence_name)
        if rng:
            s, e = rng
            entries.append({"start_us": s, "end_us": e, "file": str(p)})
    entries.sort(key=lambda d: d["start_us"])
    idx = {
        "sequence": sequence_name,
        "root": str(cache_root),
        "files": entries,  # sorted ascending by start_us
    }
    # Save
    ip = _index_path(cache_root, sequence_name)
    ip.parent.mkdir(parents=True, exist_ok=True)
    with open(ip, "w") as f:
        json.dump(idx, f)
    return idx

def _load_index(cache_root: Path, sequence_name: str):
    """Load index if present and still matches folder contents; otherwise rebuild."""
    ip = _index_path(cache_root, sequence_name)
    cache_root = Path(cache_root)
    if ip.exists():
        try:
            with open(ip, "r") as f:
                idx = json.load(f)
            # Quick freshness check: compare filenames set
            on_disk = sorted([str(p) for p in cache_root.glob(f"{sequence_name}*.pt")])
            in_idx  = sorted([e["file"] for e in idx.get("files", [])])
            if on_disk == in_idx:
                return idx
        except Exception:
            pass
    # Rebuild if missing or stale
    return _build_index(cache_root, sequence_name)


from collections import OrderedDict

class NpyWindowBuffer:
    """
    In-memory LRU buffer for .npy time-chunks.
    Expects pairs named "<prefix>_ts.npy" (int64 µs, sorted) and "<prefix>_vox.npy" ([T,C,H,W]).
    """

    def __init__(self, cache_root: Path, sequence_name: str, max_ram_bytes: int = 2 * (1 << 30)):
        self.root = Path(cache_root)
        self.sequence = sequence_name
        self.max_ram_bytes = int(max_ram_bytes)

        self.index = self._build_index()   # sorted entries with (start_us, end_us, paths)
        self.starts = [e["start_us"] for e in self.index]
        self.ends   = [e["end_us"]   for e in self.index]

        self._chunks = OrderedDict()       # key -> {"vox","ts_us","bytes","start","end"}
        self._total_bytes = 0

    # ---------- indexing ----------
    def _build_index(self):
        idx = []
        roots = []
        seq_dir = self.root / self.sequence
        if seq_dir.exists():
            roots.append(seq_dir)
        roots.append(self.root)

        seen = set()
        for r in roots:
            for ts_path in r.rglob("*_ts.npy"):
                vox_path = ts_path.with_name(ts_path.name.replace("_ts.npy", "_vox.npy"))
                if not vox_path.exists():
                    continue
                key = str(vox_path.resolve())
                if key in seen:
                    continue
                seen.add(key)

                ts_mm = np.load(ts_path, mmap_mode="r")
                if ts_mm.size == 0:
                    continue
                s = int(ts_mm[0])
                e = int(ts_mm[-1])
                idx.append({"start_us": s, "end_us": e,
                            "vox_path": str(vox_path), "ts_path": str(ts_path)})
        print(f"Found {len(idx)} npy file pairs for sequence '{self.sequence}' in '{self.root}'")
        idx.sort(key=lambda d: d["start_us"])
        return idx

    def _overlapping_entries(self, start_us: int, end_us: int):
        # first idx with end >= start_us
        j = bisect_left(self.ends, start_us)
        out = []
        for k in range(j, len(self.index)):
            e = self.index[k]
            if e["start_us"] > end_us:
                break
            # overlap if not (end <= start or start >= end)
            if not (e["end_us"] <= start_us or e["start_us"] >= end_us):
                out.append(e)
        return out

    # ---------- LRU management ----------
    def _touch(self, key: str):
        if key in self._chunks:
            self._chunks.move_to_end(key, last=True)

    def _evict_if_needed(self):
        while self._total_bytes > self.max_ram_bytes and self._chunks:
            k, item = self._chunks.popitem(last=False)  # LRU
            self._total_bytes -= item["bytes"]

    def _load_chunk(self, entry):
        key = entry["vox_path"]
        if key in self._chunks:
            self._touch(key)
            return

        # Fully load (no memmap) so subsequent requests are RAM-only
        vox_np = np.load(entry["vox_path"])      # [T,C,H,W]
        ts_np  = np.load(entry["ts_path"])       # [T] int64 µs

        vox = torch.from_numpy(vox_np)           # CPU tensors
        ts_us = torch.from_numpy(ts_np)

        bytes_est = vox.element_size() * vox.numel() + ts_us.element_size() * ts_us.numel()
        self._chunks[key] = {"vox": vox, "ts_us": ts_us,
                             "bytes": int(bytes_est),
                             "start": entry["start_us"], "end": entry["end_us"]}
        self._total_bytes += bytes_est
        self._touch(key)
        self._evict_if_needed()

    # ---------- public API ----------
    def get_window(self, start_us: int, end_us: int):
        """
        Returns (voxels[T,C,H,W], timestamps_sec[T]) for [start_us, end_us] (inclusive end),
        or (None, None) if nothing overlaps.
        """
        entries = self._overlapping_entries(start_us, end_us)
        if not entries:
            return None, None

        # Ensure required chunks resident
        for e in entries:
            self._load_chunk(e)

        vox_chunks, ts_chunks = [], []
        start_t = torch.tensor(start_us, dtype=torch.long)
        end_t   = torch.tensor(end_us,   dtype=torch.long)

        for e in entries:
            key = e["vox_path"]
            item = self._chunks[key]
            ts_us = item["ts_us"]
            # inclusive end to match your previous logic
            i0 = int(torch.searchsorted(ts_us, start_t, right=False))
            i1 = int(torch.searchsorted(ts_us, end_t,   right=True))
            if i1 <= i0:
                continue
            vox_chunks.append(item["vox"][i0:i1])
            ts_chunks.append(ts_us[i0:i1])

        if not vox_chunks:
            return None, None

        vox = torch.cat(vox_chunks, dim=0)
        ts_us = torch.cat(ts_chunks, dim=0)

        # sort + dedup (safety if multiple chunks overlap on boundaries)
        order = torch.argsort(ts_us)
        ts_us = ts_us.index_select(0, order)
        vox   = vox.index_select(0, order)
        keep = torch.ones_like(ts_us, dtype=torch.bool)
        keep[1:] = ts_us[1:] != ts_us[:-1]
        ts_us = ts_us[keep]
        vox   = vox[keep]

        ts_s = ts_us.to(torch.float64) * 1e-6
        return vox, ts_s


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



    save_dir = None
    for subsequence_name in SUPPORTED_SEQUENCES_FEATURE_TRACKING['cear']:
        gt_path = os.path.join('config/misc/cear/gt_tracks', f'{subsequence_name}.gt.txt')
        gt_tracks = np.genfromtxt(gt_path)  # [id, t, x, y] where t is in seconds
        print(f'  Using GT path: {gt_path}')
        print(f'  GT shape: {gt_tracks.shape}')
        print(f'  GT example: {gt_tracks[:,1]}')
        data_root = Path(config['data']['data_root'])
        dataset_name = config['data']['dataset_name']
        preprocessed_name = config['data']['preprocessed_name']
        if isinstance(preprocessed_name, (list, tuple)):
          preprocessed_name = list(preprocessed_name[:-1]) + [2]
        elif isinstance(preprocessed_name, str):
          parts = preprocessed_name.split('_')
          if len(parts) > 1:
            preprocessed_name = '_'.join(parts[:-1] + ['2'])
          else:
            preprocessed_name = '2'
        save_dir = data_root / dataset_name / subsequence_name / 'events' / preprocessed_name
        print(f'  Using dataset path: {data_root}')
        print(f'  save_dir: {save_dir}')
        # Where to look for caches for this sequence
        # cache_root = (project_root / 'output' / 'inference' /
        #               config['common']['exp_name'] / 'cache' / subsequence_name)
        # print(f"🔍 Cache root: {cache_root}")

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

    # Right after you set `save_dir` and `sequence_name` (before the LCM loop):
    sequence_name = ''
    buffer = NpyWindowBuffer(cache_root=save_dir, sequence_name=sequence_name, max_ram_bytes=2 * (1 << 30))

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



            # print(f"📂 Loading cached frames for window [{to_s(start_us):.6f}, {to_s(end_us):.6f}] s "
            #       f"from {save_dir}")

            # combined_voxels, combined_timestamps = load_cached_window_fast(save_dir, sequence_name, start_us, end_us)
            # combined_voxels, combined_timestamps = load_cached_window_fast_npy(save_dir, sequence_name, start_us, end_us)
            # combined_voxels, combined_timestamps  = load_cached_window_ori(save_dir, sequence_name, start_us, end_us)
            
            # new:
            combined_voxels, combined_timestamps = buffer.get_window(start_us, end_us)
            if combined_voxels is None:
                print("✗ No cached data covering this window. Skipping.")
                continue


            # print(f"✓ Loaded {combined_voxels.shape[0]} frames "
            #       f"in range [{combined_timestamps[0]:.6f} - {combined_timestamps[-1]:.6f}] s")












            # # Convert once
            # start_s = start_us * 1e-6
            # end_s   = end_us   * 1e-6

            # sample_all_voxels = []
            # sample_all_timestamps = []
            # append_vox = sample_all_voxels.append
            # append_ts  = sample_all_timestamps.append

            # frames_collected = 0
            # last_ts_s: Optional[float] = None  # track last appended timestamp (sec)








            print(f"Current time3: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            # continue

            # Combine after loop
            # if sample_all_voxels:
            #     combined_voxels = torch.cat(sample_all_voxels, dim=0)
            #     combined_timestamps = torch.cat(sample_all_timestamps, dim=0)

            #     if DEBUG_PRINTS:
            #         print("Combined timestamps:", [f"{ts:.6f}" for ts in combined_timestamps])
            #         print(f"📦 Aggregated {combined_voxels.shape[0]} frames "
            #               f"in range [{combined_timestamps[0]:.6f} - {combined_timestamps[-1]:.6f}] s")

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
