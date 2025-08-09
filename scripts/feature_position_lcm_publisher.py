#!/usr/bin/env python3
"""
Publish a single TrackingCommand over LCM.

Schema (fixed arrays of length 50):
package msgs;

struct TrackingCommand
{
    int64_t   start_time_ns;
    int64_t   end_time_ns;

    int32_t feature_ids[50];
    float   feature_x[50];   // same length as feature_ids
    float   feature_y[50];   // same length as feature_ids
}

Usage examples:
  # Manual points (two features), 5s window starting now
  python publish_command.py --topic TRACKING_COMMAND --manual 100 100 130 210 --duration-s 5

  # Manual points with explicit window (ns)
  python publish_command.py --manual 100 100 130 210 \
      --start-ns 1723114500000000000 --end-ns 1723114505000000000

  # Explicit IDs/coords, 10s window starting now
  python publish_command.py --single --ids 1 2 --xs 100 130 --ys 100 210 --duration-s 10

  # From ETAP npz: use first-frame coords + [first_ts .. last_ts]
  python publish_command.py --from-npz output/inference/exp1/seq001.npz --topic TRACKING_COMMAND

  # From ETAP txt: use earliest-timestamp coords + [min_ts .. max_ts]
  python publish_command.py --from-txt output/inference/exp1/seq001.txt --topic TRACKING_COMMAND
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np
import lcm

# ---- Import generated LCM bindings (try common layouts) ----
repo_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_dir / "lcm_gen"))
sys.path.append(str(repo_dir / "scripts" / "lcm_types"))

try:
    from msgs import TrackingCommand
except Exception:
    from lcm_types.msgs import TrackingCommand  # type: ignore


MAXF = 50  # fixed length from your .lcm schema


def now_ns() -> int:
    return int(time.time_ns())


def to_ns(ts_s: float) -> int:
    # robust conversion from seconds to integer nanoseconds
    return int(round(float(ts_s) * 1e9))


def define_features(points_xy, start_id=0):
    """points_xy: list of [x,y]; returns (ids, xs, ys)"""
    ids = list(range(start_id, start_id + len(points_xy)))
    xs = [float(p[0]) for p in points_xy]
    ys = [float(p[1]) for p in points_xy]
    return ids, xs, ys


def pad_fixed_50(ids, xs, ys):
    """Pad/truncate to exactly 50 to satisfy fixed-size arrays."""
    n = min(len(ids), len(xs), len(ys), MAXF)
    if len(ids) > MAXF:
        print(f"[WARN] Truncating {len(ids)} features to {MAXF}.")
    ids_f = list(ids[:n]) + [-1]  * (MAXF - n)
    xs_f  = list(xs[:n])  + [0.0] * (MAXF - n)
    ys_f  = list(ys[:n])  + [0.0] * (MAXF - n)
    return ids_f, xs_f, ys_f


def build_command(start_ns: int, end_ns: int, ids, xs, ys) -> TrackingCommand:
    msg = TrackingCommand()
    msg.start_time_ns = int(start_ns)
    msg.end_time_ns   = int(end_ns)
    ids_f, xs_f, ys_f = pad_fixed_50(ids, xs, ys)
    msg.feature_ids = [int(v) for v in ids_f]
    msg.feature_x   = [float(v) for v in xs_f]
    msg.feature_y   = [float(v) for v in ys_f]
    return msg


def parse_manual_points(flat_values):
    """[x1 y1 x2 y2 ...] -> [[x1,y1],[x2,y2], ...]"""
    if len(flat_values) % 2 != 0:
        raise ValueError("--manual expects an even count of numbers: x y x y ...")
    it = iter(flat_values)
    return [[float(x), float(y)] for x, y in zip(it, it)]


def features_from_npz(npz_path: Path):
    """
    Returns (ids, xs, ys, start_ns, end_ns) using:
      - ids/xs/ys from first frame of coords_predicted
      - start/end from first/last timestamps
    """
    data = np.load(npz_path)
    coords = data["coords_predicted"]  # [T, N, 2]
    ts_s   = data["timestamps"]        # [T] seconds
    if coords.ndim != 3 or coords.shape[0] < 1:
        raise ValueError("coords_predicted must be [T,N,2] with T>=1")
    xy0 = coords[0]                    # [N,2] first frame
    ids = list(range(xy0.shape[0]))
    xs  = xy0[:, 0].tolist()
    ys  = xy0[:, 1].tolist()
    start_ns = to_ns(float(ts_s[0]))
    end_ns   = to_ns(float(ts_s[-1]))
    return ids, xs, ys, start_ns, end_ns


def features_from_txt(txt_path: Path):
    """
    File lines: "<id> <timestamp_s> <x> <y>"
    Use earliest timestamp group as features, and min/max timestamps as window.
    Returns (ids, xs, ys, start_ns, end_ns).
    """
    rows = []
    with open(txt_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 4:
                continue
            fid = int(float(parts[0]))  # tolerate "1.0"
            ts_s = float(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            rows.append((ts_s, fid, x, y))

    if not rows:
        raise ValueError(f"No valid lines parsed from {txt_path}")

    # window bounds
    all_ts = [r[0] for r in rows]
    start_ns = to_ns(min(all_ts))
    end_ns   = to_ns(max(all_ts))

    # build features from the earliest timestamp group
    earliest = min(all_ts)
    items = [(fid, x, y) for (ts, fid, x, y) in rows if ts == earliest]
    items.sort(key=lambda z: z[0])
    ids = [it[0] for it in items]
    xs  = [it[1] for it in items]
    ys  = [it[2] for it in items]
    return ids, xs, ys, start_ns, end_ns


def main():
    ap = argparse.ArgumentParser(description="Publish a single TrackingCommand via LCM")
    ap.add_argument("--topic", type=str, default="TRACKING_COMMAND", help="LCM topic name")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manual", type=float, nargs="+",
                     help="Manual points: x y x y ... (IDs auto-assigned starting at --start-id)")
    src.add_argument("--single", action="store_true",
                     help="Use explicit --ids/--xs/--ys")
    src.add_argument("--from-npz", type=str,
                     help="Load first-frame coords and time window from ETAP .npz")
    src.add_argument("--from-txt", type=str,
                     help="Load earliest-frame coords and time window from ETAP .txt (id ts x y)")

    # Manual options
    ap.add_argument("--start-id", type=int, default=1000, help="Start ID for --manual")

    # Single options
    ap.add_argument("--ids", type=int, nargs="+", help="Feature IDs for --single")
    ap.add_argument("--xs", type=float, nargs="+", help="Xs for --single")
    ap.add_argument("--ys", type=float, nargs="+", help="Ys for --single")

    # Time window options
    ap.add_argument("--start-ns", type=int, default=None, help="Override start_time_ns")
    ap.add_argument("--end-ns",   type=int, default=None, help="Override end_time_ns")
    ap.add_argument("--duration-s", type=float, default=5.0,
                    help="If start/end not provided and no data file, use [now .. now+duration]")

    args = ap.parse_args()
    lc = lcm.LCM()

    # Figure out features and default window
    if args.manual is not None:
        pts = parse_manual_points(args.manual)
        ids, xs, ys = define_features(pts, start_id=args.start_id)
        default_start = now_ns()
        default_end   = default_start + int(args.duration_s * 1e9)
    elif args.single:
        if args.ids is None or args.xs is None or args.ys is None:
            ap.error("--single requires --ids, --xs, and --ys")
        if not (len(args.ids) == len(args.xs) == len(args.ys)):
            ap.error("Lengths of --ids, --xs, --ys must match")
        ids, xs, ys = args.ids, args.xs, args.ys
        default_start = now_ns()
        default_end   = default_start + int(args.duration_s * 1e9)
    elif args.from_npz:
        npz_path = Path(args.from_npz)
        if not npz_path.exists():
            ap.error(f"NPZ not found: {npz_path}")
        ids, xs, ys, default_start, default_end = features_from_npz(npz_path)
    elif args.from_txt:
        txt_path = Path(args.from_txt)
        if not txt_path.exists():
            ap.error(f"TXT not found: {txt_path}")
        ids, xs, ys, default_start, default_end = features_from_txt(txt_path)
    else:
        ap.error("one of --manual / --single / --from-npz / --from-txt is required")

    # Resolve window (CLI overrides win)
    start_ns = args.start_ns if args.start_ns is not None else default_start
    end_ns   = args.end_ns   if args.end_ns   is not None else default_end
    if end_ns <= start_ns:
        ap.error(f"end_time_ns ({end_ns}) must be > start_time_ns ({start_ns})")

    # Build and publish command
    msg = build_command(start_ns, end_ns, ids, xs, ys)
    lc.publish(args.topic, msg.encode())

    # Brief printout
    n_valid = min(len(ids), MAXF)
    print(f"[OK] Published TrackingCommand to '{args.topic}' "
          f"window=[{start_ns} .. {end_ns}] with {n_valid} features "
          f"(padded/truncated to {MAXF}).")


if __name__ == "__main__":
    main()

# # Example usage:
# # Publish two points: (100,100) and (130,210)
# python publish_features.py --manual 100 100 130 210
# # Publish three points with custom starting ID = 500
# python publish_features.py --manual 50 60 70 80 90 100 --start-id 500
# # Publish with a fixed timestamp (nanoseconds)
# python publish_features.py --manual 100 200 300 400 --timestamp-ns 1723114500000000000

# # Publish IDs 1, 2 with coords (100,100) and (130,210)
# python publish_features.py --single --ids 1 2 --xs 100 130 --ys 100 210
# # Publish with timestamp override
# python publish_features.py --single \
#     --ids 101 102 103 \
#     --xs 10.5 20.0 30.2 \
#     --ys 50.1 49.9 48.7 \
#     --timestamp-ns 1723114500000000000

# # Stream at 30 FPS once
# python publish_features.py --npz output/inference/exp1/seq001.npz --fps 30
# # Stream at 10 FPS and repeat forever
# python publish_features.py --npz output/inference/exp1/seq001.npz --fps 10 --repeat
# # Stream as fast as possible (no delay)
# python publish_features.py --npz output/inference/exp1/seq001.npz --fps 0

# # Stream at 30 FPS once
# python publish_features.py --txt output/inference/exp1/seq001.txt --fps 30
# # Stream at 5 FPS and repeat forever
# python publish_features.py --txt output/inference/exp1/seq001.txt --fps 5 --repeat
# # Stream as fast as possible
# python publish_features.py --txt output/inference/exp1/seq001.txt --fps 0