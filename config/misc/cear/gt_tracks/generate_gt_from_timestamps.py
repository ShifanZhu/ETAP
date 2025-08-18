import os
import random

def load_rgb_timestamps(txt_path):
    """
    Read lines like:
      1704749404924554_depth_rgb.png
      1704749404928000_rgb.png
    and return a sorted list of timestamps in SECONDS (float), taken only from lines
    that end with '_rgb.png' and do NOT contain 'depth'.
    """
    timestamps_us = []
    with open(txt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Only keep raw RGB (exclude depth_*)
            if not line.endswith("_rgb.png"):
                continue
            if "depth" in line:
                continue
            # Extract the integer microsecond prefix (before the first underscore or dot)
            # e.g. "1704749404928000_rgb.png" -> "1704749404928000"
            token = line.split("_", 1)[0]
            # Safety: strip possible extension if there's no underscore
            token = token.split(".", 1)[0]

            try:
                ts_us = int(token)
                timestamps_us.append(ts_us)
            except ValueError:
                # skip malformed lines
                continue

    # Sort and deduplicate (preserve order)
    timestamps_us = sorted(dict.fromkeys(timestamps_us))
    # Convert to seconds as float
    timestamps_s = [ts / 1e6 for ts in timestamps_us]
    return timestamps_s


def generate_event_data_from_timestamps(
    timestamps_s,
    num_features=10,
    uv_min=0,
    uv_max=200,
    smooth=True,
    step_std=2.0,
    seed=None
):
    """
    Generate synthetic event-like data using a provided list of timestamps (in seconds).
    Output order:
      All lines for feature 0 first (across all timestamps),
      then all lines for feature 1, etc.
    """
    rng = random.Random(seed)
    lines = []

    # Persistent positions per feature
    xs = [rng.randint(uv_min, uv_max) for _ in range(num_features)]
    ys = [rng.randint(uv_min, uv_max) for _ in range(num_features)]

    for fid in range(num_features):
        for t_sec in timestamps_s:
            if smooth:
                xs[fid] += int(rng.gauss(0, step_std))
                ys[fid] += int(rng.gauss(0, step_std))
                xs[fid] = max(uv_min, min(uv_max, xs[fid]))
                ys[fid] = max(uv_min, min(uv_max, ys[fid]))
            else:
                xs[fid] = rng.randint(uv_min, uv_max)
                ys[fid] = rng.randint(uv_min, uv_max)

            # Print timestamp with 6 decimals (no scientific notation)
            lines.append(f"{fid} {t_sec:.6f} {xs[fid]} {ys[fid]}")

    return lines


if __name__ == "__main__":
    # Path to your timestamp list
    # timestamp_file = "realsense_timestamp_kitchen_hdr_comb.txt"  # <-- change if needed
    timestamp_file = "realsense_timestamp_mocap1_blinking_comb.txt"  # <-- change if needed
    if not os.path.exists(timestamp_file):
        raise FileNotFoundError(f"Cannot find {timestamp_file}")

    # Load timestamps (seconds) from _rgb.png entries only
    timestamps_s = load_rgb_timestamps(timestamp_file)
    if not timestamps_s:
        raise RuntimeError("No valid _rgb.png timestamps found.")

    # Configs
    num_features = 1
    uv_min = 0
    uv_max = 200
    smooth = True
    step_std = 2.0
    seed = 42

    # Generate lines using provided timestamps
    events = generate_event_data_from_timestamps(
        timestamps_s=timestamps_s,
        num_features=num_features,
        uv_min=uv_min,
        uv_max=uv_max,
        smooth=smooth,
        step_std=step_std,
        seed=seed
    )

    # output_file = "mocap1_well-lit_trot.gt.txt"
    # output_file = "between_buildings_day_trot.gt.txt"
    output_file = "mocap1_blinking_comb.gt.txt"
    with open(output_file, "w") as f:
        for i, line in enumerate(events):
            # if (i+1) % 2 == 0:
            #     print("Skipping line:", i+1)
            #     continue
            f.write(line + "\n")

    print(f"Wrote {len(events)} lines to {output_file}")
    print(f"Used {len(timestamps_s)} timestamps from {timestamp_file}")
