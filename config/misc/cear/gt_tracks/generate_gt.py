import random

def generate_event_data(
    num_lines=1000,
    num_features=10,
    start_time=1704749445.580231,
    time_step=0.013,
    uv_min=0,
    uv_max=200,
    smooth=True,
    step_std=2.0,
    seed=None
):
    """
    Generate synthetic event-like data.

    Output order:
    All lines for feature 0 first (across all timestamps),
    then all lines for feature 1, etc.
    """
    rng = random.Random(seed)
    lines = []

    # Initialize persistent positions for each feature
    xs = [rng.randint(uv_min, uv_max) for _ in range(num_features)]
    ys = [rng.randint(uv_min, uv_max) for _ in range(num_features)]

    for fid in range(num_features):
        current_time = start_time
        for _ in range(num_lines):
            if smooth:
                xs[fid] += int(rng.gauss(0, step_std))
                ys[fid] += int(rng.gauss(0, step_std))
                xs[fid] = max(uv_min, min(uv_max, xs[fid]))
                ys[fid] = max(uv_min, min(uv_max, ys[fid]))
            else:
                xs[fid] = rng.randint(uv_min, uv_max)
                ys[fid] = rng.randint(uv_min, uv_max)

            lines.append(f"{fid} {current_time:.6f} {xs[fid]} {ys[fid]}")
            current_time += time_step

    return lines


if __name__ == "__main__":
    start_timestamp = 1704749445.580231
    num_lines = 200            # number of timestamps per feature
    num_features = 20         # number of unique IDs
    uv_min = 0
    uv_max = 200
    time_step = 0.013
    smooth = True
    step_std = 2.0
    seed = 42

    events = generate_event_data(
        num_lines=num_lines,
        num_features=num_features,
        start_time=start_timestamp,
        time_step=time_step,
        uv_min=uv_min,
        uv_max=uv_max,
        smooth=smooth,
        step_std=step_std,
        seed=seed
    )

    output_file = "mocap1_well-lit_trot.gt.txt"
    with open(output_file, "w") as f:
        for line in events:
            f.write(line + "\n")

    print(f"Wrote {len(events)} lines to {output_file}")
