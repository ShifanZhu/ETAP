import argparse
import subprocess
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--start_index" , type=int, required=True)
    ap.add_argument("--end_index"   , type=int, required=True)

    # Event generation parameters
    ap.add_argument("--frame_rate", type=int  , default=48)
    ap.add_argument("--num_frames", type=int  , default=96)
    ap.add_argument("--ct_lower",   type=float, default=0.16)
    ap.add_argument("--ct_upper",   type=float, default=0.34)
    ap.add_argument("--ref_period", type=int,   default=0)

    args = vars(ap.parse_args())
    dataset_path = args["dataset_path"]
    start_index  = args["start_index"]
    end_index    = args["end_index"]

    frame_rate   = args["frame_rate"]
    num_frames   = args["num_frames"]
    ct_lower     = args["ct_lower"]
    ct_upper     = args["ct_upper"]
    ref_period   = args["ref_period"]

    example_counter = start_index

    def get_current_example_path():
        return os.path.join(dataset_path, f"{example_counter:08d}")

    while example_counter < end_index:
        print(f"Converting example {example_counter}")

        script = ["python3",
                  "converter.py",
                  f"--scene_dir={os.path.join(get_current_example_path(), 'raw')}",
                  f"--output_dir={get_current_example_path()}",
                  f"--frame_rate={frame_rate}",
                  f"--num_frames={num_frames}",
                  f"--ct_lower={ct_lower}",
                  f"--ct_upper={ct_upper}",
                  f"--ref_period={ref_period}"]

        convert_result = subprocess.run(script, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if convert_result.returncode == 0:
            print(f"Successfully converted example {example_counter}")
        else:
            print(f"Failed to convert example {example_counter}, return code: {convert_result.returncode}")
            break

        example_counter += 1

if __name__ == "__main__":
    main()