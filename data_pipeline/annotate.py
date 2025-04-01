import argparse
import os
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path"    ,           required=True)
    ap.add_argument("--start_index"     , type=int, required=True)
    ap.add_argument("--end_index"       , type=int, required=True)

    ap.add_argument("--resolution"      , type=int, required=True)
    ap.add_argument("--num_frames"      , type=int, required=True)
    ap.add_argument("--tracks_to_sample", type=int, default=2048)

    args            = vars(ap.parse_args())
    dataset_path    = args["dataset_path"]
    start_index     = args["start_index"]
    end_index       = args["end_index"]

    resolution      = args["resolution"]
    num_frames      = args["num_frames"]
    tracks_to_sample= args["tracks_to_sample"]

    example_counter = start_index

    def get_current_example_path():
        return os.path.join(dataset_path, f"{example_counter:08d}")

    while example_counter < end_index:
        print(f"Annotating example {example_counter}")

        script = ["python3",
                  "annotator.py",
                  f"--scene_dir={os.path.join(get_current_example_path(), 'raw')}",
                  f"--resolution={resolution}",
                  f"--num_frames={num_frames}",
                  f"--output_dir={get_current_example_path()}",
                  f"--tracks_to_sample={tracks_to_sample}"]

        annotate_result = subprocess.run(script, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if annotate_result.returncode == 0:
            print(f"Successfully annotated example {example_counter}")
        else:
            print(f"Failed to annotate example {example_counter}, return code: {annotate_result.returncode}")
            break

        example_counter += 1

if __name__ == "__main__":
    main()