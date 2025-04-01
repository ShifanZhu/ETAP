import argparse
import subprocess
import shutil
import time
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir" ,           required=True)
    ap.add_argument("--start_index", type=int, required=True)
    ap.add_argument("--end_index"  , type=int, required=True)
    ap.add_argument("--worker_script",         required=True)
    ap.add_argument("--panning"    , action="store_true")

    args          = vars(ap.parse_args())
    output_dir    = args["output_dir"]
    start_index   = args["start_index"]
    end_index     = args["end_index"]
    worker_script = args["worker_script"] + ".py"
    panning       = args["panning"]

    example_counter = start_index
    os.makedirs(output_dir, exist_ok=True)

    def get_current_output_dir():
        return os.path.join(output_dir, f"{example_counter:08d}")

    while example_counter < end_index:
        print(f"Generating example {example_counter}")

        script = ["python3",
                  os.path.join("kubric", "challenges", "movi", worker_script),
                  f"--job-dir={os.path.join(get_current_output_dir(), 'raw')}"]

        if panning:
            script.append("--camera=linear_movement_linear_lookat")
        else:
            script.append("--camera=linear_movement")

        # Generate training example
        kubric_result = subprocess.run(script, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Regenerate training example on error
        if kubric_result.returncode == 0:
            print(f"Successfully generated example {example_counter}")
        else:
            print(f"Failed to generate example {example_counter}, return code: {kubric_result.returncode}")
            print("Retrying in 10 seconds . . .")

            if os.path.exists(get_current_output_dir()):
                shutil.rmtree(get_current_output_dir())

            time.sleep(10)
            continue

        example_counter += 1

if __name__ == "__main__":
    main()
