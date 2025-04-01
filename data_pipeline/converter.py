import numpy as np
from rpg_vid2e.upsampling.utils import Upsampler
from rpg_vid2e.esim_torch.scripts.generate_events import process_dir

import argparse
import shutil
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--output_dir", required=True)

    # Event generation parameters
    ap.add_argument("--frame_rate", type=int  , default=48)
    ap.add_argument("--num_frames", type=int  , default=96)
    ap.add_argument("--ct_lower",   type=float, default=0.16)
    ap.add_argument("--ct_upper",   type=float, default=0.34)
    ap.add_argument("--ref_period", type=int,   default=0)

    ## Only for vid2e's process_dir
    ap.add_argument("--contrast_threshold_negative", "-cp", type=float, default=0.2)
    ap.add_argument("--contrast_threshold_positive", "-cn", type=float, default=0.2)
    ap.add_argument("--refractory_period_ns", "-rp", type=int, default=0)

    args = vars(ap.parse_args())
    scene_dir = args["scene_dir"]
    output_dir = args["output_dir"]

    frame_rate   = args["frame_rate"]
    num_frames   = args["num_frames"]
    ct_lower     = args["ct_lower"]
    ct_upper     = args["ct_upper"]
    ref_period   = args["ref_period"]

    tmpf = os.path.join(output_dir, "tmp")
    os.makedirs(os.path.join(tmpf, "seq", "imgs"))

    rgbs = [f"rgba_{i:05d}.png" for i in range(num_frames)]

    for rgb in rgbs:
        shutil.copy(os.path.join(scene_dir, rgb),
                    os.path.join(tmpf, "seq", "imgs", rgb.split("_")[1]))

    # Upsample frames
    fpsf = open(os.path.join(tmpf, "seq", "fps.txt"), "w")
    fpsf.write(str(frame_rate))
    fpsf.close()

    upsampler = Upsampler(input_dir=os.path.join(tmpf, "seq"),
                          output_dir=os.path.join(tmpf, "seq_upsampled"))
    upsampler.upsample()

    # Generate events
    vid2e_args = ap.parse_args()
    vid2e_args.contrast_threshold_positive = np.random.uniform(ct_lower, ct_upper)
    vid2e_args.contrast_threshold_negative = np.random.uniform(ct_lower, ct_upper)
    vid2e_args.refractory_period_ns = ref_period

    process_dir(os.path.join(output_dir, "events"),
                os.path.join(tmpf, "seq_upsampled"),
                vid2e_args)

    # Remove temporary files
    shutil.rmtree(tmpf)

if __name__ == "__main__":
    main()