import h5py
import sys
import numpy as np

def print_h5_info(file_path, head=5, tail=5):
    with h5py.File(file_path, "r") as f:
        print(f"HDF5 file: {file_path}")
        print("=" * 40)

        # Print root attributes
        if f.attrs:
            print("Root attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            print()

        # Walk through datasets
        def visit_fn(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[Dataset] {name}  shape={obj.shape} dtype={obj.dtype}")
                data = obj[()]  # load into numpy array
                if data.size > 0:
                    head_data = data[:head]
                    tail_data = data[-tail:] if data.size > tail else np.array([])
                    print(f"  First {len(head_data)} values: {head_data}")
                    if tail_data.size > 0:
                        print(f"  Last  {len(tail_data)} values: {tail_data}")
                print()
            elif isinstance(obj, h5py.Group):
                print(f"[Group]    {name}")
                if obj.attrs:
                    for k, v in obj.attrs.items():
                        print(f"    attr {k}: {v}")

        f.visititems(visit_fn)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_h5_head_tail.py <file.h5>")
        sys.exit(1)

    # Example: print first 5 and last 5
    print_h5_info(sys.argv[1], head=5, tail=5)
