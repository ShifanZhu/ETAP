import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from src.representations import MixedDensityEventStack
from src.model.etap.model import Etap
from src.utils import Visualizer

def normalize_and_expand_channels(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input must be a torch tensor")

    *batch_dims, height, width = image.shape

    if len(image.shape) < 2:
        raise ValueError("Input tensor must have at least shape (..., height, width)")

    image_flat = image.view(-1, height, width)

    min_val = image_flat.min()
    max_val = image_flat.max()
    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    image_normalized = (image_flat - min_val) / range_val * 255
    image_rgb_flat = image_normalized.to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1)
    image_rgb = image_rgb_flat.view(*batch_dims, 3, height, width)

    return image_rgb

if __name__ == '__main__':
    device = 'cpu'
    data_dir = Path('data/demo_example')
    ckpt_path = Path('weights/ETAP_v1_cvpr25.pth')
    output_dir = Path(f'output/{data_dir.name}')
    num_bins = 10
    num_events = 60000
    height, width = 480, 640
    t_start = 1/30 # seconds
    t_end = 1.5 # seconds
    t_delta = 1/60 # seconds

    # Object to convert raw event data into grid representations
    converter = MixedDensityEventStack(
        image_shape=(height, width),
        num_stacks=num_bins,
        interpolation='bilinear',
        channel_overlap=True,
        centered_channels=False
    )

    # Load the model
    tracker = Etap(num_in_channels=num_bins, stride=4, window_len=8)
    weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    tracker.load_state_dict(weights)
    tracker = tracker.to(device)
    tracker.eval()

    # Let's choose some timestamps at which we create the frame representations
    tracking_timestamps = np.arange(t_start, t_end, t_delta)  # seconds

    # Load raw event data
    xy = np.load(data_dir / 'dataset_events_xy.npy', mmap_mode='r')
    p = np.load(data_dir / 'dataset_events_p.npy', mmap_mode='r')
    t = np.load(data_dir / 'dataset_events_t.npy', mmap_mode='r')

    assert t_start > t[0], "Start time must be greater than the first event timestamp"
    assert t_end < t[-1], "End time must be less than the last event timestamp"
    assert t_delta > 0, "Time delta must be greater than zero"
    assert t_start < t_end, "Start time must be less than end time"
    assert xy.shape[0] == p.shape[0] == t.shape[0], "Event data arrays must have the same length"

    event_indices = np.searchsorted(t, tracking_timestamps)
    event_representations = []

    # At each tracking timestep, we take the last num_events events and convert
    # them into a grid representation.
    for i_end in tqdm(event_indices, desc='Creating grid representations'):
        i_start = max(i_end - num_events, 0)
        
        events = np.stack([xy[i_start:i_end, 1],
                           xy[i_start:i_end, 0],
                            t[i_start:i_end],
                            p[i_start:i_end]], axis=1)
        ev_repr = converter(events)
        event_representations.append(ev_repr)

    voxels = np.stack(event_representations, axis=0)
    voxels = torch.from_numpy(voxels)[None].float().to(device)

    # Now let's determine some queries, meaning the initial positions
    # of the points to track.
    x = np.arange(0, width, 32)
    y = np.arange(0, height, 32)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X.flatten(), Y.flatten()], axis=1)
    query_xy = torch.from_numpy(grid).float().to(device)
    num_queries = query_xy.shape[0]
    # We track all points from the beginning (t=0)
    query_t = torch.zeros(num_queries, dtype=torch.int64, device=device)
    queries = torch.cat([query_t[:, None], query_xy], dim=1)
    queries = queries[None].to(device)

    with torch.no_grad():
        result = tracker(voxels, queries, iters=6)
        predictions, visibility = result['coords_predicted'], result['vis_predicted']

    visibility = visibility > 0.8

    # Visualization
    projection = voxels.sum(2).cpu()
    lower_bound = torch.tensor(np.percentile(projection.cpu().numpy(), 2))
    upper_bound = torch.tensor(np.percentile(projection.cpu().numpy(), 98))
    projection_clipped = torch.clamp(projection, lower_bound, upper_bound)
    video = normalize_and_expand_channels(projection_clipped)

    viz = Visualizer(save_dir=output_dir, tracks_leave_trace=-1)
    viz.visualize(
            video=video,
            tracks=predictions,
            visibility=visibility,
            filename=f"pred",
    )
    print('Done.')