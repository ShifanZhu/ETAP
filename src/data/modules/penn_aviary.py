import os
import torch
import h5py
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from src.representations import EventRepresentationFactory
from src.utils import make_grid
from ..utils import EtapData


def pad_images_to_timestamps(images, img_ts, target_timestamps):
    """Match images to the nearest timestamp."""
    indices = np.searchsorted(img_ts, target_timestamps, side='right') - 1
    indices = np.clip(indices, 0, len(images) - 1)
    imgs_padded = images[indices]
    return imgs_padded


def read_binary_mask(path):
    """Read binary mask from file."""
    img = np.array(Image.open(path).convert('L'))
    mask = (img != 255).astype(np.uint8)
    return mask

class PennAviaryDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, data_root, sequence_data, repr_config=None,
                 load_rgb=False, sequences=None):
        super().__init__()
        self.save_hyperparameters()
        self.test_datasets = []

        if self.hparams.repr_config is not None:
            self.hparams.repr_config['image_shape'] = tuple(self.hparams.repr_config['image_shape'])
            self.converter = EventRepresentationFactory.create(self.hparams.repr_config)            
        else:
            self.converter = None

    def prepare_data(self):
        data_root = Path(self.hparams.data_root)
        if self.hparams.sequences is None:
            sequences = [d for d in data_root.iterdir() if d.is_dir()]
        else:
            sequences = [data_root / seq for seq in self.hparams.sequences]

        for sequence_path in sequences:
            subsequence_name = sequence_path.name

            if not (sequence_path / 'seq.h5').exists():
                print(f"Warning: seq.h5 not found in {sequence_path}, skipping...")
                continue

            self.test_datasets.append(PennAviaryDataset(
                sequence_path=sequence_path,
                num_events=self.hparams.sequence_data[subsequence_name]['num_events'],
                start_time_s=self.hparams.sequence_data[subsequence_name]['start_time_s'],
                duration_s=self.hparams.sequence_data[subsequence_name]['duration_s'],
                step_time_s=self.hparams.sequence_data[subsequence_name]['step_time_s'],
                converter=self.converter,
                load_rgb=self.hparams.load_rgb,
                query_stride=self.hparams.sequence_data[subsequence_name].get('query_stride', 40),
                mask_name=self.hparams.sequence_data[subsequence_name]['mask_name']
            ))


class PennAviaryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_path,
        start_time_s,
        duration_s,
        step_time_s,
        num_events,
        stride=4,
        sliding_window_len=8,
        load_rgb=False,
        converter=None,
        query_stride=40,
        mask_name=None
    ):
        super().__init__()
        self.sequence_path = Path(sequence_path)
        self.subsequence_name = self.sequence_path.name
        self.num_events = num_events
        self.stride = stride
        self.sliding_window_len = sliding_window_len
        self.load_rgb = load_rgb

        # Define tracking timestamps
        start_time_us = start_time_s * 1e6
        end_time_us = start_time_us + duration_s * 1e6
        step_time_us = step_time_s * 1e6
        self.timestamps = np.arange(start_time_us, end_time_us, step_time_us)
        self.gt_tracks = None

        # Generate query points grid
        height, width = 480, 640
        query_xy = torch.from_numpy(make_grid(height, width, stride=query_stride))
        num_queries = query_xy.shape[0]
        query_t = torch.zeros(num_queries, dtype=torch.int64)
        self.query_points = torch.cat([query_t[:, None], query_xy], dim=1).float()

        # Only query points within mask
        if mask_name is not None:
            mask_path = self.sequence_path / mask_name
            segm_mask = torch.from_numpy(read_binary_mask(mask_path))
            query_x = self.query_points[:, 1].int()
            query_y = self.query_points[:, 2].int()
            segm_mask = segm_mask[query_y, query_x]
            self.query_points = self.query_points[segm_mask == 1]

        self.h5_path = self.sequence_path / 'seq.h5'

        if self.load_rgb:
            with h5py.File(self.h5_path, 'r') as f:
                images = f['images'][:]
                img_ts = f['img_ts'][:]
                self.imgs_padded = pad_images_to_timestamps(images, img_ts, self.timestamps)

        load_ev_repr_from_file = True if len(self.timestamps) > 1000 else False

        if converter is not None and not load_ev_repr_from_file:
            self.load_ev_repr = False
            self.ev_repr = self.create_representations(self.h5_path, converter, self.timestamps)
        else:
            self.load_ev_repr = True
            ev_repr_dir = self.sequence_path / str(int(1e6 * step_time_s)).zfill(9)
            self.ev_repr_paths = sorted(path for path in ev_repr_dir.iterdir() if path.is_file())
            assert len(self.ev_repr_paths) == len(self.timestamps), f"Expected {len(self.timestamps)} event representation files, but found {len(self.ev_repr_paths)}"

        self.start_indices = np.arange(0, len(self.timestamps) - self.stride, self.stride)

    def create_representations(self, h5_path, converter, timestamps):
        with h5py.File(h5_path, 'r') as f:
            indices_end = np.searchsorted(f['t'], timestamps) - 1
            indices_start = indices_end - self.num_events

            representations = []

            for i_start, i_end in tqdm(zip(indices_start, indices_end), 
                          desc=f"{self.subsequence_name}: creating event representations",
                          total=len(indices_start)):
                events = np.stack([f['y'][i_start:i_end],
                                f['x'][i_start:i_end],
                                f['t'][i_start:i_end],
                                f['p'][i_start:i_end]], axis=-1)
                repr = converter(events, t_mid=None)
                representations.append(repr)

            representations = np.stack(representations)

        return representations

    def load_event_representations(self, start_idx, end_idx):
        ev_repr = []

        for i in range(start_idx, end_idx):
            ev_repr_path = self.ev_repr_paths[i]
            sample = np.load(ev_repr_path)
            ev_repr.append(sample)

        ev_repr = torch.from_numpy(np.stack(ev_repr, axis=0)).float()
        return ev_repr

    def __len__(self):
        return len(self.start_indices) 

    def __getitem__(self, idx):
        # Load event representation
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.sliding_window_len

        if self.load_ev_repr:
            ev_repr = self.load_event_representations(start_idx, end_idx)
        else:
            ev_repr = self.ev_repr[start_idx:end_idx]
            ev_repr = torch.from_numpy(np.stack(ev_repr, axis=0)).float()

        if self.load_rgb:
            rgbs = self.imgs_padded[start_idx:end_idx]
            rgbs = torch.from_numpy(rgbs).float().permute(0, 3, 1, 2)
        else:
            rgbs = None

        sample = EtapData(
            voxels=ev_repr,  # [T, C, H, W]
            rgbs=rgbs,  # [T, 3, H, W], only for visualization
            trajectory=None,
            visibility=None,
            timestamps=torch.from_numpy(self.timestamps[start_idx:end_idx]).float() / 1e6,  # [T]
        )
        return sample, start_idx