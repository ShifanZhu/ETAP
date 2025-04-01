import os

import torch
import pytorch_lightning as pl
import numpy as np

from ..utils import EtapData
from src.utils import SUPPORTED_SEQUENCES_FEATURE_TRACKING

class FeatureTrackingDataModule(pl.LightningDataModule):
    def __init__(self, data_root,
                 dataset_name,
                 preprocessed_name=None):
        super().__init__()
        self.save_hyperparameters()
        self.test_datasets = []

    def prepare_data(self):
        if self.hparams.dataset_name == 'eds':
            supported_sequences = SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']
            data_roots = [self.hparams.data_root for _ in SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']]
        elif self.hparams.dataset_name == 'ec':
            supported_sequences = SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']
            data_roots = [self.hparams.data_root for _ in SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']]
        elif self.hparams.dataset_name == 'feature_tracking_online':
            supported_sequences = SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds'] + SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']
            data_root_eds = os.path.join(self.hparams.data_root, 'eds')
            data_root_ec = os.path.join(self.hparams.data_root, 'ec')
            data_roots = [data_root_eds for _ in SUPPORTED_SEQUENCES_FEATURE_TRACKING['eds']] \
                       + [data_root_ec for _ in SUPPORTED_SEQUENCES_FEATURE_TRACKING['ec']]
        else:
            raise ValueError(f"Unsupported dataset_name: {self.hparams.dataset_name}")

        for subsequence_name, data_root in zip(supported_sequences, data_roots):
            self.test_datasets.append(FeatureTrackingInferenceDataset(
                data_root=data_root,
                subsequence_name=subsequence_name,
                preprocessed_name=self.hparams.preprocessed_name
            ))

class FeatureTrackingInferenceDataset(torch.utils.data.Dataset):
    """Dataset for a sequence of the EDS or EC dataset for point tracking
    as used in https://arxiv.org/pdf/2211.12826.
    This dataset is implemented for use in an online manner.
    Each item provides the data chunk for 1 step (e.g. 8 frames)
    of the sequence. The next data will start from the previous + stride frames.
    """
    def __init__(
        self,
        data_root,
        subsequence_name,
        stride=4,
        sliding_window_len=8,
        preprocessed_name=None
    ):
        self.subsequence_name = subsequence_name
        self.stride = stride
        self.sliding_window_len = sliding_window_len
        self.seq_root = os.path.join(data_root, subsequence_name)
        assert preprocessed_name is not None, 'online processing of raw events not supported.'
        self.preprocessed_name = preprocessed_name

        events_path = os.path.join(self.seq_root, 'events', preprocessed_name)
        self.samples, gt_ts_for_sanity_check = self.load_event_representations(events_path)

        gt_path = os.path.join('config/misc', os.path.basename(os.path.normpath(data_root)), 'gt_tracks', f'{subsequence_name}.gt.txt')
        gt_tracks = np.genfromtxt(gt_path)  # [id, t, x, y]
        self.gt_tracks = torch.from_numpy(self.reformat_tracks(gt_tracks)).float()

        self.gt_times_s = np.unique(gt_tracks[:, 1])
        self.gt_times_us = (1e6 * self.gt_times_s).astype(int)
        assert np.allclose(gt_ts_for_sanity_check, self.gt_times_us)

        self.start_indices = np.arange(0, len(self.gt_times_us), stride)

        # Queries
        N = self.gt_tracks.shape[1]
        query_xy = self.gt_tracks[0, :]
        query_t = torch.zeros(N, dtype=torch.int64, device=query_xy.device)
        self.query_points = torch.cat([query_t[:, None], query_xy], dim=-1)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.sliding_window_len
        ev_repr = []

        for t in range(start_idx, end_idx):
            sample = np.load(os.path.join(self.seq_root, 'events', self.preprocessed_name,
                                           f'{self.gt_times_us[t]}.npy'))
            ev_repr.append(sample)

        ev_repr = torch.from_numpy(np.stack(ev_repr, axis=0)).float()

        gt_tracks = self.gt_tracks[start_idx:end_idx]
        height, width = ev_repr.shape[-2:]
        visibility = self.generate_visibility_mask(gt_tracks, height, width)

        sample = EtapData(
            voxels=ev_repr,  # [T, C, H, W]
            rgbs=None,  # [T, 3, H, W], only for visualization
            trajectory=gt_tracks,  # [T, N, 2]
            visibility=visibility,  # [T, N]
            timestamps=torch.from_numpy(self.gt_times_s[start_idx:end_idx])  # [T]
        )
        return sample, start_idx

    def load_event_representations(self, events_path):
        repr_files = [f for f in os.listdir(events_path) if f.endswith('.npy')]

        # Extract the integer numbers from the file names
        timestamps = []
        for file in repr_files:
            number = int(os.path.splitext(file)[0])
            timestamps.append(number)

        return repr_files, sorted(timestamps)

    def reformat_tracks(self, tracks):
        # Extract unique timestamps and point IDs
        timestamps = np.unique(tracks[:, 1])
        point_ids = np.unique(tracks[:, 0]).astype(int)

        T = len(timestamps)
        N = len(point_ids)
        output = np.full((T, N, 2), np.nan)
        time_to_index = {t: i for i, t in enumerate(timestamps)}

        for row in tracks:
            point_id, timestamp, x, y = row
            t_index = time_to_index[timestamp]
            p_index = int(point_id)
            output[t_index, p_index, :] = [x, y]

        return output

    def generate_visibility_mask(self, points, height, width):
        x = points[..., 0]
        y = points[..., 1]
        x_in_bounds = (x >= 0) & (x < width)
        y_in_bounds = (y >= 0) & (y < height)
        mask = x_in_bounds & y_in_bounds
        return mask
