from pathlib import Path

import h5py
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import ConcatDataset
import numpy as np

from ..utils import EtapData

class Evimo2DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, data_root, preprocessed_name, metadata_path):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        samples = []

        df = pd.read_csv(self.hparams.metadata_path)

        for _, row in df.iterrows():
            sample_path = Path(self.hparams.data_root) / row['name']
            samples.append(Evimo2SequenceDataset(
                data_root=sample_path,
                preprocessed_name=self.hparams.preprocessed_name,
                t_start=row['t_start'],
                t_end=row['t_end'],
            ))

        self.test_dataset = ConcatDataset(samples)

class Evimo2SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: Path,
        preprocessed_name: str,
        t_start: float,
        t_end: float,
    ):
        super(Evimo2SequenceDataset, self).__init__()
        self.data_root = data_root
        self.preprocessed_name = preprocessed_name
        self.t_start = t_start
        self.t_end = t_end
        
        # Load timestamps to determine valid indices
        repr_path = self.data_root / 'event_representations' / f'{self.preprocessed_name}.h5'
        with h5py.File(repr_path, 'r') as f:
            self.timestamps = f['timestamps'][:]
            
        self.valid_indices = np.where(
            (self.timestamps >= self.t_start) & 
            (self.timestamps <= self.t_end)
        )[0]
        
        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No timestamps found in range [{t_start}, {t_end}] "
                f"for sequence {self.data_root.name}"
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        repr_path = self.data_root / 'event_representations' / f'{self.preprocessed_name}.h5'

        # Event representations
        with h5py.File(repr_path, 'r') as f:
            # Only load data for valid time range
            representations = f['representations'][self.valid_indices]
            timestamps = f['timestamps'][self.valid_indices]

        representations = torch.from_numpy(representations).float()
        timestamps = torch.from_numpy(timestamps)

        # GT data
        gt_path = self.data_root / 'dataset_tracks.h5'
        with h5py.File(gt_path, 'r') as f:
            gt_tracks = f['tracks'][self.valid_indices]
            gt_occlusion = f['occlusions'][self.valid_indices]

        gt_tracks = torch.from_numpy(gt_tracks).float()
        gt_visibility = torch.from_numpy(~gt_occlusion).bool()

        # Queries
        T, N, _ = gt_tracks.shape
        first_positive_inds = torch.argmax(gt_visibility.int(), dim=0)
        query_xy = gt_tracks[first_positive_inds, torch.arange(N)]
        queries = torch.cat([first_positive_inds[..., None], query_xy], dim=-1)

        return EtapData(
            voxels=representations,  # [T', C, H, W]
            rgbs=None,
            trajectory=gt_tracks,    # [T', N, 2]
            visibility=gt_visibility,# [T', N]
            query_points=queries,
            seq_name=self.data_root.name,
            timestamps=timestamps,   # [T']
        )
