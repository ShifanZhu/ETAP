import torch
import numpy as np
from pathlib import Path
import pytorch_lightning as pl

from ..utils import EtapData


class E2d2DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, data_root, preprocessed_name, sequences=None):
        super().__init__()
        self.save_hyperparameters()
        self.test_datasets = []

    def prepare_data(self):
        """Prepare test datasets using preprocessed event representations."""
        if self.hparams.sequences is None:
            sequences = [d for d in self.hparams.data_root.iterdir() if d.is_dir()]
        else:
            sequences = [Path(self.hparams.data_root) / seq for seq in self.hparams.sequences]

        for sequence_path in sequences:
            if not (sequence_path / 'seq.h5').exists():
                print(f"Warning: seq.h5 not found in {sequence_path}, skipping...")
                continue
                
            self.test_datasets.append(E2D2InferenceDataset(
                sequence_path=sequence_path,
                preprocessed_name=self.hparams.preprocessed_name,
                stride=4,
                sliding_window_len=8
            ))


class E2D2InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_path,
        preprocessed_name,
        stride=4,
        sliding_window_len=8
    ):
        super().__init__()
        self.sequence_path = Path(sequence_path)
        self.subsequence_name = self.sequence_path.name
        self.stride = stride
        self.sliding_window_len = sliding_window_len

        self.ev_repr_dir = self.sequence_path / 'event_representations' / preprocessed_name

        if not self.ev_repr_dir.exists():
            raise FileNotFoundError(f"Preprocessed event representations not found in {self.sequence_path}")

        print(f"Using preprocessed event representations from {self.ev_repr_dir}")

        self.ev_repr_paths = sorted(path for path in self.ev_repr_dir.iterdir() 
                                   if path.is_file() and path.suffix == '.npy')

        if not self.ev_repr_paths:
            raise FileNotFoundError(f"No .npy files found in {self.ev_repr_dir}")

        self.timestamps = np.array([int(path.stem) for path in self.ev_repr_paths])
        
        gt_path = self.sequence_path / 'gt_positions.npy'
        if gt_path.exists():
            self.gt_tracks = torch.from_numpy(np.load(gt_path))
        else:
            self.gt_tracks = None
        
        # Load query points from file
        query_xy = torch.from_numpy(np.load(self.sequence_path / 'queries.npy'))
        query_t = torch.zeros(query_xy.shape[0], dtype=torch.int64)
        self.query_points = torch.cat([query_t[:, None], query_xy], dim=1).float()
        
        self.start_indices = np.arange(0, len(self.timestamps) - self.sliding_window_len + 1, self.stride)

    def load_event_representations(self, start_idx, end_idx):
        """Load event representations for the given index range."""
        ev_repr = []

        for i in range(start_idx, end_idx):
            ev_repr_path = self.ev_repr_paths[i]
            sample = np.load(ev_repr_path)
            ev_repr.append(sample)

        ev_repr = torch.from_numpy(np.stack(ev_repr, axis=0)).float()
        return ev_repr

    def __len__(self):
        """Return the number of start indices."""
        return len(self.start_indices)

    def __getitem__(self, idx):
        """Get a data sample for the given index."""
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.sliding_window_len

        ev_repr = self.load_event_representations(start_idx, end_idx)
        gt_tracks = self.gt_tracks[start_idx:end_idx] if self.gt_tracks is not None else None
        visibility = torch.ones((self.sliding_window_len, gt_tracks.shape[1])) if gt_tracks is not None else None

        sample = EtapData(
            voxels=ev_repr,  # [T, C, H, W]
            rgbs=None,
            trajectory=gt_tracks,  # [T, N, 2] or None
            visibility=visibility,  # [T, N] or None
            timestamps=torch.from_numpy(self.timestamps[start_idx:end_idx]).float() / 1e6,  # [T]
        )
        return sample, start_idx