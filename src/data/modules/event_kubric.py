import torch
import numpy as np
import h5py
from pathlib import Path
import pytorch_lightning as pl
from ..utils import EtapData

def calculate_frame_times(num_frames=96, total_time=2.0):
    time_step = total_time / (num_frames - 1)
    frame_times = np.arange(num_frames) * time_step
    return frame_times

class EventKubricDataModule(pl.LightningDataModule):
    def __init__(self, data_root, seq_len, traj_per_sample,
                 dataset_name, preprocessed_name=None):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        test_path = Path(self.hparams.data_root) / 'test'
        self.test_dataset = EventKubricDataset(
            data_root=test_path,
            seq_len=self.hparams.seq_len,
            traj_per_sample=self.hparams.traj_per_sample,
            preprocessed_name=self.hparams.preprocessed_name,
        )

class EventKubricDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        preprocessed_name=None,
    ):
        super(EventKubricDataset, self).__init__()
        self.data_root = data_root
        self.traj_per_sample = traj_per_sample
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.seq_len = seq_len

        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.preprocessed_name = preprocessed_name
        self.indices = np.arange(3, 93, 3)[3:-3]

        data_root_path = Path(self.data_root)
        self.samples = [d for d in data_root_path.iterdir() if d.is_dir()]

        # Validate samples
        valid_samples = []
        for sample_path in self.samples:
            gt_path = sample_path / 'annotations.npy'
            if not gt_path.exists():
                continue
            gt_data = np.load(str(gt_path), allow_pickle=True).item()
            visibility = gt_data['occluded']
            if len(visibility) > self.traj_per_sample:
                valid_samples.append(sample_path.name)

        self.samples = valid_samples
        print(f"Found {len(self.samples)} valid samples")

    def rgba_to_rgb(self, rgba):
        if rgba.shape[-1] == 3:
            return rgba

        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]
        alpha = alpha.astype(np.float32) / 255.0
        rgb = rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)

        return rgb.astype(np.uint8)

    def load_rgb_frames(self, seq_path):
        seq_path = Path(seq_path)
        h5_path = seq_path / 'data.hdf5'
        if h5_path.exists():
            with h5py.File(str(h5_path), 'r') as f:
                if 'rgba' in f:
                    rgba = f['rgba'][self.indices]
                    rgb = self.rgba_to_rgb(rgba)
                    # Convert to (N, C, H, W) for PyTorch and correct channel order
                    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).float()
                    rgb = rgb.flip(1)  # Flip the channels to correct the order (BGR -> RGB)
                    return rgb
                else:
                    raise KeyError("'rgba' dataset not found in data.hdf5")
        raise FileNotFoundError(f"Could not find data.hdf5 in {seq_path}")

    def load_ground_truth(self, seq_path, seq_name):
        """
        Load ground truth trajectory data from annotations file.
        
        Args:
            seq_path: Path to sequence directory
            seq_name: Name of the sequence
            
        Returns:
            tuple: (trajectory data, visibility data)
        """
        data_root_path = Path(self.data_root)
        gt_path = data_root_path / seq_name / 'annotations.npy'
        gt_data = np.load(str(gt_path), allow_pickle=True).item()
        
        # Extract and process trajectory data
        traj_2d = gt_data['target_points']
        visibility = gt_data['occluded']  # Here a value of 1 means point is visible
        
        traj_2d = np.transpose(traj_2d, (1, 0, 2))[self.indices]
        visibility = np.transpose(np.logical_not(visibility), (1, 0))[self.indices]
        
        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)
        
        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        if self.sample_vis_1st_frame:
            visibile_pts_inds = visibile_pts_first_frame_inds
        else:
            visibile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(as_tuple=False)[:, 0]
            visibile_pts_inds = torch.cat(
                (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
            )

        return traj_2d, visibility, visibile_pts_inds

    def load_preprocessed_representations(self, seq_path):
        seq_path = Path(seq_path)
        ev_path = seq_path / 'event_representations' / f'{self.preprocessed_name}.h5'
        
        with h5py.File(str(ev_path), 'r') as f:
            ev_repr = f['representations'][:]
        return torch.from_numpy(ev_repr).float()

    def normalize_representation(self, repr):
        mask = repr != 0
        mean = repr.sum(dim=(0, 2, 3), keepdim=True) / mask.sum(dim=(0, 2, 3), keepdim=True)
        var = ((repr - mean)**2 * mask).sum(dim=(0, 2, 3), keepdim=True) / mask.sum(dim=(0, 2, 3), keepdim=True)
        std = torch.sqrt(var + 1e-8)
        return torch.where(mask, (repr - mean) / std, repr)

    def __getitem__(self, index):
        seq_name = self.samples[index]
        seq_path = Path(self.data_root) / seq_name

        traj_2d, visibility, visibile_pts_inds = self.load_ground_truth(seq_path, seq_name)

        # Select points
        point_inds = torch.arange(min(len(visibile_pts_inds), self.traj_per_sample))
        visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valids = torch.ones((self.seq_len, self.traj_per_sample))

        rgbs = self.load_rgb_frames(seq_path)
        ev_repr = self.load_preprocessed_representations(seq_path)
        
        # Channelwise std-mean normalization
        ev_repr = self.normalize_representation(ev_repr)

        _, first_positive_inds = torch.max(visibles, dim=0)  # Find first frame where each point is visible
        num_points = visibles.shape[1]
        query_coords = torch.zeros((num_points, 2), dtype=trajs.dtype)

        for p in range(num_points):
            first_frame = first_positive_inds[p].item()
            query_coords[p] = trajs[first_frame, p, :2]

        queries = torch.cat([first_positive_inds.unsqueeze(1), query_coords], dim=1)

        sample = EtapData(
            voxels=ev_repr,
            rgbs=rgbs,
            trajectory=trajs,
            visibility=visibles,
            valid=valids,
            query_points=queries,
            seq_name=seq_name,
        )
        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_full_sample(self, index):
        '''Helper function to retrieve full sample data'''
        sample = {}
        seq_name = self.samples[index]
        seq_path = Path(self.data_root) / seq_name
        h5_path = seq_path / 'data.hdf5'

        # Get Kubric data and ground truth
        with h5py.File(str(h5_path), 'r') as f:
            indices = f['subsample_indices'][:]
            sample['rgba'] = f['rgba'][indices]
            sample['depths'] = f['depth'][:]
            sample['forward_flow'] = f['forward_flow'][:]
            sample['normal'] = f['normal'][:]
            sample['object_coordinates'] = f['object_coordinates'][:]
            sample['segmentation'] = f['segmentation'][:]

        # Get timestamps
        t_min, t_max = 0.0, 2.0
        timestamps = calculate_frame_times(num_frames=96, total_time=t_max)
        sample['timestamps'] = timestamps[indices]

        # Get Events
        event_root = seq_path / 'events'
        event_paths = sorted(event_root.iterdir())
        events = []
        for event_path in event_paths:
            event_mini_batch = np.load(str(event_path))
            events.append(np.stack([
                event_mini_batch['y'],
                event_mini_batch['x'],
                event_mini_batch['t'],
                event_mini_batch['p'],
            ], axis=1))
        events = np.concatenate(events, axis=0)
        sample['events'] = events

        # Get Point Tracks
        gt_path = Path(self.data_root) / seq_name / 'annotations.npy'
        gt_data = np.load(str(gt_path), allow_pickle=True).item()
        traj_2d = gt_data['target_points']
        visibility = gt_data['occluded']  # Here a value of 1 means point is visible
        traj_2d = np.transpose(traj_2d, (1, 0, 2))[indices]
        visibility = np.transpose(np.logical_not(visibility), (1, 0))[indices]
        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)
        sample['point_tracks'] = traj_2d
        sample['visibility'] = visibility

        return sample