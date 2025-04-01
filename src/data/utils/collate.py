import torch
import dataclasses
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(eq=False)
class EtapData:
    """
    Dataclass for storing video tracks data.
    """
    voxels: torch.Tensor  # B, S, C, H, W
    trajectory: torch.Tensor  # B, S, N, 2
    visibility: torch.Tensor  # B, S, N
    # optional data
    rgbs: Optional[torch.Tensor]  # B, S, 3, H, W, only for visualization
    valid: Optional[torch.Tensor] = None  # B, S, N
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W
    seq_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    timestamps: Optional[torch.Tensor] = None  # For EDS evaluation
    pair_indices: Optional[torch.Tensor] = None  # For contrastive loss
    video: Optional[torch.Tensor] = None  # B, S, C, H, W
    e2vid: Optional[torch.Tensor] = None  # B, S, C, H, W
    dataset_name: Optional[str] = None


def collate_fn(batch: List[EtapData]) -> EtapData:
    """
    Collate function for video tracks data.
    """
    voxels = torch.stack([b.voxels for b in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b in batch], dim=0)
    visibility = torch.stack([b.visibility for b in batch], dim=0)
    query_points = segmentation = None
    if batch[0].query_points is not None:
        query_points = torch.stack([b.query_points for b in batch], dim=0)
    if batch[0].segmentation is not None:
        segmentation = torch.stack([b.segmentation for b in batch], dim=0)
    seq_name = [b.seq_name for b in batch]

    return EtapData(
        voxels=voxels,
        trajectory=trajectory,
        visibility=visibility,
        segmentation=segmentation,
        seq_name=seq_name,
        query_points=query_points,
    )


def collate_fn_train(batch: List[Tuple[EtapData, bool]]) -> Tuple[EtapData, List[bool]]:
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]
    voxels = torch.stack([b.voxels for b, _ in batch], dim=0)
    rgbs = torch.stack([b.rgbs for b, _ in batch], dim=0) if batch[0][0].rgbs is not None else None
    trajectory = torch.stack([b.trajectory for b, _ in batch], dim=0)
    visibility = torch.stack([b.visibility for b, _ in batch], dim=0)
    valid = torch.stack([b.valid for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    return (
        EtapData(
            voxels=voxels,
            rgbs=rgbs,
            trajectory=trajectory,
            visibility=visibility,
            valid=valid,
            seq_name=seq_name,
        ),
        gotit,
    )


def collate_fn_load_inverted(batch: List[Tuple[EtapData, EtapData, bool]]) -> Tuple[EtapData, List[bool]]:
    """
    Collate function for video tracks data with load_inverted case.
    Combines original and inverted samples into a single EtapData object.
    """
    # Separate original and inverted samples
    orig_samples = [b[0] for b in batch]
    inv_samples = [b[1] for b in batch]
    gotit = [gotit for _, _, gotit in batch]

    # Combine original and inverted samples
    combined_samples = orig_samples + inv_samples

    voxels = torch.stack([b.voxels for b in combined_samples], dim=0)
    rgbs = torch.stack([b.rgbs for b in combined_samples], dim=0) if combined_samples[0].rgbs is not None else None
    trajectory = torch.stack([b.trajectory for b in combined_samples], dim=0)
    visibility = torch.stack([b.visibility for b in combined_samples], dim=0)
    valid = torch.stack([b.valid for b in combined_samples], dim=0)
    seq_name = [b.seq_name for b in combined_samples]

    # Create a tensor to keep track of paired samples using explicit indices
    batch_size = len(orig_samples)
    pair_indices = torch.arange(batch_size).repeat(2)

    return EtapData(
        voxels=voxels,
        rgbs=rgbs,
        trajectory=trajectory,
        visibility=visibility,
        valid=valid,
        seq_name=seq_name,
        pair_indices=pair_indices,
    ), gotit


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input tensor or other object

    Returns:
        t_cuda: `t` moved to a cuda device, if supported
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj: Any) -> Any:
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        obj: Input dataclass object to move to CUDA

    Returns:
        obj: The same object with its tensor fields moved to CUDA
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj
