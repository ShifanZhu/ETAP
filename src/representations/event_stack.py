from typing import Tuple

import numpy as np


class MixedDensityEventStack:
    """Create an mixed density event stack from events.
    Implementation inspired by https://github.com/yonseivnl/se-cff.

    Args:
        image_shape: (height, width)
        num_stacks: number of channels
        interpolation: interpolation method to use when building histograms
    """
    def __init__(self, image_shape: Tuple[int, int], num_stacks: int,
                 interpolation: str = 'nearest_neighbor', channel_overlap=False,
                 centered_channels=False) -> None:
        assert image_shape[0] > 0
        assert image_shape[1] > 0
        assert num_stacks > 0
        self.image_shape = image_shape
        self.num_stacks = num_stacks
        self.interpolation = interpolation
        self.channel_overlap = channel_overlap
        self.centered_channels = centered_channels
        assert self.interpolation in ['nearest_neighbor', 'bilinear']
        assert not self.centered_channels or (self.centered_channels and self.channel_overlap), "Centered channels require channel overlap"

    def __call__(self, events: np.ndarray, t_mid=None) -> np.ndarray:
        """Create mixed density event stack.

        Args:
            events: events: a NumPy array of size [n x d], where n is the number of events and d = 4.
                    Every event is encoded with 4 values (y, x, t, p).

        Returns:
            A mixed density event stack representation of the event data.
        """
        assert events.shape[1] == 4
        assert not self.centered_channels or t_mid is not None, "Centered channels require t_mid"
        stacked_event_list = []
        curr_num_events = len(events)

        for _ in range(self.num_stacks):
            if self.interpolation == 'nearest_neighbor':
                stacked_event = self.stack_data_nearest_neighbor(events)
            elif self.interpolation == 'bilinear':
                stacked_event = self.stack_data_bilinear(events)

            stacked_event_list.append(stacked_event)
            curr_num_events = curr_num_events // 2

            if self.centered_channels:
                i_mid = np.searchsorted(events[:, 2], t_mid)
                i_start = max(i_mid - curr_num_events // 2, 0)
                i_end = min(i_mid + curr_num_events // 2, len(events))
                events = events[i_start:i_end]
            else:
                events = events[curr_num_events:]

        if not self.channel_overlap:
            for stack_idx in range(self.num_stacks - 1):
                prev_stack_polarity = stacked_event_list[stack_idx]
                next_stack_polarity = stacked_event_list[stack_idx + 1]
                diff_stack_polarity = prev_stack_polarity - next_stack_polarity
                stacked_event_list[stack_idx] = diff_stack_polarity

        return np.stack(stacked_event_list, axis=0)

    def stack_data_nearest_neighbor(self, events):
        height, width = self.image_shape
        y = events[:, 0].astype(int)
        x = events[:, 1].astype(int)
        p = events[:, 3]
        p[p == 0] = -1

        stacked_polarity = np.zeros([height, width], dtype=np.int8)
        index = (y * width) + x
        stacked_polarity.put(index, p)

        return stacked_polarity

    def stack_data_bilinear(self, events):
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        weight = events[..., 3]
        weight[weight == 0] = -1

        h, w = self.image_shape
        nb = len(events)
        image = np.zeros((nb, h * w), dtype=np.float64)

        floor_xy = np.floor(events[..., :2] + 1e-8)
        floor_to_xy = events[..., :2] - floor_xy

        x1 = floor_xy[..., 1]
        y1 = floor_xy[..., 0]
        inds = np.concatenate(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            axis=-1,
        )
        inds_mask = np.concatenate(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )
        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = np.concatenate([w_pos0, w_pos1, w_pos2, w_pos3], axis=-1)
        inds = (inds * inds_mask).astype(np.int64)
        vals = vals * inds_mask
        for i in range(nb):
            np.add.at(image[i], inds[i], vals[i])
        return image.reshape((nb,) + self.image_shape).squeeze()
