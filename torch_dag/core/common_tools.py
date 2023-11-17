import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def per_channel_noise_to_signal_ratio(
        x: torch.Tensor,
        y: torch.Tensor,
        non_channel_dim: Tuple[int, ...] = (0, 2, 3),
        epsilon: float = 1e-3,

) -> torch.Tensor:
    y_per_channel_variance = torch.square(torch.std(y, dim=non_channel_dim))
    per_channel_squared_difference = torch.square((x - y)).mean(dim=non_channel_dim)

    return torch.divide(per_channel_squared_difference, y_per_channel_variance + epsilon).mean()
