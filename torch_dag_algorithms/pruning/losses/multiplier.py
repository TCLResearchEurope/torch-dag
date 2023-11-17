import torch


def anneal(
        global_step: torch.int64,
        decay_steps: int,
        decay_rate: float = 0.1,
) -> torch.Tensor:
    return 1.0 - decay_rate ** (global_step / decay_steps)
