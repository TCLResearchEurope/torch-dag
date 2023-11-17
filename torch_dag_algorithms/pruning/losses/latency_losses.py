#
# Copyright © TCL Research Europe. All rights reserved.
#

import torch

from torch_dag_algorithms.pruning.losses.multiplier import anneal


def latency_loss(
        computed_kmapp: torch.Tensor,
        target_kmapp: torch.Tensor,
        decay_steps: int,
        global_step: torch.int64,
        lmbda: float = 0.1,
        decay_rate: float = 0.1,
        annealing: bool = True,
) -> torch.Tensor:
    if annealing:
        multiplier = anneal(
            global_step=global_step,
            decay_rate=decay_rate,
            decay_steps=decay_steps
        )
    else:
        multiplier = 1.0

    return lmbda * multiplier * torch.relu(torch.divide(computed_kmapp, target_kmapp) - 1.)
