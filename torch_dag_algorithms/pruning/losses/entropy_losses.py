#
# Copyright © TCL Research Europe. All rights reserved.
#

import torch

# from torch_dag.core.dag_module import DagModule
from torch_dag_algorithms.pruning.commons import get_orbits_dict
from torch_dag_algorithms.pruning.losses.multiplier import anneal


def entropy_loss(
        dag,#: DagModule,
        decay_steps: int,
        global_step: torch.int64,
        lmbda: float = 0.1,
        epsilon: float = 0.01,
        decay_rate: float = 0.1,
        annealing: bool = True,
) -> torch.Tensor:
    orbits_dict = get_orbits_dict(dag)
    orbit_modules = set(orbits_dict.values())
    result = []
    if annealing:
        multiplier = anneal(
            global_step=global_step,
            decay_rate=decay_rate,
            decay_steps=decay_steps
        )
    else:
        multiplier = 1.0
    if len(orbit_modules) > 0:
        for orb in orbit_modules:
            logits = orb.logits
            probs_ = torch.sigmoid(logits)[..., None]
            probs = torch.cat([probs_, 1. - probs_], axis=1)
            loss = torch.maximum(- (probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon))
            result.append(loss)

        return lmbda * multiplier * torch.stack(result, dim=0).mean()
    return torch.tensor(0.0)
