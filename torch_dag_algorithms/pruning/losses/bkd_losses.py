#
# Copyright © TCL Research Europe. All rights reserved.
#
import torch

# from torch_dag.core.dag_module import DagModule
from torch_dag_algorithms.pruning.commons import get_orbits_dict

# TODO: find better way to avoid circular dependency
def bkd_loss(
        dag,#: DagModule,
        lmbda: float = 0.001,
) -> torch.Tensor:
    orbits_dict = get_orbits_dict(dag)
    orbit_modules = set(orbits_dict.values())
    result = 0.0
    if len(orbit_modules) > 0:
        for orb in orbit_modules:
            losses_per_orbit = list(orb.bkd_masking_losses.values())
            loss_per_orbit = torch.stack(losses_per_orbit, dim=0).sum()
            result += loss_per_orbit

        return lmbda * result / len(orbit_modules)
    return torch.tensor(0.0)
