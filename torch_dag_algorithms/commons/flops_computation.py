import torch
from torch_dag.core import dag_module
from torch_dag_algorithms.pruning.module_multipliers import compute_multipliers


from typing import List, Tuple


def compute_kmapp(
        dag: dag_module.DagModule,
        input_shape_without_batch: Tuple[int, ...],
        full_flops_list: List[int],
) -> torch.Tensor:
    if dag.forward_dict is None:
        raise AssertionError(f'To run dynamic FLOPs computation one needs to set `dag.cache_forward_dict = True`. '
                             f'Moreover, `dag.forward_dict` must not be None. In other words this method '
                             f'can only br called after a forward pass before which we set'
                             f' `dag.cache_forward_dict = True`')
    if not dag.flat:
        raise AssertionError(f'Dynamic kmapp computation can only be done for flat DagModule instances.')
    if len(input_shape_without_batch) != 3:
        raise NotImplementedError(f'kmapp computation makes sense only for (B, C, H, W) input shape. '
                                  f'Received input shape: {input_shape_without_batch}.')
    multipliers = compute_multipliers(dag)
    normalization = input_shape_without_batch[1] * input_shape_without_batch[2] * 1e3
    return torch.stack([m * f for m, f in zip(multipliers, full_flops_list)], dim=0).sum() / normalization