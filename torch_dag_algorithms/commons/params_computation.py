#
# Copyright © TCL Research Europe. All rights reserved.
#
from typing import List

import torch

from torch_dag_algorithms.pruning.module_multipliers import compute_multipliers
from torch_dag.commons.mixins import CustomCountParamsMixin
from torch_dag.core import dag_module


def build_full_params_list(
        dag: dag_module.DagModule,
        normalize: bool = True,
) -> List:
    """
    Builds a list of parameters of all vertices in the DAG model.

    :param dag: DagModule instance.
    :param normalize: If True, the list of parameters is scaled by 1e-6.
    :return: List of parameters of all vertices in the DAG model.
    """
    if dag.training:
        dag.eval()
    dag_params = []
    for k, vertex in enumerate(dag.inner_vertices):
        if isinstance(vertex.module, CustomCountParamsMixin):
            vertex_params = vertex.module.count_params()
        else:
            vertex_params = sum(p.numel() for p in vertex.module.parameters() if p.requires_grad)
        dag_params += [vertex_params]
    if dag.training:
        dag.train()
    if normalize:
        dag_params = [vertex_params * 1e-6 for vertex_params in dag_params]
    return dag_params


def compute_params(
        dag: dag_module.DagModule,
        full_params_list: List[float],
) -> torch.Tensor:
    if dag.forward_dict is None:
        raise AssertionError(
            f'To run dynamic parameters computation one needs to set `dag.cache_forward_dict = True`. '
            f'Moreover, `dag.forward_dict` must not be None. In other words this method '
            f'can only br called after a forward pass before which we set'
            f' `dag.cache_forward_dict = True`',
        )
    if not dag.flat:
        raise AssertionError(f'Dynamic parameters computation can only be done for flat DagModule instances.')
    multipliers = compute_multipliers(dag)
    return torch.stack([m * f for m, f in zip(multipliers, full_params_list)], dim=0).sum()


def params_to_megabytes(
        full_params_list: List[float],
        bits: int = 32,
) -> List[float]:
    """
    Converts a list of parameters to megabytes.

    NOTE: when using this method, make sure that the list of parameters is not normalized.

    :param full_params_list: List of parameters of vertices.
    :param bits: Number of bits per parameter.
    :return: List of parameters of vertices in megabytes.
    """
    return [vertex_params * bits / 1024 ** 2 / 8 for vertex_params in full_params_list]
