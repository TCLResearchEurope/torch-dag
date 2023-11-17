#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging
from typing import Tuple

import torch

from torch_dag import structured_modules as smodules
from torch_dag.core import dag_module_utils
from torch_dag.core.dag_module import DagModule
from torch_dag_algorithms.pruning import mask_propagation
from torch_dag_algorithms.pruning.channel_removal_interface import remove_module_channels
from torch_dag_algorithms.pruning.commons import is_source, get_orbits_dict

logger = logging.getLogger(__name__)


def run_local_distill(dag: DagModule):
    orbits_dict = get_orbits_dict(dag)
    if dag.forward_dict is None:
        raise AssertionError(f'To run channel removal algorithm one needs to set `dag.cache_forward_dict = True`')
    channels_masks_list = [[torch.ones(size=(dag.forward_dict[input_vertex].shape[1],), dtype=torch.int32)]
                           for input_vertex in dag.input_vertices]
    for vertex in dag.inner_vertices:
        pd_indices = dag._get_inner_vertex_predecessor_indices(vertex)
        predecessors_channels_masks = mask_propagation.get_not_none_masks([channels_masks_list[k] for k in pd_indices])
        if vertex.orbit is not None and is_source(vertex.module):
            orbit = orbits_dict[vertex.orbit]
            output_channels_masks = orbit.compute_output_channel_masks(
                predecessors_channel_masks=predecessors_channels_masks)
        else:
            output_channels_masks = None

        channels_masks = mask_propagation.compute_channels_masks(
            vertex=vertex,
            predecessors_channels_masks=predecessors_channels_masks,
            output_channels_masks=output_channels_masks,
        )
        vertex_is_empty = mask_propagation.check_if_vertex_is_empty(
            vertex=vertex,
            predecessors_channels_masks=predecessors_channels_masks,
            channels_masks=channels_masks,
        )
        if vertex_is_empty:
            logger.info(f'The vertex: {vertex} is being removed...')
            vertex.module = smodules.EmptyModule()
            channels_masks = [torch.zeros_like(mask) for mask in channels_masks]

        else:
            vertex.module = remove_module_channels(
                vertex.module,
                vertex,
                predecessors_channels_masks,
                channels_masks,
            )
        channels_masks_list.append(channels_masks)


def remove_channels_in_dag(
        dag: DagModule,
        input_shape: Tuple[int, ...],
) -> DagModule:
    dag.clear_custom_buffers()
    is_caching = dag.cache_forward_dict
    if not is_caching:
        dag.cache_forward_dict = True
    """
    A function to remove channels from an orbitalized model after pruning training phase.
    :param input_shape: A tuple (for example (B, C, H, W)) describing the shape of the input to model.
    :return: A model in which channels have been physically removed.
    """
    if not dag.flat:
        raise AssertionError(f'The dag must be flat!')
    dag.eval()
    x = torch.ones(size=input_shape)
    _ = dag(x)
    run_local_distill(dag)
    # do some additional cleanup

    dag_module_utils.remove_redundant_vertices_from_flat_dag(dag)
    if not is_caching:
        dag.clear_tensor_dicts()
        dag.cache_forward_dict = False
    dag.clear_custom_buffers()
    for iv in dag.inner_vertices:
        iv.orbit = None
    return dag
