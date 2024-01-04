#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging
from typing import List, Tuple, Dict, Union

import torch

from torch_dag import structured_modules as smodules
from torch_dag.core.dag_module import DagModule
from torch_dag.core.dag_module import InputVertex, InnerVertex, Vertex
from torch_dag_algorithms.pruning.commons import PASS_THROUGH_CHANNELS_CLASSES
from torch_dag_algorithms.pruning.commons import is_source, get_orbits_dict, is_linear_source, is_depthwise_conv
from torch_dag_algorithms.pruning.modules import OrbitModule
from torch_dag_timm_plugin.module_multipliers import compute_timm_average_num_channels, \
    CUSTOM_AVERAGE_CHANNELS_TIMM_CLASSES

logger = logging.getLogger(__name__)

PASS_THROUGH_MULTIPLIER_CLASSES = PASS_THROUGH_CHANNELS_CLASSES


def shape_to_float(shape, device, dim=1):
    return torch.tensor(shape[dim], device=device).to(torch.float32)


def compute_elementwise_op_average_channels(average_number_input_channels: List[List[torch.Tensor]], ):
    average_number_input_channels = [e for e in average_number_input_channels if e is not None]
    if len(average_number_input_channels) == 0:
        return None
    return [torch.max(torch.stack([e[0] for e in average_number_input_channels]))]


def compute_average_num_channels(
        vertex: InnerVertex,
        average_number_input_channels: List[List[torch.Tensor]],
        orbits_dict: Dict[str, OrbitModule],
        forward_dict: Dict[Vertex, Union[torch.Tensor, List[torch.Tensor]]]
) -> Union[List[torch.Tensor], None]:
    device = forward_dict[vertex.dag_module.input_vertices[0]].device
    if isinstance(vertex.module, PASS_THROUGH_MULTIPLIER_CLASSES):
        return [average_number_input_channels[0][0]]

    if is_source(vertex.module):
        if vertex.orbit is not None:
            orbit_module = orbits_dict[vertex.orbit]
            return [orbit_module.compute_average_number_of_output_channels()]
        else:
            if is_linear_source(vertex.module):
                return [shape_to_float(forward_dict[vertex].shape, dim=-1, device=device)]
            else:
                return [shape_to_float(forward_dict[vertex].shape, device=device)]
    elif is_depthwise_conv(vertex.module):
        return [average_number_input_channels[0][0]]
    elif isinstance(vertex.module, (smodules.AddModule, smodules.SubModule, smodules.MulModule)):
        return compute_elementwise_op_average_channels(average_number_input_channels)

    elif isinstance(vertex.module, smodules.ConcatModule):
        return [torch.stack([x[0] for x in average_number_input_channels]).sum()]

    elif isinstance(vertex.module, smodules.ChunkModule):
        assert vertex.module.dim == 1
        channels = average_number_input_channels[0][0]
        return [channels / vertex.module.chunks for _ in range(vertex.module.chunks)]
    elif isinstance(vertex.module, smodules.ParameterModule):
        # the heuristic here is that the channel dim will be the axis with max shape
        max_shape = max(forward_dict[vertex].shape)
        return [torch.tensor(max_shape, device=device).to(torch.float32)]
    elif isinstance(vertex.module, smodules.TfTokenizeModule):
        return [shape_to_float(forward_dict[vertex].shape, dim=2, device=device)]
    elif isinstance(vertex.module, smodules.TfDetokenizeModule):
        return [shape_to_float(forward_dict[vertex].shape, dim=1, device=device)]
    elif isinstance(vertex.module, CUSTOM_AVERAGE_CHANNELS_TIMM_CLASSES):
        return compute_timm_average_num_channels(vertex.module, vertex, average_number_input_channels, orbits_dict,
                                                 forward_dict)
        return [shape_to_float(forward_dict[vertex].shape, dim=2, device=device)]
    elif isinstance(vertex.module, smodules.SliceModule):
        # TODO: what in the case slice is performed on a dim different than 1?
        return [shape_to_float(forward_dict[vertex].shape, dim=1, device=device)]
    elif isinstance(vertex.module, smodules.AuxiliaryTokenModule):
        # TODO: what in the case slice is performed on a dim different than 1?
        return [shape_to_float(forward_dict[vertex].shape, dim=-1, device=device)]
    else:
        return None
        # raise NotImplementedError(f'Averag channel computation not implemented for module of type: '
        #                           f'{vertex.module.__class__}')


def compute_full_num_channels(
        vertex: InnerVertex,
        device,
) -> List[torch.Tensor]:
    """
    Computes full number of channels (no pruning) for a given vertex.
    """
    output_tensors = vertex.dag_module.forward_dict[vertex]
    # validate input channels -> we only handle tensors and lists of tensors otherwise a default value
    # of 1.0 is returned
    if not isinstance(output_tensors, (torch.Tensor, List)):
        return [torch.tensor(1.0, device=device)]
    if isinstance(output_tensors, List) and any([not isinstance(t, torch.Tensor) for t in output_tensors]):
        return [torch.tensor(1.0, device=device)]
    if not isinstance(output_tensors, List):
        output_tensors = [output_tensors]
    # If output_tensor shape has rank 4 then the channel dim is 1. If it has rank 3,
    # then the channel dim equals 2.
    if len(output_tensors[0].shape) == 4:
        if is_linear_source(vertex.module):
            channel_dim = -1
        else:
            channel_dim = 1
    elif len(output_tensors[0].shape) in (2, 3):
        channel_dim = -1
    elif len(output_tensors[0].shape) in (0, 1):
        channel_dim = 0
    else:
        raise NotImplementedError

    tensor_shapes = []
    for t in output_tensors:
        if len(t.shape) > 0:
            tensor_shapes.append(t.shape[channel_dim])
        else:
            tensor_shapes.append(1)
    return [torch.tensor(ts, device=device) for ts in tensor_shapes]


def compute_average_channels(
        dag: DagModule,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """
    * we implicitly assume that the channle dim in input to the `dag` is 1
    """
    orbits_dict = get_orbits_dict(dag)
    input_tensors = [dag.forward_dict[vertex] for vertex in dag.input_vertices]
    device = input_tensors[0].device
    # compute channels for input cell nodes first
    average_channels_list = [
        [torch.tensor(tensor.shape[1], device=device).to(torch.float32)] for tensor in input_tensors]
    full_channels_list = [
        [torch.tensor(tensor.shape[1], device=device).to(torch.float32)] for tensor in input_tensors]

    for inner_vertex in dag.inner_vertices:
        pd_indices = dag._get_inner_vertex_predecessor_indices(inner_vertex)
        average_number_input_channels = [average_channels_list[k] for k in pd_indices]
        average_number_channels = compute_average_num_channels(
            vertex=inner_vertex,
            average_number_input_channels=average_number_input_channels,
            orbits_dict=orbits_dict,
            forward_dict=dag.forward_dict,
        )
        # TODO can we pass the information below in a safer way?
        full_number_channels = compute_full_num_channels(
            vertex=inner_vertex,
            device=device,
        )
        if average_number_channels is None:  # trivial 1.0 multiplier will be computed
            average_number_channels = full_number_channels

        average_channels_list.append(average_number_channels)
        full_channels_list.append(full_number_channels)

    return average_channels_list[len(dag.input_vertices):], full_channels_list[len(dag.input_vertices):]


def compute_source_left_multiplier(
        vertex: InnerVertex,
        right_multipliers: List[torch.Tensor],
) -> torch.Tensor:
    predecessor = vertex.predecessors[0]
    left_multiplier = torch.tensor(1.0, device=vertex.dag_module.forward_dict[vertex].device)
    # adjusting for the fact that for sources the multiplier will depend also on the predecessor
    # multiplier
    if isinstance(predecessor, InnerVertex):
        predecessor_index = vertex.dag_module.inner_vertices.index(predecessor)
        left_multiplier = right_multipliers[predecessor_index]
    return left_multiplier


def compute_matmul_multiplier(
        vertex: InnerVertex,
        average_channels_list: List[List[torch.Tensor]],
        full_channels_list: List[List[torch.Tensor]],
):
    """
    WARNING: this only makes sense for the use of `MatMulNode` in attention based scenarios!
    *   If `transpose` is True, the multplier will be the same as for the query predecessor, since the number of
        multiadds will be proportional to T * T * dim, where T - num of tokens, dim - num of channels.
    *   If `transpose` is False, the multplier will be the same as for the value predecessor, since the number of
        multiadds will be proportional to T * T * dim, where T - num of tokens, dim - num of channels.
    """
    transpose = vertex.module.transpose
    if transpose:
        query_predecessor = vertex.predecessors[0]
        query_predecessor_index = vertex.dag_module.inner_vertices.index(query_predecessor)
        query_multiplier = average_channels_list[query_predecessor_index][0] / \
                           full_channels_list[query_predecessor_index][0]
        return query_multiplier
    else:
        value_predecessor = vertex.predecessors[1]
        value_predecessor_index = vertex.dag_module.inner_vertices.index(value_predecessor)
        value_multiplier = average_channels_list[value_predecessor_index][0] / \
                           full_channels_list[value_predecessor_index][0]
        return value_multiplier


def compute_right_multipliers(
        dag: DagModule,
        average_channels_list: List[List[torch.Tensor]],
        full_channels_list: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    A method to compute `right mutlipliers'. A `right multiplier` of a module is a number in [0.0,1.0] which
    designetes the proportion of output channels of the module that will be kept. For nodes like `Conv2D` and `Linear`,
    there will be separate `right` and `left` multipliers. For almost all the other nodes the average proportion
    of channels that will be kept is inherited from their predecessor (for example `BarchNorm`). The bulk of the
    implementation is actually in the `compute_average_channels` method.
    """
    right_multipliers = []
    for k, vertex in enumerate(dag.inner_vertices):
        if isinstance(vertex.module, PASS_THROUGH_CHANNELS_CLASSES):
            if isinstance(vertex.predecessors[0], InputVertex):
                right_multiplier = torch.tensor(1.0, device=average_channels_list[k][0].device)
            else:
                pd_index = dag.inner_vertices.index(vertex.predecessors[0])
                right_multiplier = right_multipliers[pd_index]
        elif vertex.orbit is None:
            right_multiplier = torch.tensor(1.0, device=average_channels_list[k][0].device)
        else:
            right_multiplier = average_channels_list[k][0] / full_channels_list[k][
                0]  # linearly proportional to the number of channels
        if isinstance(vertex.module, smodules.TfMatmulModule):
            # fancy behaviour here needed for attention based pruning
            right_multiplier = compute_matmul_multiplier(
                vertex=vertex,
                average_channels_list=average_channels_list,
                full_channels_list=full_channels_list,
            )

        right_multipliers.append(right_multiplier)
    return right_multipliers


def compute_multipliers(
        dag: DagModule,
) -> List[torch.Tensor]:
    """
    A function to compute 'multipliers' for each vertex. These are numbers in [0.0, 1.0] that
    are computed based on orbits and propagated through the `dag`. They are necessary to compute (differentiable)
    FLOPs number given the full FLOPs list. In essence, for each node we compute its full `fvcore` and then,
    based on the specific characteristics of the vertex we compute a `multiplier` which will tell us what proportion
    of the node's FLOPs will remain after pruning.
    We introduce here the concepts of `right multiplier` and `left mutliplier`. For most vertices we only need the
    `right multiplier`, i.e., the proportion of output channels that wil be kept. For some nodes, like `Conv2D` and
    `Linear` we also need a `left multplier`, because their number of FLOPs after removing channels will also depend on
    the average number of input channels they receive.
    :return: `multipliers` a List of float32 tensors that are differentiable with respect to orbit
    variables.
    """
    if dag.forward_dict is None:
        raise AssertionError(f'To run multiplier computation one needs to set `dag.cache_forward_dict = True`')
    average_channels_list, full_channels_list = compute_average_channels(dag)
    right_multipliers = compute_right_multipliers(
        dag=dag,
        average_channels_list=average_channels_list,
        full_channels_list=full_channels_list,
    )
    multipliers = []
    for k, vertex in enumerate(dag.inner_vertices):
        multiplier = right_multipliers[k].clone()
        if is_source(vertex.module):
            # adjusting for the fact that for conv and dense nodes the multiplier will depend also on the predecessor
            # multiplier
            left_multiplier = compute_source_left_multiplier(
                vertex=vertex,
                right_multipliers=right_multipliers,
            )
            multiplier *= left_multiplier

        if multiplier > 1.0:
            raise AssertionError(
                f'The multiplier computed for {vertex} of type {type(vertex.module)} is greater than 1.0. ')

        multipliers.append(multiplier)
    return multipliers
