#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging
from typing import List, Optional

import numpy as np
import torch

from torch_dag import structured_modules as smodules
from torch_dag.core.dag_module import InnerVertex
from torch_dag_algorithms.pruning.commons import PASS_THROUGH_CHANNELS_CLASSES, is_depthwise_conv
from torch_dag_algorithms.pruning.commons import is_source, is_conv_source
from torch_dag_timm_plugin.mask_propagation import compute_timm_module_channels_masks, CUSTOM_MASKS_TIMM_CLASSES

logger = logging.getLogger(__name__)


def check_if_vertex_is_empty(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
        channels_masks: List[torch.Tensor],
) -> bool:
    if channels_masks is None:
        return False
    """
    A vertex may become empty for various reasons:
    * all its predecessors are empty
    * one of its predecessors is empty and the node does not make sense without it, e.g. MatMal
    """
    if isinstance(vertex.module, smodules.ParameterModule):
        return False
    if channels_masks[0].sum() == 0:
        return True
    # all input masks are zeros
    input_masks = [e[0] for e in predecessors_channels_masks]
    if len(input_masks) > 0 and all([mask.sum() == 0 for mask in input_masks]):
        return True
    if isinstance(vertex.module, smodules.TfMatmulModule):
        if any([mask.sum() == 0 for mask in input_masks]):
            return True
    return False


def masks_are_equal(x: torch.Tensor, y: torch.Tensor):
    return (x == y).sum()


def get_not_none_masks(predecessors_channels_masks: List[List[torch.Tensor]]):
    return [mask_list for mask_list in predecessors_channels_masks if mask_list is not None]


def check_if_any_mask_is_zeros(predecessors_channels_masks: List[List[torch.Tensor]]):
    for predecessor_masks in predecessors_channels_masks:
        for mask in predecessor_masks:
            if mask.sum() == 0:
                return True
    return False


def compute_matmul_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
):
    """
    NOTE this method is implemented strictly for usage of `MatMulNode` in attention-based settings.
    We only assume two types of behavior here:
    * transpose is `True` - we are mat-multiplying two tensors `x` and `y` coming from `query` and `key`
    dense nodes, respectively. Here the whole idea of masks does not really make sense as we have no channels. In a hacky
    twist we propage a list [1, 1, ..., 1] of length equal to the number of tokens.
    * transpose is `False - we are mat-multiplying the attention scores `x` of shape (B, T, T) (where
    B - batch size, T - num of tokens) and output `y` for `value` dense node. Here we just pass on the channel masks from
    `value`.
    """
    x_mask, y_mask = predecessors_channels_masks[0][0], predecessors_channels_masks[1][0]
    if x_mask.sum() == 0 or y_mask.sum() == 0:
        return [torch.zeros(size=(predecessors_channels_masks[0][0].shape[0],), dtype=torch.int32)]
    if vertex.module.transpose:
        return [torch.ones(size=(predecessors_channels_masks[0][0].shape[0],), dtype=torch.int32)]
    else:
        return [predecessors_channels_masks[1][0]]


def compute_reshape_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    some_masks_are_zeros = check_if_any_mask_is_zeros(predecessors_channels_masks)
    if some_masks_are_zeros:
        return [torch.zeros(size=(1,), dtype=torch.int32)]
    return None


def compute_subixel_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    vertex_output = vertex.dag_module.forward_dict[vertex]
    num_channels = vertex_output.shape[1]
    return [torch.ones(size=(num_channels,), dtype=torch.int32)]


def compute_elementwise_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    There are a couple of possible scenarios here:
    1. All the masks are the same -> propagate the first one
    2. The masks that are non-fully-zero differ -> raise ValueError
    3. Some masks are fully-zero, but the ones that are not are equal -> propagate the first non-zero mask
    When some input masks are de-facto scalars then we need to do broadcasting
    """
    if any([e is None for e in predecessors_channels_masks]):
        return None
    non_scalar_input_masks = [e[0] for e in predecessors_channels_masks if len(e[0]) > 1]
    if len(non_scalar_input_masks) == 0:
        return None
    max_mask = non_scalar_input_masks[0]
    for mask in non_scalar_input_masks[1:]:
        max_mask = torch.max(mask, max_mask)
    return [max_mask]


def compute_chunk_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
):
    return torch.chunk(predecessors_channels_masks[0][0], chunks=vertex.module.chunks, dim=0)


def compute_merger_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
):
    return [predecessors_channels_masks[k][0] for k in range(len(vertex.predecessors))]


def compute_extractor_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
):
    try:
        return [predecessors_channels_masks[0][vertex.module.index]]
    except IndexError:
        return None


def compute_channels_masks(
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
        output_channels_masks: Optional[List[torch.Tensor]] = None,
):
    non_trivial_predecessors_channels_masks = get_not_none_masks(predecessors_channels_masks)
    if output_channels_masks is not None:
        result = output_channels_masks

    elif len(non_trivial_predecessors_channels_masks) == 0:
        result = None
    elif isinstance(vertex.module, PASS_THROUGH_CHANNELS_CLASSES):
        result = [non_trivial_predecessors_channels_masks[0][0]]
    elif is_depthwise_conv(vertex.module):
        result = [non_trivial_predecessors_channels_masks[0][0]]
    elif is_source(vertex.module):
        vertex_output = vertex.dag_module.forward_dict[vertex]
        # A source (conv or dense) but not within the scope of any orbit - we return zero mask (all zeros)
        # if the predecessor has all-zero mask, otherwise we return full mask
        channels_dim = 1 if is_conv_source(vertex.module) else - 1
        if non_trivial_predecessors_channels_masks[0][0].sum() == 0:
            result = [torch.zeros(size=(vertex_output.shape[channels_dim],), dtype=torch.int32)]
        else:
            result = [torch.ones(size=(vertex_output.shape[channels_dim],), dtype=torch.int32)]
    elif isinstance(vertex.module, smodules.ConcatModule):
        vertex_output = vertex.dag_module.forward_dict[vertex]
        output_rank = len(vertex_output.shape)
        # We just concatenate the masks from predecessors and pass them on as long as concat
        # happens on the channel dim
        concat_masks = False
        if output_rank == 4 and vertex.module.dim == 1:
            concat_masks = True
        if output_rank == 3 and vertex.module.dim in (-1, 2):
            concat_masks = True
        if concat_masks:
            result = [torch.cat([pd[0] for pd in non_trivial_predecessors_channels_masks], dim=0)]
        else:
            result = [non_trivial_predecessors_channels_masks[0][0]]

    elif isinstance(vertex.module, (smodules.AddModule, smodules.SubModule, smodules.MulModule)):
        result = compute_elementwise_channels_masks(
            vertex=vertex,
            predecessors_channels_masks=predecessors_channels_masks,
        )
    elif isinstance(vertex.module, smodules.ParameterModule):
        if len(vertex.module.param.shape) == 3:
            return [torch.ones(size=(vertex.module.param.shape[2],), dtype=torch.int32)]
        elif len(vertex.module.param.shape) == 1:
            return [torch.ones(size=(vertex.module.param.shape[0],), dtype=torch.int32)]
        else:
            raise NotImplementedError
    elif isinstance(vertex.module, smodules.TfTokenizeModule):
        return [torch.ones(size=(vertex.dag_module.forward_dict[vertex].shape[2],), dtype=torch.int32)]
    elif isinstance(vertex.module, smodules.TfDetokenizeModule):
        return [torch.ones(size=(vertex.dag_module.forward_dict[vertex].shape[1],), dtype=torch.int32)]
    elif isinstance(vertex.module, (smodules.SpaceToDepthModule, smodules.DepthToSpaceModule)):
        return [torch.ones(size=(vertex.dag_module.forward_dict[vertex].shape[1],), dtype=torch.int32)]

    # timm modules
    elif isinstance(vertex.module, CUSTOM_MASKS_TIMM_CLASSES):
        return compute_timm_module_channels_masks(vertex.module, vertex, predecessors_channels_masks,
                                                  output_channels_masks)
    elif isinstance(vertex.module, smodules.TfMatmulModule):
        result = compute_matmul_channels_masks(
            vertex=vertex,
            predecessors_channels_masks=non_trivial_predecessors_channels_masks,
        )
    elif isinstance(vertex.module, smodules.ChunkModule):
        result = compute_chunk_channels_masks(
            vertex=vertex,
            predecessors_channels_masks=non_trivial_predecessors_channels_masks,
        )
    elif isinstance(vertex.module, (smodules.ReshapeModule, smodules.ReshapeModuleV2, smodules.ReshapeWithSpecModule)):
        result = compute_reshape_channels_masks(
            vertex=vertex,
            predecessors_channels_masks=non_trivial_predecessors_channels_masks,
        )
    elif isinstance(vertex.module, smodules.TensorMergerModule):
        result = compute_merger_channels_masks(
            vertex=vertex,
            predecessors_channels_masks=non_trivial_predecessors_channels_masks,
        )
    elif isinstance(vertex.module, smodules.TensorExtractorModule):
        result = compute_extractor_channels_masks(
            vertex=vertex,
            predecessors_channels_masks=non_trivial_predecessors_channels_masks,
        )
    elif isinstance(vertex.module, smodules.AuxiliaryTokenModule):
        result = [torch.ones(size=(vertex.module.token.shape[0],), dtype=torch.int32)]
    elif isinstance(vertex.module, smodules.SliceModule):
        input_rank = len(vertex.dag_module.forward_dict[vertex.predecessors[0]].shape)
        slice_spec: np.s_ = vertex.module.slice_spec
        # there are two cases we cover here (B, C, H, W) input shape and (B, T, dim) input shape
        if input_rank == 3:
            if len(slice_spec) < 3:  # no slicing ind the channel dimension
                channel_spec = np.s_[:]
            else:
                channel_spec = slice_spec[2]
        elif input_rank == 4:
            channel_spec = slice_spec[1]
        else:
            raise NotImplementedError
        # TODO: this does not work in general.
        # There is no way of knowing if slice is being performed
        # on channel dimension

        result = [non_trivial_predecessors_channels_masks[0][0][channel_spec]]
    elif isinstance(vertex.module, smodules.SplitModule):
        input_mask = non_trivial_predecessors_channels_masks[0][0]
        split_size_or_sections = vertex.module.split_size_or_sections
        result = list(torch.split(input_mask, split_size_or_sections=split_size_or_sections, dim=0))

    else:
        logger.info(f'No explicit mask propagation for vertex: {vertex}, of type {type(vertex.module)}. '
                    f'Returning `None` masks.')
        return None
    return result
