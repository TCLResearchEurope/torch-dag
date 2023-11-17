import logging
from functools import singledispatch
from typing import List, Union

import torch
from torch import nn

from torch_dag.core.dag_module import InnerVertex
from torch_dag_algorithms.pruning.channel_removal_primitives import (remove_module_channels_base,
                                                                     input_masks_are_empty, \
                                                                     get_full_input_masks)

logger = logging.getLogger(__name__)

CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ = ()


@singledispatch
def remove_timm_module_channels(
        module: nn.Module,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    raise NotImplementedError(f'Channel removal not implemented for module class: {module.__class__}')


try:
    from timm.layers.activations import GELU
    from timm.layers.drop import DropPath
    from timm.models.vision_transformer import PatchEmbed

    trivial_behavior_timm_modules = (
        GELU,
        PatchEmbed,
        DropPath,
    )
    CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ += trivial_behavior_timm_modules


    @remove_timm_module_channels.register(GELU)
    @remove_timm_module_channels.register(DropPath)
    @remove_timm_module_channels.register(PatchEmbed)
    def _(
            module: Union[GELU, DropPath, PatchEmbed],
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        return module

except ImportError:
    pass

try:
    from timm.layers.grn import GlobalResponseNorm

    CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ += (GlobalResponseNorm,)


    @remove_timm_module_channels.register
    def _(
            module: GlobalResponseNorm,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        if input_masks_are_empty(predecessors_masks):
            predecessors_masks = get_full_input_masks(module.weight.shape[0], module.weight.device)
        input_mask = predecessors_masks[0][0]
        indices = torch.where(input_mask == 1)[0]
        new_dim = len(indices)
        channels_last = (module.channel_dim == -1)
        new_module = GlobalResponseNorm(
            dim=new_dim,
            eps=module.eps,
            channels_last=channels_last,
        )
        new_weight = torch.take_along_dim(module.weight, dim=0, indices=indices)
        new_bias = torch.take_along_dim(module.bias, dim=0, indices=indices)
        new_module.weight.data = new_weight
        new_module.bias.data = new_bias
        return new_module
except ImportError:
    pass

try:
    from timm.models.vision_transformer import LayerScale

    CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ += (LayerScale,)


    @remove_timm_module_channels.register
    def _(
            module: LayerScale,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        if input_masks_are_empty(predecessors_masks):
            predecessors_masks = get_full_input_masks(module.gamma.shape[0], module.gamma.device)
        input_mask = predecessors_masks[0][0]
        indices = torch.where(input_mask == 1)[0]
        # remove in channels
        new_gamma = torch.take_along_dim(module.gamma, dim=0, indices=indices)
        module.gamma.data = new_gamma
        return module
except ImportError:
    pass

try:
    from timm.models.efficientformer_v2 import ConvNorm

    CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ += (ConvNorm,)


    @remove_timm_module_channels.register
    def _(
            module: ConvNorm,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        if input_masks_are_empty(predecessors_masks):
            predecessors_masks = get_full_input_masks(module.conv.in_channels, module.conv.weight.device)
        output_masks = [torch.ones(size=(module.conv.out_channels,), dtype=torch.int32)]
        module.conv = remove_module_channels_base(
            module.conv, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)

        module.bn = remove_module_channels_base(
            module.bn, vertex=vertex, predecessors_masks=output_masks, output_masks=output_masks)
        return module
except ImportError:
    pass

try:
    from timm.models.efficientformer_v2 import Attention2d, Attention2dDownsample

    CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ += (Attention2d, Attention2dDownsample)


    @remove_timm_module_channels.register
    def _(
            module: Attention2d,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        if input_masks_are_empty(predecessors_masks):
            predecessors_masks = get_full_input_masks(module.proj.conv.out_channels, module.proj.conv.weight.device)
        mask = predecessors_masks[0][0]
        indices = torch.where(mask == 1)[0]
        if len(indices) == len(mask):
            return module

        if module.stride_conv is not None:
            conv: nn.Conv2d = module.stride_conv.conv
            assert conv.groups == len(mask)

            new_weight = torch.take_along_dim(conv.weight, indices=indices.view(-1, 1, 1, 1), dim=0)
            conv.weight.data = new_weight
            if conv.bias is not None:
                new_bias = torch.take_along_dim(conv.bias, indices=indices, dim=0)
                conv.bias.data = new_bias

            new_bn = remove_module_channels_base(
                module.stride_conv.bn, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)
            module.stride_conv.bn = new_bn

            q_conv = module.q
            k_conv = module.k
            v_conv = module.v

            module.q = remove_module_channels_base(
                q_conv, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)

            module.k = remove_module_channels_base(
                k_conv, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)

            module.v = remove_module_channels_base(
                v_conv, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)



        else:
            module.q = remove_module_channels_base(
                module.q, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)

            module.k = remove_module_channels_base(
                module.k, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)

            module.v = remove_module_channels_base(
                module.v, vertex=vertex, predecessors_masks=predecessors_masks, output_masks=output_masks)

        return module


    @remove_timm_module_channels.register
    def _(
            module: Attention2dDownsample,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        if input_masks_are_empty(predecessors_masks):
            predecessors_masks = get_full_input_masks(module.k.conv.in_channels, module.k.conv.weight.device)
        mask = predecessors_masks[0][0]
        indices = torch.where(mask == 1)[0]
        if len(indices) == len(mask):
            return module
        q_local_output_masks = [torch.ones(module.q.local.out_channels)]
        module.q.local = remove_module_channels_base(
            module.q.local, predecessors_masks=predecessors_masks, output_masks=q_local_output_masks, vertex=vertex)

        module.k = remove_module_channels_base(
            module.k, predecessors_masks=predecessors_masks, output_masks=output_masks, vertex=vertex)
        module.v = remove_module_channels_base(
            module.v, predecessors_masks=predecessors_masks, output_masks=output_masks, vertex=vertex)
        return module
except ImportError:
    pass

try:
    from timm.models.beit import Attention as BeitAttention

    CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ += (BeitAttention,)


    @remove_timm_module_channels.register
    def _(
            module: BeitAttention,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        # TODO: this is a pass through method in anticipaton of adding actual pruning logic to `BeitAttention`
        return module
except ImportError:
    pass

try:
    from timm.layers.std_conv import ScaledStdConv2d, ScaledStdConv2dSame

    CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_ += (ScaledStdConv2d, ScaledStdConv2dSame)


    @remove_timm_module_channels.register
    def _(
            module: ScaledStdConv2d,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        # we need to handle `gain`
        intermediate_module = remove_module_channels_base(module, vertex, predecessors_masks, output_masks)
        out_indices = torch.where(output_masks[0] == 1)[0].view(-1, 1, 1, 1)
        new_gain = torch.take_along_dim(module.gain, dim=0, indices=out_indices)
        intermediate_module.gain.data = new_gain
        return intermediate_module


    @remove_timm_module_channels.register
    def _(
            module: ScaledStdConv2dSame,
            vertex: InnerVertex,
            predecessors_masks: List[List[torch.Tensor]],
            output_masks: List[torch.Tensor],
    ):
        # we need to handle `gain`
        intermediate_module = remove_module_channels_base(module, vertex, predecessors_masks, output_masks)
        out_indices = torch.where(output_masks[0] == 1)[0].view(-1, 1, 1, 1)
        new_gain = torch.take_along_dim(module.gain, dim=0, indices=out_indices)
        intermediate_module.gain.data = new_gain
        return intermediate_module

except ImportError:
    pass

CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES = CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES_
