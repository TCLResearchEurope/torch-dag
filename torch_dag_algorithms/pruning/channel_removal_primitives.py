import logging
from functools import singledispatch
from typing import List, Union

import torch
from torch import nn

from torch_dag import structured_modules as smodules
from torch_dag.core.dag_module import InnerVertex
from torch_dag.core.dag_tracer import _autowrap_modules
from torch_dag.core.module_handling_for_pruning import unprunable_modules
from torch_dag_algorithms.pruning.modules import MaskModule

logger = logging.getLogger(__name__)

trivial_behavior_modules = (
    nn.SiLU,
    smodules.ChunkModule,
    smodules.TensorExtractorModule,
    smodules.AddModule,
    smodules.ConcatModule,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.Dropout,
    nn.ReLU,
    nn.ReLU6,
    nn.Upsample,
    smodules.TensorMergerModule,
    nn.Identity,
    smodules.MulModule,
    smodules.MeanModule,
    nn.Sigmoid,
    nn.Softmax,
    nn.AdaptiveAvgPool2d,
    nn.Flatten,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.LeakyReLU,
    smodules.FlattenModule,
    smodules.TransposeModule,
    smodules.TfMatmulModule,
    nn.GELU,
    smodules.SliceModule,
    smodules.ParameterModule,
    torch.nn.ZeroPad2d,
    smodules.SpaceToDepthModule,
    smodules.DepthToSpaceModule,
    smodules.TfTokenizeModule,
    smodules.TfDetokenizeModule,
    smodules.PadModule,
    smodules.PermuteModule,
    smodules.GlobalMeanPool2DModule,
    smodules.ReshapeModule,
    smodules.ReshapeModuleV2,
    smodules.SubModule,
    smodules.ScalarMul,
    smodules.BilinearUpsampling,
    nn.LogSoftmax,
    smodules.PowerModule,
    smodules.SplitModule,
    smodules.GetShapeModule,
    smodules.GetShapeModuleV2,
    smodules.ReshapeWithSpecModule,
    smodules.ReshapeWithSpecModuleV2,
    smodules.AddcmulModule,
    smodules.UnbindModule,
    smodules.AutoWrapFunctionModule,
    smodules.ArgModule,
    nn.Conv1d,
    # GlobalResponseNorm,  # TODO this is only for block pruning!
)


def input_masks_are_empty(predecessors_masks: List[List[torch.Tensor]]):
    if predecessors_masks is None:
        return True
    if len(predecessors_masks) == 0:
        return True
    if len(predecessors_masks) == 1 and predecessors_masks[0] is None:
        return True
    if predecessors_masks is None:
        return True
    return False


def ouput_masks_are_empty(output_masks: List[torch.Tensor]):
    if output_masks is None:
        return True
    if len(output_masks) == 0:
        return True
    if len(output_masks) == 1 and output_masks[0] is None:
        return True
    return False


def get_full_input_masks(num_channels: int, device):
    return [[torch.ones(size=(num_channels,), device=device, dtype=torch.int32)]]


def get_full_output_masks(num_channels: int, device):
    return [torch.ones(size=(num_channels,), device=device, dtype=torch.int32)]


@singledispatch
def remove_module_channels_base(
        module: nn.Module,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if isinstance(module, trivial_behavior_modules + tuple(unprunable_modules)):
        return module
    if isinstance(module, tuple(_autowrap_modules)):
        logger.warning(f'No explicit channel removal logic implemented for {module.__class__}. '
                       f'Pruning will return the module as it is. In case of errors make sure that the module can handle '
                       f'variable number of input channels.')
        return module
    raise NotImplementedError(f'Channel removal not implemented for module class: {module.__class__}')


@remove_module_channels_base.register
def _(
        module: nn.Conv2d,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if input_masks_are_empty(predecessors_masks):
        predecessors_masks = get_full_input_masks(module.in_channels, module.weight.device)
    if ouput_masks_are_empty(output_masks):
        output_masks = get_full_output_masks(module.out_channels, module.weight.device)
    input_mask = predecessors_masks[0][0]
    output_mask = output_masks[0]
    in_indices = torch.where(input_mask == 1)[0]
    out_indices = torch.where(output_mask == 1)[0]

    # remove in channels
    if module.groups == module.in_channels and module.in_channels > 1:
        indices = in_indices.view(-1, 1, 1, 1)
        new_weight = torch.take_along_dim(module.weight, dim=0, indices=indices)
    elif module.groups == 1:
        indices = in_indices.view(1, -1, 1, 1)
        new_weight = torch.take_along_dim(module.weight, dim=1, indices=indices)
    else:
        new_weight = module.weight

    if 1 == module.groups:
        fraction = len(out_indices) / len(output_mask)
        logger.info(f'Pruning conv {vertex}: leaving fraction: {fraction} of out channels.')
        indices = out_indices.view(-1, 1, 1, 1)
        new_weight = torch.take_along_dim(new_weight, dim=0, indices=indices)
    if 1 < module.groups < module.in_channels:
        fraction = len(out_indices) / len(output_mask)
        logger.info(f'Pruning conv {vertex}: leaving fraction: {fraction} of out channels.')
        indices = out_indices.view(-1, 1, 1, 1)
        new_weight = torch.take_along_dim(new_weight, dim=0, indices=indices)
    if module.groups == module.in_channels:
        module.groups = len(in_indices)
    module.weight.data = new_weight
    module.in_channels = len(in_indices)
    module.out_channels = len(out_indices)
    if module.bias is not None:
        new_bias = torch.take_along_dim(module.bias, dim=0, indices=torch.where(output_mask == 1)[0])
        module.bias.data = new_bias
    return module


@remove_module_channels_base.register
def _(
        module: nn.ConvTranspose2d,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if input_masks_are_empty(predecessors_masks):
        predecessors_masks = get_full_input_masks(module.in_channels, module.weight.device)
    if ouput_masks_are_empty(output_masks):
        output_masks = get_full_output_masks(module.out_channels, module.weight.device)
    input_mask = predecessors_masks[0][0]
    output_mask = output_masks[0]
    in_indices = torch.where(input_mask == 1)[0]
    out_indices = torch.where(output_mask == 1)[0]

    if module.groups != 1:
        raise NotImplementedError

    indices = in_indices.view(-1, 1, 1, 1)
    new_weight = torch.take_along_dim(module.weight, dim=0, indices=indices)

    fraction = len(out_indices) / len(output_mask)
    logger.info(f'Pruning conv {vertex}: leaving fraction: {fraction} of out channels.')
    indices = out_indices.view(1, -1, 1, 1)
    new_weight = torch.take_along_dim(new_weight, dim=1, indices=indices)

    module.weight.data = new_weight
    module.in_channels = len(in_indices)
    module.out_channels = len(out_indices)
    if module.bias is not None:
        new_bias = torch.take_along_dim(module.bias, dim=0, indices=torch.where(output_mask == 1)[0])
        module.bias.data = new_bias
    return module


def _remove_batch_norm_channels(
        module: Union[nn.BatchNorm2d, nn.BatchNorm2d],
        predecessors_masks: List[List[torch.Tensor]],
):
    if input_masks_are_empty(predecessors_masks):
        predecessors_masks = get_full_input_masks(module.num_features, module.weight.device)
    input_mask = predecessors_masks[0][0]
    indices = torch.where(input_mask == 1)[0]
    # remove in channels
    new_weight = torch.take_along_dim(module.weight, dim=0, indices=indices)
    new_bias = torch.take_along_dim(module.bias, dim=0, indices=indices)
    new_mean = torch.take_along_dim(module.running_mean, dim=0, indices=indices)
    new_var = torch.take_along_dim(module.running_var, dim=0, indices=indices)
    module.weight.data = new_weight
    module.bias.data = new_bias
    module.running_mean.data = new_mean
    module.running_var.data = new_var
    module.num_features = module.weight.shape[0]
    return module


@remove_module_channels_base.register
def _(
        module: nn.BatchNorm2d,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    return _remove_batch_norm_channels(module, predecessors_masks)


@remove_module_channels_base.register
def _(
        module: nn.BatchNorm1d,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    return _remove_batch_norm_channels(module, predecessors_masks)


@remove_module_channels_base.register
def _(
        module: nn.Linear,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if input_masks_are_empty(predecessors_masks):
        predecessors_masks = get_full_input_masks(module.in_features, module.weight.device)
    if ouput_masks_are_empty(output_masks):
        output_masks = get_full_output_masks(module.out_features, module.weight.device)
    input_mask = predecessors_masks[0][0]
    output_mask = output_masks[0]
    in_indices = torch.where(input_mask == 1)[0]
    out_indices = torch.where(output_mask == 1)[0]

    module.in_features = len(in_indices)
    module.out_features = len(out_indices)

    # remove in channels
    in_indices = in_indices.view(1, -1)
    new_weight = torch.take_along_dim(module.weight, dim=1, indices=in_indices)
    # remove out channels
    fraction = len(out_indices) / len(output_mask)
    logger.info(f'Pruning conv {vertex}: leaving fraction: {fraction} of out channels.')
    out_indices = out_indices.view(-1, 1)
    new_weight = torch.take_along_dim(new_weight, dim=0, indices=out_indices)
    module.weight.data = new_weight
    if module.bias is not None:
        new_bias = torch.take_along_dim(module.bias, dim=0, indices=torch.where(output_mask == 1)[0])
        module.bias.data = new_bias
    return module


@remove_module_channels_base.register
def _(
        module: MaskModule,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    return nn.Identity()


@remove_module_channels_base.register
def _(
        module: smodules.AuxiliaryTokenModule,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if vertex.orbit is None:
        return module
    else:
        raise NotImplementedError


@remove_module_channels_base.register
def _(
        module: nn.LayerNorm,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if input_masks_are_empty(predecessors_masks):
        predecessors_masks = get_full_input_masks(module.weight.shape[0], module.weight.device)
    input_mask = predecessors_masks[0][0]
    indices = torch.where(input_mask == 1)[0]
    # remove in channels
    new_weight = torch.take_along_dim(module.weight, dim=0, indices=indices)
    module.weight.data = new_weight
    if module.bias is not None:
        new_bias = torch.take_along_dim(module.bias, dim=0, indices=indices)
        module.bias.data = new_bias
    module.normalized_shape = (len(indices),)
    return module


@remove_module_channels_base.register
def _(
        module: smodules.ChannelAffineModule,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if input_masks_are_empty(predecessors_masks):
        predecessors_masks = get_full_input_masks(module.weight.shape[0], module.weight.device)
    input_mask = predecessors_masks[0][0]
    indices = torch.where(input_mask == 1)[0]
    # remove in channels
    new_weight = torch.take_along_dim(module.weight, dim=0, indices=indices)
    module.weight.data = new_weight
    if module.use_bias:
        new_bias = torch.take_along_dim(module.bias, dim=0, indices=indices)
        module.bias.data = new_bias
    module.num_channels = len(indices)

    return module


@remove_module_channels_base.register
def _(
        module: nn.GroupNorm,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if input_masks_are_empty(predecessors_masks) is None:
        predecessors_masks = get_full_input_masks(module.weight.shape[0], module.weight.device)
    input_mask = predecessors_masks[0][0]
    indices = torch.where(input_mask == 1)[0]
    # remove in channels
    new_weight = torch.take_along_dim(module.weight, dim=0, indices=indices)
    module.weight.data = new_weight
    if module.bias is not None:
        new_bias = torch.take_along_dim(module.bias, dim=0, indices=indices)
        module.bias.data = new_bias
    module.num_channels = len(indices)

    return module


@remove_module_channels_base.register
def _(
        module: smodules.EfficientAttention,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    mask = predecessors_masks[0][0]
    indices = torch.where(mask == 1)[0]
    if len(indices) == len(mask):
        return module
