import logging
from functools import singledispatch
from typing import Tuple

import ipdb
import numpy as np
import tensorflow as tf
import timm

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = False
session = tf.compat.v1.InteractiveSession(config=config)
import torch

import node_api as nd
from torch_dag.core import dag_module
from torch_dag import structured_modules as smodules
from node_api_conversion.utils import tf_per_channel_noise_to_signal_ratio

logger = logging.getLogger(__name__)


def convert_conv_weight(weight: torch.nn.Parameter, is_depthwise: bool = False):
    if is_depthwise:
        return torch.permute(weight, (2, 3, 0, 1)).detach().numpy()
    else:
        return torch.permute(weight, (2, 3, 1, 0)).detach().numpy()


@singledispatch
def convert_module(module: torch.nn.Module, iv: dag_module.InnerVertex) -> nd.nodes.Node:
    raise NotImplementedError(f"`Module` to `node` not implemented for {type(module)}")


def adjust_padding(torch_padding):
    if torch_padding == (1, 1):
        return ((0, 0,), (1, 1), (1, 1), (0, 0))
    elif torch_padding == (2, 2):
        return ((0, 0,), (2, 2), (2, 2), (0, 0))
    elif torch_padding == (3, 3):
        return ((0, 0,), (3, 3), (3, 3), (0, 0))
    elif torch_padding == (0, 0):
        return 'VALID'
    else:
        raise NotImplementedError


def adjust_padding_zero_pad(torch_padding):
    if torch_padding == (2, 2, 2, 2):
        return (2, 2)
    elif torch_padding == (1, 1, 1, 1):
        return (1, 1)
    else:
        raise NotImplementedError


@convert_module.register
def _(module: torch.nn.Conv2d, iv: dag_module.InnerVertex):
    assert module.kernel_size[0] == module.kernel_size[1]

    if module.groups != 1:
        node = nd.ops.DepthwiseConv.from_conv_op_params(
            name=iv.name,
            in_channels=module.in_channels,
            filter_size=module.kernel_size[0],
            use_bias=module.bias is not None,
            strides=module.stride,
            expansion_rate=1,
            padding=adjust_padding(module.padding)
        )
        node.filters.assign(convert_conv_weight(module.weight, is_depthwise=True))

    else:
        node = nd.ops.Conv2D.from_conv_op_params(
            name=iv.name,
            in_channels=module.weight.shape[1],
            out_channels=module.weight.shape[0],
            filter_size=module.kernel_size[0],
            use_bias=module.bias is not None,
            strides=module.stride,
            padding=adjust_padding(module.padding)
        )
        node.filters.assign(convert_conv_weight(module.weight))
    if module.bias is not None:
        node.biases.assign(module.bias.detach().numpy())

    return node


@convert_module.register
def _(module: torch.nn.BatchNorm2d, iv: dag_module.InnerVertex):
    node = nd.ops.BatchNorm.build_for_nb_of_filters(
        name=iv.name,
        nb_of_filters=module.weight.shape[0],
        momentum=1.0 - module.momentum,
        variance_epsilon=module.eps
    )
    node.mean.assign(module.running_mean.numpy())
    node.variance.assign(module.running_var.detach().numpy())
    node.scale.assign(module.weight.detach().numpy())
    node.offset.assign(module.bias.detach().numpy())
    return node

@convert_module.register
def _(module: torch.nn.BatchNorm1d, iv: dag_module.InnerVertex):
    node = nd.ops.BatchNorm.build_for_nb_of_filters(
        name=iv.name,
        nb_of_filters=module.weight.shape[0],
        momentum=1.0 - module.momentum,
        variance_epsilon=module.eps
    )
    node.mean.assign(module.running_mean.numpy())
    node.variance.assign(module.running_var.detach().numpy())
    node.scale.assign(module.weight.detach().numpy())
    node.offset.assign(module.bias.detach().numpy())
    return node

@convert_module.register
def _(module: smodules.TfBatchNorm1d, iv: dag_module.InnerVertex):
    return convert_module(module.bn, iv)



@convert_module.register
def _(module: smodules.ChunkModule, iv: dag_module.InnerVertex):
    return nd.ops.Split(
        name=iv.name,
        num_or_size_splits=module.chunks,
        axis=-1,
    )


@convert_module.register
def _(module: smodules.FlattenModule, iv: dag_module.InnerVertex):
    return nd.ops.Flatten(
        name=iv.name,
    )


@convert_module.register
def _(module: torch.nn.SiLU, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='swish',
    )


@convert_module.register
def _(module: torch.nn.Identity, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='none',
    )


@convert_module.register
def _(module: torch.nn.Hardswish, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='hard_swish',
    )


@convert_module.register
def _(module: torch.nn.Hardsigmoid, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='hard_sigmoid',
    )


@convert_module.register
def _(module: torch.nn.ReLU, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='relu',
    )


@convert_module.register
def _(module: torch.nn.ReLU6, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='relu6',
    )


@convert_module.register
def _(module: torch.nn.Softmax, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name_and_kwargs(
        name=iv.name,
        activation_name='softmax',
        activation_kwargs={'axis': module.dim}
    )

@convert_module.register
def _(module: torch.nn.Sigmoid, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='sigmoid',
    )


@convert_module.register
def _(module: torch.nn.Dropout, iv: dag_module.InnerVertex):
    return nd.ops.Dropout(
        name=iv.name,
        rate=module.p,
    )


@convert_module.register
def _(module: smodules.TensorExtractorModule, iv: dag_module.InnerVertex):
    return nd.ops.TensorExtractor(
        name=iv.name,
        index=module.index,
    )


@convert_module.register
def _(module: smodules.AddModule, iv: dag_module.InnerVertex):
    return nd.ops.Sum(
        name=iv.name,
    )


@convert_module.register
def _(module: smodules.SubModule, iv: dag_module.InnerVertex):
    return nd.ops.Sub(
        name=iv.name,
    )


@convert_module.register
def _(module: smodules.ChannelAffineModule, iv: dag_module.InnerVertex):
    node = nd.ops.ChannelAffine.from_op_params(
        name=iv.name,
        use_bias=module.use_bias,
        num_units=module.num_channels,
    )
    node.weight.assign(module.weight.detach().numpy())
    if module.bias is not None:
        node.bias.assign(module.bias.detach().numpy())
    return node


@convert_module.register
def _(module: smodules.MeanModule, iv: dag_module.InnerVertex):
    if isinstance(module.dim, int):
        if module.dim == 1:
            axis = 1
        elif module.dim == 2:
            axis = 2
        else:
            raise NotImplementedError
    elif tuple(module.dim) == (2, 3):
        axis = (1, 2)
    elif module.dim == 1:
        axis = 1
    else:
        raise NotImplementedError
    return nd.ops.Reduce(
        name=iv.name,
        reduce_function_name='mean',
        axis=axis,
        keepdims=module.keepdim,
    )


@convert_module.register
def _(module: smodules.MulModule, iv: dag_module.InnerVertex):
    return nd.ops.Mul(
        name=iv.name,
    )


@convert_module.register
def _(module: smodules.ConcatModule, iv: dag_module.InnerVertex):
    return nd.ops.Concat(
        name=iv.name,
        axis=-1,
    )


@convert_module.register
def _(module: torch.nn.ZeroPad2d, iv: dag_module.InnerVertex):
    return nd.ops.ZeroPadding2D(
        name=iv.name,
        padding=adjust_padding_zero_pad(torch_padding=module.padding)
    )

@convert_module.register
def _(module: smodules.PadModule, iv: dag_module.InnerVertex):
    return nd.ops.ZeroPadding2D(
        name=iv.name,
        padding=adjust_padding_zero_pad(torch_padding=module.pad)
    )


@convert_module.register
def _(module: torch.nn.MaxPool2d, iv: dag_module.InnerVertex):
    return nd.ops.MaxPool2D(
        name=iv.name,
        kernel_size=module.kernel_size,
        strides=module.stride,
        padding='SAME',
    )


@convert_module.register
def _(module: torch.nn.Upsample, iv: dag_module.InnerVertex):
    if module.mode == 'nearest' and module.scale_factor is None:
        raise NotImplementedError
    if module.size is not None:
        return nd.ops.TfImageResize(
            name=iv.name,
            exact_size=module.size,
            method=module.mode
        )
    else:
        if not isinstance(module.scale_factor, Tuple):
            scale_factor = (module.scale_factor, module.scale_factor)
        else:
            scale_factor = module.scale_factor
        size = (int(scale_factor[0]), int(scale_factor[0]))
        return nd.ops.TfImageResize(
            name=iv.name,
            size=size,
            method=module.mode
        )


@convert_module.register
def _(module: torch.nn.AvgPool2d, iv: dag_module.InnerVertex):
    return nd.ops.MeanPool2D(
        name=iv.name,
        kernel_size=module.kernel_size,
        strides=module.stride,
        padding='SAME',
    )


@convert_module.register
def _(module: smodules.TensorMergerModule, iv: dag_module.InnerVertex):
    return nd.ops.TensorMerger(
        name=iv.name,
    )


@convert_module.register
def _(module: smodules.SpaceToDepthModule, iv: dag_module.InnerVertex):
    return nd.ops.SubpixelDownSampling(
        name=iv.name,
        scale=module.block_size,
    )


@convert_module.register
def _(module: smodules.DepthToSpaceModule, iv: dag_module.InnerVertex):
    return nd.ops.SubpixelUpSampling(
        name=iv.name,
        scale=module.block_size,
    )


@convert_module.register
def _(module: smodules.TfTokenizeModule, iv: dag_module.InnerVertex):
    return nd.ops.TokenizeImage(
        name=iv.name,
        patch_size=module.patch_size,
    )


@convert_module.register
def _(module: smodules.TfDetokenizeModule, iv: dag_module.InnerVertex):
    return nd.ops.DetokenizeImage(
        name=iv.name,
        patch_size=module.patch_size,
    )


@convert_module.register
def _(module: smodules.TfMatmulModule, iv: dag_module.InnerVertex):
    return nd.ops.MatMul(
        name=iv.name,
        transpose=module.transpose,
        normalize=module.normalize,
    )


@convert_module.register
def _(module: torch.nn.LayerNorm, iv: dag_module.InnerVertex):
    node = nd.ops.LayerNormalization.build_for_nb_of_filters(
        name=iv.name,
        nb_of_filters=module.normalized_shape[0],
        epsilon=module.eps,
    )
    node.gamma.assign(module.weight.numpy())
    node.beta.assign(module.bias.numpy())
    return node


@convert_module.register
def _(module: timm.models.layers.LayerNorm2d, iv: dag_module.InnerVertex):
    node = nd.ops.LayerNormalization.build_for_nb_of_filters(
        name=iv.name,
        nb_of_filters=module.normalized_shape[0],
        epsilon=module.eps,
    )
    node.gamma.assign(module.weight.numpy())
    node.beta.assign(module.bias.numpy())
    return node


@convert_module.register
def _(module: torch.nn.Linear, iv: dag_module.InnerVertex):
    node = nd.ops.Dense.from_dense_op_params(
        name=iv.name,
        in_units=module.weight.shape[1],
        out_units=module.weight.shape[0],
        use_bias=module.bias is not None,
    )
    node.kernel.assign(module.weight.transpose(0, 1).numpy())
    if module.bias is not None:
        node.bias.assign(module.bias.numpy())
    return node


# TODO: We may need these in the future. There are corresponding nodes in `node-api` written
# TODO: already we may just have to merge them.
@convert_module.register
def _(module: smodules.PatchifyModule, iv: dag_module.InnerVertex):
    return nd.ops.Patchify(
        name=iv.name,
    )


@convert_module.register
def _(module: smodules.DePatchifyModule, iv: dag_module.InnerVertex):
    return nd.ops.Depatchify(
        name=iv.name,
    )


@convert_module.register
def _(module: smodules.ParameterModule, iv: dag_module.InnerVertex):
    param = tf.Variable(module.param.numpy())
    return nd.ops.Param(
        name=iv.name,
        param=param,
    )


# @convert_module.register
# def _(module: smodules.TransposeModule, iv: dag_module.InnerVertex):
#     return nd.ops.Transpose(
#         name=iv.name,
#         perm=(0, 1, 2),
#     )

@convert_module.register
def _(module: smodules.PermuteModule, iv: dag_module.InnerVertex):
    return nd.ops.Transpose(
        name=iv.name,
        perm=module.perm,
    )


@convert_module.register
def _(module: smodules.AuxiliaryTokenModule, iv: dag_module.InnerVertex):
    param = tf.Variable(module.token.numpy()[None, :])
    return nd.ops.AddAuxiliaryTokens(
        name=iv.name,
        token_variable=param,
    )


@convert_module.register
def _(module: torch.nn.GELU, iv: dag_module.InnerVertex):
    return nd.ops.Activation.from_activation_name(
        name=iv.name,
        activation_name='gelu',
    )


# @convert_module.register
# def _(module: timm.models.vision_transformer.LayerScale, iv: dag_module.InnerVertex):
#     # TODO: Fix this. This is done just for latency measurement
#     return nd.ops.Activation.from_activation_name(
#         name=iv.name,
#         activation_name='none',
#     )


# TODO: this may be needed in the future (will require additional node in `node-api`
@convert_module.register
def _(module: smodules.NormalizeModule, iv: dag_module.InnerVertex):
    raise NotImplementedError


# TODO: investigate nn.functional.padding conversion
# @convert_module.register
# def _(module: smodules.PadModule, iv: dag_module.InnerVertex):
#     print(module.pad)
#     return nd.ops.ZeroPadding2D(name=iv.name, padding=module.pad)


@convert_module.register
def _(module: smodules.InterpolateModule, iv: dag_module.InnerVertex):
    if module.scale_factor is None and module.mode == 'nearest':
        raise NotImplementedError
    if module.scale_factor:
        size = (module.scale_factor, module.scale_factor)
        return nd.ops.TfImageResize(
            name=iv.name,
            size=size,
            method=module.mode,
            antialias=module.antialias,
        )
    else:
        return nd.ops.TfImageResize(
            name=iv.name,
            exact_size=module.size,
            method=module.mode,
            antialias=module.antialias,
        )


@convert_module.register
def _(module: smodules.GlobalMeanPool2DModule, iv: dag_module.InnerVertex):
    if module.dim == (2, 3):
        return nd.ops.GlobalMeanPool2D(
            name=iv.name,
            keepdims=module.keepdim,
        )
    else:
        raise NotImplementedError


@convert_module.register
def _(module: smodules.ReshapeModule, iv: dag_module.InnerVertex):
    return nd.ops.Reshape(
        name=iv.name,
        target_shape=module.target_shape,
    )

@convert_module.register
def _(module: torch.nn.Flatten, iv: dag_module.InnerVertex):
    return nd.ops.Flatten(
        name=iv.name,
    )

@convert_module.register
def _(module: torch.nn.AdaptiveAvgPool2d, iv: dag_module.InnerVertex):
    if module.output_size in (1, (1, 1)):
        return nd.ops.GlobalMeanPool2D(
            name=iv.name,
            keepdims=True,
        )
    else:
        raise NotImplementedError


@convert_module.register
def _(module: smodules.EfficientAttention, iv: dag_module.InnerVertex):

    if module.query.bias is not None:
        use_bias = True
    else:
        use_bias = False

    query = nd.ops.Conv2D.from_conv_op_params(
        name=f'{iv.name}_query',
        in_channels=module.query.in_channels,
        out_channels=module.query.out_channels,
        filter_size=module.query.kernel_size[0],
        use_bias=module.query.bias is not None,
        strides=module.query.stride,
        padding=adjust_padding(module.query.padding)
    )
    query.filters.assign(convert_conv_weight(module.query.weight))
    if use_bias is not None:
        query.biases.assign(module.query.bias.detach().numpy())

    key = nd.ops.Conv2D.from_conv_op_params(
        name=f'{iv.name}_key',
        in_channels=module.key.in_channels,
        out_channels=module.key.out_channels,
        filter_size=module.key.kernel_size[0],
        use_bias=module.key.bias is not None,
        strides=module.key.stride,
        padding=adjust_padding(module.key.padding)
    )
    key.filters.assign(convert_conv_weight(module.key.weight))
    if use_bias is not None:
        key.biases.assign(module.key.bias.detach().numpy())

    value = nd.ops.Conv2D.from_conv_op_params(
        name=f'{iv.name}_value',
        in_channels=module.value.in_channels,
        out_channels=module.value.out_channels,
        filter_size=module.value.kernel_size[0],
        use_bias=module.value.bias is not None,
        strides=module.value.stride,
        padding=adjust_padding(module.value.padding)
    )
    value.filters.assign(convert_conv_weight(module.value.weight))
    if use_bias is not None:
        value.biases.assign(module.value.bias.detach().numpy())

    output = nd.ops.Conv2D.from_conv_op_params(
        name=f'{iv.name}_output',
        in_channels=module.output.in_channels,
        out_channels=module.output.out_channels,
        filter_size=module.output.kernel_size[0],
        use_bias=module.output.bias is not None,
        strides=module.output.stride,
        padding=adjust_padding(module.output.padding)
    )
    output.filters.assign(convert_conv_weight(module.output.weight))
    if use_bias is not None:
        output.biases.assign(module.output.bias.detach().numpy())
    node = nd.ops.EfficientAttention(
        name=iv.name,
        dim=module.dim,
        num_heads=module.num_heads,
        query=query,
        key=key,
        value=value,
        output=output,
        dropout_rate=module.dropout_rate,
        output_dropout_rate=module.output_dropout_rate,
        include_reshapings=module.include_reshapings
    )

    return node


@convert_module.register
def _(module: smodules.BilinearUpsampling, iv: dag_module.InnerVertex):
    upsample = nd.ops.BilinearUpSampling(
        name=iv.name,
        size=(int(module.scale_factor), int(module.scale_factor)),
        half_pixel_centers=False
    )

    return upsample


@convert_module.register
def _(module: dag_module.DagModule, iv: dag_module.InnerVertex):
    return build_from_dag(module)


def build_from_dag(
        dag: dag_module.DagModule,
        flatten: bool = True,
) -> nd.cells.Cell:
    if flatten:
        dag = dag.flatten()
    input_nodes = [nd.cells.InputCellNode() for _ in dag.input_vertices]
    cell = nd.cells.Cell(
        name=dag.name,
        input_cell_nodes=input_nodes,
    )
    for iv in dag.inner_vertices:
        pd_indices = dag._get_inner_vertex_predecessor_indices(inner_vertex=iv)
        node = convert_module(iv.module, iv)
        predecessors = [cell.cell_nodes[k] for k in pd_indices]
        icn = cell.register_node(node=node, predecessors=predecessors)
        if iv == dag.output_vertex:
            cell.set_output(icn)
    return cell


def convert_dag_module_to_cell(
        dag: dag_module.DagModule,
        input_shape_without_batch: Tuple[int, ...],
        batch_size_for_verification: int = 8,
        flatten: bool = False,
) -> Tuple[nd.cells.Cell, float]:
    if len(input_shape_without_batch) != 3:
        raise NotImplementedError(f'Currently supporting only BHWC input tensors')
    input_shape = (batch_size_for_verification,) + input_shape_without_batch
    dag.eval()
    cell = build_from_dag(dag, flatten=flatten)
    cell.predict()

    x = np.random.normal(size=input_shape).astype(np.float32)
    x_tf = tf.transpose(x, (0, 2, 3, 1))
    x_torch = torch.tensor(x)

    out_torch = dag(torch.zeros(size=input_shape))
    if isinstance(out_torch, list):
        out_torch = out_torch[0]
    output_rank = len(out_torch.shape)

    y_torch = dag(x_torch)
    if isinstance(y_torch, list):
        y_torch = y_torch[0].detach().numpy()
    else:
        y_torch = y_torch.detach().numpy()
    if output_rank == 3:
        y_nd = tf.transpose(cell(x_tf).output_tensors[0], (0, 2, 1)).numpy()
    elif output_rank == 2:
        y_nd = cell(x_tf).output_tensors[0].numpy()
    elif output_rank == 4:
        y_nd = tf.transpose(cell(x_tf).output_tensors[0], (0, 3, 1, 2)).numpy()
    else:
        raise NotImplementedError

    diff = np.mean(np.abs(y_nd - y_torch))
    tf_nsr = tf_per_channel_noise_to_signal_ratio(x=tf.constant(y_torch), y=tf.constant(y_nd))
    logger.warning(f'Conversion MSE error: {diff}')
    logger.warning(f'Conversion NSR error: {tf_nsr}')

    return cell, tf_nsr.numpy()
