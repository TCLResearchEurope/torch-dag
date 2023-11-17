import logging
from functools import singledispatch
from typing import Tuple

import numpy as np
import tensorflow as tf
from timm.models.layers import LayerNorm2d

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = False
session = tf.compat.v1.InteractiveSession(config=config)
import torch

import node_api as nd
from node_api.nodify_nodes import LambdaOpNode
from torch_dag.core import dag_module
from torch_dag import structured_modules
from torch_dag.core.common_tools import per_channel_noise_to_signal_ratio

logger = logging.getLogger(__name__)


def adjust_padding(padding, kernel_size):
    if isinstance(kernel_size, Tuple):
        kernel_size = kernel_size[0]
    if isinstance(padding, Tuple):
        h_padding = padding[1]
        w_padding = padding[2]
        assert h_padding[0] == h_padding[1]
        assert w_padding[0] == w_padding[1]
        padding = (h_padding[0], w_padding[0])
    elif padding == 'SAME':
        padding = kernel_size // 2
    elif padding == 'VALID':
        padding = 0
    else:
        # TODO: therea are some issues here. Once there was a need to run something like:
        # padding = padding[1][0]
        # padding = (padding, padding)
        raise NotImplementedError
    return padding


@singledispatch
def convert_node(node: nd.nodes, inst: nd.nodes.NodeInstance = None) -> torch.nn.Module:
    raise NotImplementedError(f"`Node` to `module` not implemented for {type(node)}")


@convert_node.register
def _(node: nd.cells.Cell, inst: nd.nodes.NodeInstance = None) -> torch.nn.Module:
    logger.info(f'Converting nested cell: {node}')
    return build_from_cell(node, cell_instance=inst)


@convert_node.register
def _(node: nd.ops.Conv2D, inst: nd.nodes.NodeInstance = None):
    if node.strides[1] > 1 and node.padding == 'SAME':
        module = structured_modules.Conv2DSameModule(
            in_channels=node.in_channels,
            out_channels=node.out_channels,
            kernel_size=node.filter_size,
            stride=node.strides[1:3],
            bias=node.biases is not None,
        )
    else:
        module = torch.nn.Conv2d(
            in_channels=node.in_channels,
            out_channels=node.out_channels,
            kernel_size=node.filter_size,
            stride=node.strides[1:3],
            bias=node.biases is not None,
            padding=adjust_padding(padding=node.padding, kernel_size=node.filter_size),
        )
    torch_weight = get_torch_tensor(tf.transpose(node.filters, (3, 2, 0, 1)))
    module.weight.data = torch_weight
    if module.bias is not None:
        module.bias.data = get_torch_tensor(node.biases)
    return module


@convert_node.register
def _(node: nd.ops.DepthwiseConv, inst: nd.nodes.NodeInstance = None):
    if node.strides[1] > 1 and node.padding == 'SAME':
        module = structured_modules.Conv2DSameModule(
            in_channels=node.in_channels,
            out_channels=node.out_channels,
            kernel_size=node.filter_size,
            stride=node.strides[1:3],
            bias=node.biases is not None,
            groups=node.in_channels,
        )
    else:
        module = torch.nn.Conv2d(
            in_channels=node.in_channels,
            out_channels=node.out_channels,
            kernel_size=node.filter_size,
            stride=node.strides[1:3],
            bias=node.biases is not None,
            groups=node.in_channels,
            padding=adjust_padding(padding=node.padding, kernel_size=node.filter_size),
        )
    torch_weight = torch.tensor(tf.transpose(node.filters, (2, 3, 0, 1)).numpy())
    module.weight.data = torch_weight
    if module.bias is not None:
        module.bias.data = torch.tensor(node.biases.numpy())
    return module


@convert_node.register
def _(node: nd.ops.Conv2DTranspose, inst: nd.nodes.NodeInstance = None):
    # @TODO: handle different cases, and add tests for Conv2DTranspose in future
    if node.strides[1] > 1 and node.padding == 'SAME':
        module = torch.nn.ConvTranspose2d(
            in_channels=node.in_channels,
            out_channels=node.out_channels,
            kernel_size=node.filter_size,
            stride=node.strides[1:3],
            padding=0,
            output_padding=0,
            bias=node.biases is not None,
            groups=1,
            dilation=node.dilations[1],
        )
    else:
        raise NotImplementedError(f"Please check `strides` and `padding` for {type(node)}, it will be possible to add specific cases here.")
    # node.filter shape: [filter_size, filter_size, out_channels, in_channels]
    # torch weights shape: [in_channels, out_channels , filter_size, filter_size]
    torch_weight = torch.tensor(tf.transpose(node.filters, (3, 2, 0, 1)).numpy())
    module.weight.data = torch_weight
    if module.bias is not None:
        module.bias.data = torch.tensor(node.biases.numpy())
    return module


@convert_node.register
def _(node: nd.ops.Dense, inst: nd.nodes.NodeInstance = None):
    module = torch.nn.Linear(
        in_features=node.kernel.shape[0],
        out_features=node.kernel.shape[1],
        bias=node.bias is not None,
    )
    torch_weight = get_torch_tensor(tf.transpose(node.kernel))
    module.weight.data = torch_weight
    if module.bias is not None:
        module.bias.data = get_torch_tensor(tf.transpose(node.bias))
    return module


@convert_node.register
def _(node: nd.ops.BatchNorm, inst: nd.nodes.NodeInstance):
    rank = len(inst.output_tensors[0].shape)
    if rank == 4:
        module = torch.nn.BatchNorm2d(
            num_features=node.nb_of_filters,
            eps=node.variance_epsilon,
            momentum=1.0 - node.momentum,
        )
        module.weight.data = get_torch_tensor(node.scale)
        module.bias.data = get_torch_tensor(node.offset)
        module.running_mean = get_torch_tensor(node.mean)
        module.running_var = get_torch_tensor(node.variance)
    elif rank == 3:
        bn = torch.nn.BatchNorm1d(
            num_features=node.nb_of_filters,
            eps=node.variance_epsilon,
            momentum=1.0 - node.momentum,
        )
        bn.weight.data = get_torch_tensor(node.scale)
        bn.bias.data = get_torch_tensor(node.offset)
        bn.running_mean = get_torch_tensor(node.mean)
        bn.running_var = get_torch_tensor(node.variance)
        module = structured_modules.TfBatchNorm1d(bn=bn)
    else:
        raise NotImplementedError

    return module


@convert_node.register
def _(node: nd.ops.EfficientAttention, inst: nd.nodes.NodeInstance):
    if node.query.biases is not None:
        use_bias = True
    else:
        use_bias = False
    module = structured_modules.EfficientAttention(dim=node.dim,
                                                   num_heads=node.num_heads,
                                                   use_bias=use_bias,
                                                   dropout_rate=node.dropout_rate,
                                                   output_dropout_rate=node.output_dropout_rate,
                                                   include_reshapings=node.include_reshapings)
    module.query.weight.data = get_torch_tensor(tf.transpose(node.query.filters, (3, 2, 0, 1)))
    module.key.weight.data = get_torch_tensor(tf.transpose(node.key.filters, (3, 2, 0, 1)))
    module.value.weight.data = get_torch_tensor(tf.transpose(node.value.filters, (3, 2, 0, 1)))
    module.output.weight.data = get_torch_tensor(tf.transpose(node.output.filters, (3, 2, 0, 1)))

    if use_bias:
        module.query.bias.data = get_torch_tensor(node.query.biases)
        module.key.bias.data = get_torch_tensor(node.key.biases)
        module.value.bias.data = get_torch_tensor(node.value.biases)
        module.output.bias.data = get_torch_tensor(node.output.biases)

    return module


@convert_node.register
def _(node: nd.ops.LayerNormalization, inst: nd.nodes.NodeInstance):
    rank = len(inst.output_tensors[0].shape)
    if rank == 3:
        ln = torch.nn.LayerNorm(
            normalized_shape=node.gamma.shape[0],
            eps=node.epsilon,
        )
        ln.weight.data = get_torch_tensor(node.gamma)
        ln.bias.data = get_torch_tensor(node.beta)
        module = ln
    elif rank == 4:
        ln = LayerNorm2d(
            num_channels=node.gamma.shape[0],
            eps=node.epsilon,
            affine=False
        )
        ln.weight = torch.nn.Parameter(data=get_torch_tensor(node.gamma))
        ln.bias = torch.nn.Parameter(data=get_torch_tensor(node.beta))
        module = ln
    else:
        raise NotImplementedError

    return module


@convert_node.register
def _(node: nd.ops.Activation, inst: nd.nodes.NodeInstance = None):
    if node.activation_name == 'softmax':
        dim = -1 if node.activation_kwargs is None else node.activation_kwargs.get('dim')
        return torch.nn.Softmax(dim=dim)
    return structured_modules.ActivationModuleBuilder.build_activation_module(node.activation_name)


@convert_node.register
def _(node: nd.ops.ZeroPadding2D, inst: nd.nodes.NodeInstance = None):
    return torch.nn.ZeroPad2d(node.padding)


@convert_node.register
def _(node: nd.ops.MaxPool2D, inst: nd.nodes.NodeInstance = None):
    kernel_size = node.kernel_size[1:3]
    stride = node.strides[1:3]
    if node.padding == 'SAME':
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = 0
    # print(kernel_size, stride, padding)
    return torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


@convert_node.register
def _(node: nd.ops.GlobalMeanPool2D, inst: nd.nodes.NodeInstance = None):
    return structured_modules.GlobalMeanPool2DModule(dim=(2, 3), keepdim=node.keepdims)


@convert_node.register
def _(node: nd.ops.Flatten, inst: nd.nodes.NodeInstance = None):
    return torch.nn.Flatten()


@convert_node.register
def _(node: nd.ops.Sum, inst: nd.nodes.NodeInstance = None):
    return structured_modules.AddModule()


@convert_node.register
def _(node: nd.ops.Sub, inst: nd.nodes.NodeInstance = None):
    return structured_modules.SubModule()


@convert_node.register
def _(node: nd.ops.Mul, inst: nd.nodes.NodeInstance = None):
    return structured_modules.MulModule()


@convert_node.register
def _(node: nd.ops.ChannelAffine, inst: nd.nodes.NodeInstance = None):
    module = structured_modules.ChannelAffineModule(
        num_channels=node.weight.shape[0],
        use_bias=node.bias is not None,
    )
    module.weight.data = get_torch_tensor(node.weight)
    if node.bias is not None:
        module.bias.data = get_torch_tensor(node.bias)
    return module


@convert_node.register
def _(node: LambdaOpNode, inst: nd.nodes.NodeInstance = None):
    return structured_modules.GlobalMeanPool2DModule(dim=(2, 3), keepdim=True)


@convert_node.register
def _(node: nd.ops.MeanPool2D, inst: nd.nodes.NodeInstance = None):
    if node.padding == 'SAME':
        padding = 0
    else:
        padding = (node.kernel_size[1] // 2, node.kernel_size[2] // 2)
    # TODO: Fix some padding issues
    module = torch.nn.AvgPool2d(
        kernel_size=node.kernel_size[1:3],
        stride=node.strides[1:3],
        padding=padding,
        count_include_pad=False,
    )
    return module


@convert_node.register
def _(node: nd.ops.TokenizeImage, inst: nd.nodes.NodeInstance = None):
    return structured_modules.TfTokenizeModule(patch_size=node.patch_size)


@convert_node.register
def _(node: nd.ops.DetokenizeImage, inst: nd.nodes.NodeInstance = None):
    return structured_modules.TfDetokenizeModule(patch_size=node.patch_size)


@convert_node.register
def _(node: nd.ops.MatMul, inst: nd.nodes.NodeInstance = None):
    return structured_modules.TfMatmulModule(transpose=node.transpose, normalize=node.normalize)


@convert_node.register
def _(node: nd.ops.SubpixelDownSampling, inst: nd.nodes.NodeInstance = None):
    return structured_modules.SpaceToDepthModule(block_size=node.scale)


@convert_node.register
def _(node: nd.ops.Concat, inst: nd.nodes.NodeInstance = None):
    if node.axis in (3, -1) and len(inst.output_tensors[0].shape) == 4:
        dim = 1
    elif len(inst.output_tensors[0].shape) == 3 and node.axis in (2, -1):
        dim = -1
    else:
        raise NotImplementedError
    return structured_modules.ConcatModule(dim=dim)


@convert_node.register
def _(node: nd.ops.TfImageResize, inst: nd.nodes.NodeInstance = None):
    if node.size is not None:
        return structured_modules.InterpolateModule(scale_factor=node.size, mode=node.method, antialias=node.antialias)
    elif node.exact_size is not None:
        return structured_modules.InterpolateModule(size=node.exact_size, mode=node.method, antialias=node.antialias)
    else:
        raise NotImplementedError


@convert_node.register
def _(node: nd.ops.Reshape, inst: nd.nodes.NodeInstance = None):
    return structured_modules.ReshapeModule(target_shape=node.target_shape)


@convert_node.register
def _(node: nd.ops.Dropout, inst: nd.nodes.NodeInstance = None):
    return torch.nn.Dropout(p=node.rate)


@convert_node.register
def _(node: nd.ops.TensorMerger, inst: nd.nodes.NodeInstance = None):
    return structured_modules.TensorMergerModule()


@convert_node.register
def _(node: nd.ops.TensorExtractor, inst: nd.nodes.NodeInstance = None):
    return structured_modules.TensorExtractorModule(index=node.index)


@convert_node.register
def _(node: nd.ops.BilinearUpSampling, inst: nd.nodes.NodeInstance = None):
    assert isinstance(node.size[0], int) and isinstance(node.size[1], int)
    if node.half_pixel_centers is False and node.align_corners is False:
        assert node.size[0] == node.size[1]
        return structured_modules.BilinearUpsampling(scale_factor=node.size[0])
    elif node.half_pixel_centers is False and node.align_corners is True:
        raise NotImplementedError(
            f'`tf.compat.v1.image.resize_bilinear` node.half_pixel_centers = False and node.align_corners = true is '
            f'not implemented.')
    else:
        return torch.nn.Upsample(scale_factor=node.size, mode='bilinear', align_corners=node.align_corners)


@convert_node.register
def _(node: nd.ops.Reduce, inst: nd.nodes.NodeInstance = None):
    if node.reduce_function_name == 'mean':
        return structured_modules.MeanModule(dim=node.axis, keepdim=node.keepdims)
    else:
        raise NotImplementedError


@convert_node.register
def _(node: nd.ops.Split, inst: nd.nodes.NodeInstance = None):
    return structured_modules.ChunkModule(dim=1, chunks=node.num_or_size_splits)


@convert_node.register
def _(node: nd.ops.SubpixelDownSampling, inst: nd.nodes.NodeInstance = None):
    return structured_modules.SpaceToDepthModule(block_size=node.scale)


@convert_node.register
def _(node: nd.ops.SubpixelUpSampling, inst: nd.nodes.NodeInstance = None):
    return structured_modules.DepthToSpaceModule(block_size=node.scale)


def get_torch_tensor(tensor: nd.backend.VARIABLE_AND_TENSOR_TYPE):
    return torch.tensor(tensor.numpy())


def find_instance(
        icn: nd.cells.InnerCellNode,
        cell_instance: nd.cells.CellInstance,
) -> nd.nodes.NodeInstance:
    for inst in cell_instance.inner_nodes_instances:
        if inst.instantiated_from == icn.node:
            return inst
        if isinstance(inst, nd.cells.CellInstance):
            return find_instance(icn, inst)


def build_from_cell(
        cell: nd.cells.Cell,
        cell_instance: nd.cells.CellInstance,
) -> dag_module.DagModule:
    vertices = [dag_module.InputVertex(name=f'input_{k}') for k in range(len(cell.input_cell_nodes))]
    dag = dag_module.DagModule(name=cell.name, vertices=vertices)
    for icn in cell.inner_cell_nodes:
        inst = find_instance(icn, cell_instance)
        pd_indices = cell._get_inner_node_predecessor_indices(icn)
        if isinstance(icn.node, nd.cells.Cell):
            torch_module = build_from_cell(icn.node, inst)
        else:
            torch_module = convert_node(icn.node, inst)
        predecessors = [dag.vertices[k] for k in pd_indices]
        vertex = dag.add_vertex(name=icn.node.name, module=torch_module, predecessors=predecessors)
        if icn == cell.output_nodes[0]:
            dag.output_vertex = vertex
    return dag


def convert_cell_to_torch_dag_module(
        cell: nd.cells.Cell,
        input_shape_without_batch: Tuple[int, ...],
        batch_size_for_verification: int = 4,
) -> Tuple[dag_module.DagModule, float]:
    if len(input_shape_without_batch) != 3:
        raise NotImplementedError(f'Currently supporting only BHWC input tensors')
    input_shape = (batch_size_for_verification,) + input_shape_without_batch
    cell.predict()
    cell_instance = cell(tf.ones(shape=input_shape))
    output_rank = len(cell_instance.output_tensors[0].shape)
    torch_dag_module = build_from_cell(cell, cell_instance=cell_instance)
    torch_dag_module.eval()

    x_tf = tf.random.normal(shape=input_shape)
    x_torch = torch.permute(torch.tensor(x_tf.numpy()), (0, 3, 1, 2))
    y_torch = torch_dag_module(x_torch)
    if isinstance(y_torch, list):
        # TODO: add real checks for all output tensors
        y_torch = y_torch[0]
    y_torch = y_torch.detach()
    if output_rank == 3:
        y_nd = cell(x_tf).output_tensors[0].numpy()
        non_channel_dim = (0, 1)
    elif output_rank == 2:
        y_nd = cell(x_tf).output_tensors[0].numpy()
        non_channel_dim = (0,)
    elif output_rank == 4:
        y_nd = tf.transpose(cell(x_tf).output_tensors[0], (0, 3, 1, 2)).numpy()
        non_channel_dim = (0, 2, 3)
    else:
        raise NotImplementedError

    diff = np.mean(np.abs(y_nd - y_torch.numpy()))
    nsr = per_channel_noise_to_signal_ratio(x=y_torch, y=torch.tensor(y_nd), non_channel_dim=non_channel_dim)
    logger.warning(f'Conversion MSE error: {diff}')
    logger.warning(f'Conversion NSR error: {nsr}')

    return torch_dag_module, nsr.numpy()
