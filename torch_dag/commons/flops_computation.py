#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging
from functools import singledispatch
from typing import Tuple, List, Union, Collection, Type

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from torch_dag import structured_modules as smodules
from torch_dag.commons.mixins import CustomComputeFlopsMixin
from torch_dag.core import dag_module
from torch_dag_timm_plugin.timm_flops_computation import CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES, \
    timm_custom_compute_flops

logger = logging.getLogger(__name__)

CUSTOM_FLOPS_COMPUTATION_MODULES = (
    smodules.AddModule,
)


@singledispatch
def custom_compute_flops(
        module: torch.nn.Module,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    raise NotImplementedError


@custom_compute_flops.register(smodules.AddModule)
@custom_compute_flops.register(smodules.MulModule)
@custom_compute_flops.register(smodules.ScalarMul)
def _(
        module: Union[smodules.AddModule, smodules.MulModule, smodules.ScalarMul],
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    x = inputs[0]
    return np.prod(x.shape[1:])


@custom_compute_flops.register(torch.nn.Dropout)
@custom_compute_flops.register(smodules.ConcatModule)
@custom_compute_flops.register(smodules.ChunkModule)
@custom_compute_flops.register(smodules.UnbindModule)
@custom_compute_flops.register(torch.nn.Identity)
@custom_compute_flops.register(smodules.TensorExtractorModule)
@custom_compute_flops.register(smodules.ArgModule)
def _(
        module: Union[
            torch.nn.Dropout,
            smodules.ConcatModule,
            smodules.ChunkModule,
            smodules.UnbindModule,
            torch.nn.Identity,
            smodules.TensorExtractorModule,
            smodules.ArgModule,
        ],
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    return 0


@custom_compute_flops.register
def _(
        module: torch.nn.Softmax,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    return 5 * np.prod(inputs.shape[1:])


@custom_compute_flops.register
def _(
        module: torch.nn.ReLU,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    return 2 * np.prod(inputs.shape[1:])


def compute_2d_pooling_flops(
        module: Union[torch.nn.MaxPool2d, torch.nn.AvgPool2d],
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    # tested against TF flops computation and inspierd by `node-api`
    if isinstance(module.kernel_size, int):
        k0 = k1 = module.kernel_size
    else:
        k0, k1 = module.kernel_size[0], module.kernel_size[1]
    if isinstance(module.stride, int):
        s0 = s1 = module.kernel_size
    else:
        s0, s1 = module.stride[0], module.stride[1]
    c, h, w = inputs.shape[1:]
    num_tiles = (h + s0 - 1) // s0 * (w + s1) // s1
    return k0 * k1 * num_tiles * c


@custom_compute_flops.register
def _(
        module: torch.nn.MaxPool2d,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    return compute_2d_pooling_flops(module, inputs)


@custom_compute_flops.register
def _(
        module: torch.nn.AvgPool2d,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    return compute_2d_pooling_flops(module, inputs)


@custom_compute_flops.register
def _(
        module: CustomComputeFlopsMixin,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    return module.compute_flops(inputs)


def validate_flops_analyzer_input(x) -> bool:
    if isinstance(x, torch.Tensor):
        return True
    if isinstance(x, list):
        for element in x:
            if not isinstance(element, torch.Tensor):
                return False
    return True


def compute_torch_flops_for_vertex_and_inputs(
        vertex: dag_module.InnerVertex, inputs,
        verbose: bool = False,
        custom_zero_flops_modules: Tuple[Type[torch.nn.Module]] = None,
) -> int:
    if not validate_flops_analyzer_input(inputs):
        return 0
    if isinstance(inputs, List) and len(inputs) == 1:
        inputs = inputs[0]
    if isinstance(vertex.module, custom_zero_flops_modules):
        return 0
    if isinstance(vertex.module, CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES):
        return timm_custom_compute_flops(vertex.module, inputs)
    try:
        custom_flops = custom_compute_flops(vertex.module, inputs)
        return custom_flops / 2.0
    except NotImplementedError:
        pass

    flops_analyser = FlopCountAnalysis(vertex.module, inputs)
    if not verbose:
        flops_analyser.uncalled_modules_warnings(False)
        flops_analyser.unsupported_ops_warnings(False)

    torch_flops = flops_analyser.total()
    if torch_flops == 0:
        logger.debug(f'Returning 0 flops for vertex: {vertex} with module class: '
                     f'{vertex.module.__class__}.')
    return torch_flops


def compute_torch_flops_for_dag_module(
        dag_module: dag_module.DagModule,
        input_shape_without_batch: Tuple[int, ...],
        verbose: bool = False,
        custom_zero_flops_modules: Collection[Type[torch.nn.Module]] = [],
):
    """
    `fvcore` computes FLOPs which are multiadds / 2.0, therefore whenever we port flops computation
    form `node-api` we divide the results by 2.0
    :param dag_module:
    :param input_shape_without_batch:
    :return:
    """
    custom_zero_flops_modules = tuple(custom_zero_flops_modules)
    dag_module.clear_custom_buffers()
    is_caching = dag_module.cache_forward_dict
    if not is_caching:
        dag_module.cache_forward_dict = True
    input_tensor = torch.ones(size=(1,) + input_shape_without_batch, device=dag_module.device)
    _ = dag_module(input_tensor)
    result = {}
    for vertex in dag_module.inner_vertices:
        x = dag_module.inputs_dict[vertex]
        # no input vertices usually stand for input variables and we return 0 flops
        if not x:
            torch_flops = 0
        else:
            torch_flops = compute_torch_flops_for_vertex_and_inputs(
                vertex=vertex,
                inputs=x,
                verbose=verbose,
                custom_zero_flops_modules=custom_zero_flops_modules,
            )
        result[vertex] = torch_flops
    if not is_caching:
        dag_module.clear_tensor_dicts()
        dag_module.cache_forward_dict = False
    dag_module.clear_custom_buffers()
    return result


def compute_static_kmapp(
        dag_module: dag_module.DagModule,
        input_shape_without_batch: Tuple[int, ...],
        custom_zero_flops_modules: List[Type[torch.nn.Module]] = [],
):
    dag_module.clear_custom_buffers()
    is_caching = dag_module.cache_forward_dict
    if not is_caching:
        dag_module.cache_forward_dict = True
    x = torch.ones(size=(1,) + input_shape_without_batch, device=dag_module.device)
    if len(x.shape) != 4:
        raise NotImplementedError(f'kmapp computation is only implemented for (B, C, H, W) '
                                  f'input shape.')
    h, w = x.shape[2], x.shape[3]
    flops_dict = compute_torch_flops_for_dag_module(
        dag_module=dag_module,
        input_shape_without_batch=input_shape_without_batch,
        custom_zero_flops_modules=custom_zero_flops_modules,
    )
    if not is_caching:
        dag_module.clear_tensor_dicts()
        dag_module.cache_forward_dict = False
    dag_module.clear_custom_buffers()
    return 2.0 * sum(flops_dict.values()) / (1e3 * h * w)


def build_full_flops_list(
        dag: dag_module.DagModule,
        input_shape_without_batch: Tuple[int, ...],
        normalize: bool = False,
        custom_zero_flops_modules: List[Type[torch.nn.Module]] = [],
) -> List:
    dag.clear_custom_buffers()
    if dag.training:
        dag.eval()
    flops_dict = compute_torch_flops_for_dag_module(
        dag_module=dag,
        input_shape_without_batch=input_shape_without_batch,
        custom_zero_flops_modules=custom_zero_flops_modules,
    )
    result = []
    for k, vertex in enumerate(flops_dict.keys()):
        flops = 2.0 * flops_dict[vertex]
        result.append(flops)
    if dag.training:
        dag.train()
    if normalize:
        normalization = 1e3 * input_shape_without_batch[1] * input_shape_without_batch[2]
        result = [e / normalization for e in result]
    dag.clear_custom_buffers()
    return result


def get_num_params(
        module: torch.nn.Module,
) -> int:
    result = 0
    for param in module.parameters():
        result += np.prod(param.shape)
    return result


def log_dag_characteristics(
        dag: dag_module.DagModule,
        input_shape_without_batch: Tuple[int, ...],
):
    dag.clear_custom_buffers()
    if len(input_shape_without_batch) != 3:
        logger.warning(f'One cannot compute `kmapp` for cell: {dag.name}, since the input_shape_without_batch '
                       f'has length less than 2.')
        return
    dag.eval()
    x = torch.ones(size=(1,) + input_shape_without_batch, device=dag.device)
    y = dag(x)
    if isinstance(y, torch.Tensor):
        y = [y]
    static_kmapp = compute_static_kmapp(dag, input_shape_without_batch)
    logger.info(f'static_kmapp: {static_kmapp}')
    num_params = get_num_params(dag) / 1e6
    logger.info(f'number params (M): {num_params}')
    logger.info(f'number of output tensors: {len(y)}')
    for k, tensor in enumerate(y):
        logger.info(f'output shape of output tensor {k}: {tensor.shape}')
    dag.clear_custom_buffers()
