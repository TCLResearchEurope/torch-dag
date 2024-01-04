#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging

import pytest
import timm
import torch

import torch_dag as td
from torch_dag.core.unstructured_to_structured import build_from_unstructured_module
from torch_dag_algorithms.commons.flops_computation import compute_kmapp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

timm_name = 'hardcorenas_a'


def get_toy_dag_and_input_shape():
    model = torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2),
        torch.nn.Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.ReLU(),
    )
    return build_from_unstructured_module(model), (1, 3, 224, 224)


def get_dag_and_input_shape_from_timm(timm_name: str):
    timm_model = timm.create_model(timm_name, pretrained=False)
    timm_model.eval()
    input_shape = timm_model.default_cfg['input_size']
    input_shape = (1,) + input_shape
    dag = build_from_unstructured_module(timm_model)
    return dag, input_shape


@pytest.mark.parametrize(
    ['timm_model'],
    [
        [False],
        [True],
    ],
)
def test_flops_computation_cpu(timm_model: bool):
    cpu_device = torch.device('cpu')
    if timm_model:
        dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    else:
        dag, input_shape = get_toy_dag_and_input_shape()
    dag.to(cpu_device)
    assert dag.device.type == cpu_device.type
    static_kmapp = td.commons.compute_static_kmapp(dag_module=dag, input_shape_without_batch=input_shape[1:])


@pytest.mark.parametrize(
    ['timm_model'],
    [
        [False],
        [True],
    ],
)
@pytest.mark.skipif(DEVICE.type != 'cuda', reason='No cuda device available')
def test_flops_computation_gpu(timm_model: bool):
    gpu_device = DEVICE
    if timm_model:
        dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    else:
        dag, input_shape = get_toy_dag_and_input_shape()
    dag.to(gpu_device)
    assert dag.device.type == gpu_device.type
    static_kmapp = td.commons.compute_static_kmapp(dag_module=dag, input_shape_without_batch=input_shape[1:])


@pytest.mark.parametrize(
    ['timm_model'],
    [
        [False],
        [True],
    ],
)
@pytest.mark.skipif(DEVICE.type != 'cuda', reason='No cuda device available')
def test_flops_list_building_gpu(timm_model: bool):
    gpu_device = DEVICE
    if timm_model:
        dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    else:
        dag, input_shape = get_toy_dag_and_input_shape()
    dag.to(gpu_device)
    full_flops_list = td.commons.build_full_flops_list(dag, input_shape_without_batch=input_shape[1:])
    dynamic_kmapp = compute_kmapp(dag, input_shape_without_batch=input_shape[1:], full_flops_list=full_flops_list)


@pytest.mark.parametrize(
    ['timm_model'],
    [
        [False],
        [True],
    ],
)
def test_flops_list_building_cpu(timm_model: bool):
    cpu_device = torch.device('cpu')
    if timm_model:
        dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    else:
        dag, input_shape = get_toy_dag_and_input_shape()
    dag.to(cpu_device)
    assert dag.device.type == cpu_device.type
    full_flops_list = td.commons.build_full_flops_list(dag, input_shape_without_batch=input_shape[1:])
    dag = dag.flatten(input_shape)
    dag.cache_forward_dict = True
    dag(torch.ones(size=input_shape))
    dynamic_kmapp = compute_kmapp(dag, input_shape_without_batch=input_shape[1:], full_flops_list=full_flops_list)
