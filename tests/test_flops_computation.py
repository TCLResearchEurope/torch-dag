#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging

import pytest
import timm
import torch

import torch_dag as td
from torch_dag.core.unstructured_to_structured import build_from_unstructured_module

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

timm_name = 'hardcorenas_a'


def get_dag_and_input_shape_from_timm(timm_name: str):
    timm_model = timm.create_model(timm_name, pretrained=False)
    timm_model.eval()
    input_shape = timm_model.default_cfg['input_size']
    input_shape = (1,) + input_shape
    dag = build_from_unstructured_module(timm_model)
    return dag, input_shape


def test_flops_computation_cpu():
    cpu_device = torch.device('cpu')
    dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    dag.to(cpu_device)
    assert dag.device.type == cpu_device.type
    static_kmapp = td.commons.compute_static_kmapp(dag_module=dag, input_shape_without_batch=input_shape[1:])


@pytest.mark.skipif(DEVICE.type != 'cuda', reason='No cuda device available')
def test_flops_computation_gpu():
    gpu_device = DEVICE
    dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    dag.to(gpu_device)
    assert dag.device.type == gpu_device.type
    static_kmapp = td.commons.compute_static_kmapp(dag_module=dag, input_shape_without_batch=input_shape[1:])


@pytest.mark.skipif(DEVICE.type != 'cuda', reason='No cuda device available')
def test_flops_list_building_gpu():
    gpu_device = DEVICE
    dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    dag.to(gpu_device)
    full_flops_list = td.commons.build_full_flops_list(dag, input_shape_without_batch=input_shape[1:])


def test_flops_list_building_cpu():
    cpu_device = torch.device('cpu')
    dag, input_shape = get_dag_and_input_shape_from_timm(timm_name)
    dag.to(cpu_device)
    assert dag.device.type == cpu_device.type
    full_flops_list = td.commons.build_full_flops_list(dag, input_shape_without_batch=input_shape[1:])
