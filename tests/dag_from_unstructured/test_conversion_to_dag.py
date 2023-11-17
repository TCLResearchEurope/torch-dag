#
# Copyright © TCL Research Europe. All rights reserved.
#

import pytest
import torch

import timm
from torch_dag.core.unstructured_to_structured import build_from_unstructured_module

DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)

MODELS_AND_INPUT_SHAPES = {
    #'hardcorenas_a': DEFAULT_INPUT_SHAPE,
    #'convmixer_768_32': DEFAULT_INPUT_SHAPE,
    #'convnext_nano': DEFAULT_INPUT_SHAPE, # TODO: fix
    #'convnextv2_nano': DEFAULT_INPUT_SHAPE,
    #'cs3darknet_focus_s': DEFAULT_INPUT_SHAPE,
    #'cspresnet50d': DEFAULT_INPUT_SHAPE,
    #'efficientformerv2_s0': DEFAULT_INPUT_SHAPE,
    # 'efficientnet_lite0': DEFAULT_INPUT_SHAPE,
    #'fbnetv3_b': DEFAULT_INPUT_SHAPE,
    'ghostnet_050': DEFAULT_INPUT_SHAPE,
    'gluon_resnet18_v1b': DEFAULT_INPUT_SHAPE,
    # 'edgenext_xx_small': DEFAULT_INPUT_SHAPE, # TODO: add no trace to some timm layers
    # 'dm_nfnet_f0': DEFAULT_INPUT_SHAPE, # TODO: add no trace to some timm layers
    #'davit_tiny': DEFAULT_INPUT_SHAPE, # TODO: tracing issue
    # 'mixer_s16_224': DEFAULT_INPUT_SHAPE, # TODO: fix pruning here
    # 'resnetv2_50d':                             DEFAULT_INPUT_SHAPE,
    # 'mobilenetv2_050.lamb_in1k':                DEFAULT_INPUT_SHAPE,
    # 'mobilenetv3_large_100.miil_in21k_ft_in1k': DEFAULT_INPUT_SHAPE,
    # 'efficientnet_b1.ft_in1k':                  DEFAULT_INPUT_SHAPE,
    # 'efficientnetv2_rw_m.agc_in1k':             (1, 3, 480, 480), # TODO: too large ;)
}


@pytest.mark.ioheavy
@pytest.mark.parametrize("model", MODELS_AND_INPUT_SHAPES.keys())

def test_conversion_from_timm(
        model: str,
        atol: float = 1e-9
):
    timm_model = timm.create_model(model, pretrained=False)
    timm_model.eval()
    try:
        input_shape = timm_model.default_cfg['input_size']
        input_shape = (1,) + input_shape
    except:
        input_shape = MODELS_AND_INPUT_SHAPES[model]
    dag = build_from_unstructured_module(model=timm_model)
    dag.eval()
    x = torch.normal(mean=torch.ones(size=input_shape))
    with torch.no_grad():
        y0 = timm_model(x)
        y1 = dag(x)
        diff = torch.abs(y0 - y1).max()
        assert diff < atol


