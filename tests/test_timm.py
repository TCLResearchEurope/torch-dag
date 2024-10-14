#
# Copyright © TCL Research Europe. All rights reserved.
#

from copy import deepcopy

import pytest
import timm
import torch

from dev_tools.algorithms_testing_tools import test_conversion_to_dag as _test_conversion_to_dag, \
    test_orbitalization_and_channel_removal as _test_orbitalization_and_channel_removal
from dev_tools.constants import SUCCESS_RESULT
from .test_constants import TimmModelTestCase, base_test_cases, extra_test_cases


def run_timm_model_test(
        test_case: TimmModelTestCase,
        tmpdir
):
    timm_model = timm.create_model(test_case.timm_name, pretrained=False)
    timm_model.eval()
    input_shape = timm_model.default_cfg['input_size']
    input_shape = (1,) + input_shape

    dag, msg = _test_conversion_to_dag(timm_model, input_shape=input_shape)
    result_channel_pruning = _test_orbitalization_and_channel_removal(
        dag=deepcopy(dag),
        input_shape=input_shape,
        prob_removal=test_case.prob_removal,
    )
    assert result_channel_pruning['status'] == SUCCESS_RESULT
    assert result_channel_pruning['prunable_fraction'] > test_case.channel_pruning_threshold


@pytest.mark.parametrize(
    "test_case",
    base_test_cases,
)
def test_timm_model_light(test_case: TimmModelTestCase, tmpdir):
    run_timm_model_test(test_case, tmpdir)


@pytest.mark.ioheavy
@pytest.mark.parametrize(
    "test_case",
    extra_test_cases,
)
def test_timm_model_heavy(test_case: TimmModelTestCase, tmpdir):
    run_timm_model_test(test_case, tmpdir)

def test_multi_output_model(
):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), torch.nn.ReLU())
            self.head_0 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.ReLU())
            self.head_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 128, 3, 1, 1), torch.nn.ReLU())
            self.head_2 = torch.nn.Sequential(torch.nn.Conv2d(3, 256, 3, 1, 1), torch.nn.ReLU())

        def forward(self, x):
            x = self.input(x)
            out_0 = self.head_0(x)
            out_1 = self.head_1(x)
            out_2 = self.head_2(x)
            return out_0, out_1, out_2

    torch_model = DummyModel()
    torch_model.eval()
    input_shape = (1, 3, 224, 224)
    dag, msg = _test_conversion_to_dag(torch_model, input_shape=input_shape)
    result_channel_pruning = _test_orbitalization_and_channel_removal(
        dag=deepcopy(dag),
        input_shape=input_shape,
        prob_removal=0.5,
    )
    assert result_channel_pruning['status'] == SUCCESS_RESULT
    assert result_channel_pruning['prunable_fraction'] > 0.0