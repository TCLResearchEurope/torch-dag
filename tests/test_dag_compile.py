#
# Copyright © TCL Research Europe. All rights reserved.
#

import pytest
import timm
import torch
from numpy import testing

from dev_tools.algorithms_testing_tools import test_conversion_to_dag as _test_conversion_to_dag
from .test_constants import test_cases, TimmModelTestCase


@pytest.mark.ioheavy
@pytest.mark.parametrize(
    'test_case',
    test_cases,
)
def test_timm_timm_models(
        test_case: TimmModelTestCase,
        tmpdir
):
    timm_model = timm.create_model(test_case.timm_name, pretrained=False)
    timm_model.eval()
    input_shape = timm_model.default_cfg['input_size']
    input_shape = (1,) + input_shape
    dag, msg = _test_conversion_to_dag(timm_model, input_shape=input_shape)
    dag.eval()
    x = torch.randn(size=input_shape)

    with torch.no_grad():
        y0 = dag(x)

    dag.compile()
    with torch.no_grad():
        y1 = dag(x)

    testing.assert_allclose(y0, y1)
