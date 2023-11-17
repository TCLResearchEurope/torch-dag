#
# Copyright © TCL Research Europe. All rights reserved.
#

import pytest
import timm
import torch
from numpy import testing

from dev_tools.algorithms_testing_tools import test_conversion_to_dag as _test_conversion_to_dag
from .test_constants import test_cases, TimmModelTestCase

test_cases = [case for case in test_cases if case.test_compile]


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
    dag.compile(inputs=torch.ones(size=input_shape))
    dag_compiled = torch.compile(dag)
    model_compiled = torch.compile(timm_model)
    dag_compiled.eval()
    x = torch.randn(size=input_shape)
    with torch.no_grad():
        y0 = dag(x)
        y1 = dag_compiled(x)
        z0 = timm_model(x)
        z1 = model_compiled(x)

    testing.assert_allclose(y0, y1, rtol=0.005, atol=1e-4)
    testing.assert_allclose(z0, z1, rtol=0.005, atol=1e-4)
