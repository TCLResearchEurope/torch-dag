#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging

import pytest
import timm
import torch

from torch_dag.core.unstructured_to_structured import build_from_unstructured_module
from .test_constants import test_cases, TimmModelTestCase

test_cases_for_tracing = [case for case in test_cases if case.test_regular_tracing]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    'test_case',
    test_cases_for_tracing,
)
def test_timm_timm_models(
        test_case: TimmModelTestCase,
        tmpdir
):
    timm_model = timm.create_model(test_case.timm_name, pretrained=False)
    timm_model.eval()
    dag = build_from_unstructured_module(timm_model)
    tracer = torch.fx.Tracer()
    traced = tracer.trace(dag)
