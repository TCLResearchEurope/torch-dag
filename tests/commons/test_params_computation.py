import pytest
import torch

from torch_dag_algorithms.commons.params_computation import build_full_params_list
from torch_dag.core.unstructured_to_structured import build_from_unstructured_module


def build_toy_dag():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        ),
        torch.nn.ReLU(),
    )
    return build_from_unstructured_module(model)


@pytest.mark.parametrize(
    ['normalize', 'expected_params'],
    [
        (False, [108, 0, 296, 0]),
        (True, [0.000108, 0, 0.000296, 0]),
    ],
)
def test__build_full_params_list(normalize, expected_params):
    dag = build_toy_dag()
    actual_params = build_full_params_list(
        dag=dag,
        normalize=normalize,
    )
    assert actual_params == expected_params
