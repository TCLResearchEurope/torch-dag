import random
import string
from abc import ABC
from typing import Tuple

import torch
from numpy import testing

DEFAULT_ATOL = 1e-6
DEFAULT_ATOL_MIXED_PRECISION = 2e-2
DEFAULT_RTOL = 1e-6
DEFAULT_NUM_OF_TESTS = 50


class TestBase(ABC):
    @staticmethod
    def get_random_tensor(shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.normal(mean=torch.zeros(size=shape))

    @staticmethod
    def get_random_string(length=20):
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

    @staticmethod
    def assert_all_equal(x, y):
        return testing.assert_equal(x, y)

    @staticmethod
    def assert_array_equal(x, y):
        return testing.assert_array_equal(x, y)

    @staticmethod
    def assert_all_close(x: torch.Tensor, y: torch.Tensor, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
        return testing.assert_allclose(x.detach(), y.detach(), atol=atol, rtol=rtol)

    @staticmethod
    def assert_true(expression):
        if not expression:
            raise AssertionError
