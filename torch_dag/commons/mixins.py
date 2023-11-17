from typing import List, Union

import torch


class CustomComputeFlopsMixin:
    def compute_flops(self, inputs: Union[torch.Tensor, List[torch.Tensor]]):
        raise NotImplementedError


class ZeroFlopsMixin(CustomComputeFlopsMixin):
    def compute_flops(self, inputs: Union[torch.Tensor, List[torch.Tensor]]):
        return 0


class CustomCountParamsMixin:
    def count_params(self):
        raise NotImplementedError


class ZeroParamsMixin(CustomCountParamsMixin):
    def count_params(self):
        return 0
