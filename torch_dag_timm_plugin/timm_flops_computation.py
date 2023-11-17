#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging
from functools import singledispatch
from typing import List, Union

import numpy as np
import timm
import torch
from fvcore.nn import FlopCountAnalysis

logger = logging.getLogger(__name__)

CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES = ()


@singledispatch
def timm_custom_compute_flops(
        module: torch.nn.Module,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
):
    raise NotImplementedError


try:
    from timm.models.gcvit import WindowAttentionGlobal

    CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES += (WindowAttentionGlobal,)


    @timm_custom_compute_flops.register
    def _(
            module: timm.models.gcvit.WindowAttentionGlobal,
            inputs: Union[torch.Tensor, List[torch.Tensor]],
    ):
        # TODO: maybe try to compute the flops manually here in a more detailed way
        flops_analyser = FlopCountAnalysis(module.qkv, inputs[0])
        flops_analyser.uncalled_modules_warnings(False)
        flops_analyser.unsupported_ops_warnings(False)

        return flops_analyser.total()


except ImportError:
    pass

try:
    from timm.models.swin_transformer_v2_cr import WindowMultiHeadAttention

    CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES += (WindowMultiHeadAttention,)


    @timm_custom_compute_flops.register
    def _(
            module: WindowMultiHeadAttention,
            inputs: Union[torch.Tensor, List[torch.Tensor]],
    ):
        # TODO: maybe try to compute the flops manually here in a more detailed way
        if isinstance(inputs, torch.Tensor):
            flops_analyser = FlopCountAnalysis(module, inputs)
        else:
            flops_analyser = FlopCountAnalysis(module, inputs[0])
        flops_analyser.uncalled_modules_warnings(False)
        flops_analyser.unsupported_ops_warnings(False)
        return flops_analyser.total()

except ImportError:
    pass

try:
    from timm.models.swin_transformer_v2 import WindowAttention

    CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES += (WindowAttention,)


    @timm_custom_compute_flops.register
    def _(
            module: WindowAttention,
            inputs: Union[torch.Tensor, List[torch.Tensor]],
    ):
        # TODO: maybe try to compute the flops manually here in a more detailed way
        if isinstance(inputs, torch.Tensor):
            flops_analyser = FlopCountAnalysis(module, inputs)
        else:
            flops_analyser = FlopCountAnalysis(module, inputs[0])
        flops_analyser.uncalled_modules_warnings(False)
        flops_analyser.unsupported_ops_warnings(False)
        return flops_analyser.total()

except ImportError:
    pass

try:
    from timm.models.vision_transformer import LayerScale

    CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES += (LayerScale,)


    @timm_custom_compute_flops.register
    def _(
            module: LayerScale,
            inputs: Union[torch.Tensor, List[torch.Tensor]],
    ):
        return np.prod(inputs.shape[1:])

except ImportError:
    pass

try:
    from timm.models.xcit import PositionalEncodingFourier

    CUSTOM_TIMM_FLOPS_COMPUTATION_MODULES += (PositionalEncodingFourier,)


    @timm_custom_compute_flops.register
    def _(
            module: PositionalEncodingFourier,
            inputs: Union[torch.Tensor, List[torch.Tensor]],
    ):
        return 0

except ImportError:
    pass
