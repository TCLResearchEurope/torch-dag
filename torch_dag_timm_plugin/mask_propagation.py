from functools import singledispatch
from typing import List, Optional, Union

import torch

from torch_dag.core import InnerVertex

try:
    import timm
    from timm.layers.grn import GlobalResponseNorm

    PASS_THROUGH_CHANNELS_TIMM_CLASSES = (
        timm.layers.activations.GELU,
        timm.layers.norm.LayerNorm,
        timm.layers.norm.GroupNorm1,
        timm.layers.norm.GroupNorm,
        timm.layers.norm.LayerNorm2d,
        GlobalResponseNorm,
        timm.layers.pool2d_same.AvgPool2dSame,
        timm.layers.pool2d_same.MaxPool2dSame,
    )
except ImportError:
    PASS_THROUGH_CHANNELS_TIMM_CLASSES = ()


@singledispatch
def compute_timm_module_channels_masks(
        module: torch.nn.Module,
        vertex: InnerVertex,
        predecessors_channels_masks: List[List[torch.Tensor]],
        output_channels_masks: Optional[List[torch.Tensor]] = None,
):
    raise NotImplementedError


CUSTOM_MASKS_TIMM_CLASSES = ()

try:
    from timm.models.vision_transformer import PatchEmbed


    @compute_timm_module_channels_masks.register
    def _(
            module: PatchEmbed,
            vertex: InnerVertex,
            predecessors_channels_masks: List[List[torch.Tensor]],
            output_channels_masks: Optional[List[torch.Tensor]] = None,
    ):
        if vertex.module.flatten:
            return [torch.ones(size=(vertex.dag_module.forward_dict[vertex].shape[2],), dtype=torch.int32)]
        else:
            return None


    CUSTOM_MASKS_TIMM_CLASSES += (PatchEmbed,)

except ImportError:
    pass

try:
    from timm.models.efficientformer_v2 import Attention2d, Attention2dDownsample


    @compute_timm_module_channels_masks.register(Attention2d)
    @compute_timm_module_channels_masks.register(Attention2dDownsample)
    def _(
            module: Union[Attention2d, Attention2dDownsample],
            vertex: InnerVertex,
            predecessors_channels_masks: List[List[torch.Tensor]],
            output_channels_masks: Optional[List[torch.Tensor]] = None,
    ):
        return [torch.ones(size=(vertex.dag_module.forward_dict[vertex].shape[1],), dtype=torch.int32)]


    CUSTOM_MASKS_TIMM_CLASSES += (Attention2d, Attention2dDownsample)

except ImportError:
    pass

try:
    from timm.models.xcit import PositionalEncodingFourier


    @compute_timm_module_channels_masks.register
    def _(
            module: PositionalEncodingFourier,
            vertex: InnerVertex,
            predecessors_channels_masks: List[List[torch.Tensor]],
            output_channels_masks: Optional[List[torch.Tensor]] = None,
    ):
        return [torch.ones(size=(vertex.dag_module.forward_dict[vertex].shape[1],), dtype=torch.int32)]


    CUSTOM_MASKS_TIMM_CLASSES += (PositionalEncodingFourier,)

except ImportError:
    pass
