#
# Copyright © TCL Research Europe. All rights reserved.
#
from functools import singledispatch
from typing import List, Dict, Union

import torch

from torch_dag.core.dag_module import InnerVertex, Vertex
from torch_dag_algorithms.pruning.modules import OrbitModule

CUSTOM_AVERAGE_CHANNELS_TIMM_CLASSES = ()


def shape_to_float(shape, device, dim=1):
    return torch.tensor(shape[dim], device=device).to(torch.float32)


@singledispatch
def compute_timm_average_num_channels(
        module: torch.nn.Module,
        vertex: InnerVertex,
        average_number_input_channels: List[List[torch.Tensor]],
        orbits_dict: Dict[str, OrbitModule],
        forward_dict: Dict[Vertex, Union[torch.Tensor, List[torch.Tensor]]]
) -> Union[List[torch.Tensor], None]:
    raise NotImplementedError


try:
    from timm.models.vision_transformer import PatchEmbed

    CUSTOM_AVERAGE_CHANNELS_TIMM_CLASSES += (PatchEmbed,)


    @compute_timm_average_num_channels.register
    def _(
            module: PatchEmbed,
            vertex: InnerVertex,
            average_number_input_channels: List[List[torch.Tensor]],
            orbits_dict: Dict[str, OrbitModule],
            forward_dict: Dict[Vertex, Union[torch.Tensor, List[torch.Tensor]]]
    ) -> Union[List[torch.Tensor], None]:
        device = forward_dict[vertex.dag_module.input_vertices[0]].device
        return [shape_to_float(forward_dict[vertex].shape, dim=2, device=device)]
except ImportError:
    pass
