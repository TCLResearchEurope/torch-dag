import logging
from typing import List

import torch
from torch import nn

from torch_dag.core.dag_module import InnerVertex
from torch_dag_algorithms.pruning.channel_removal_primitives import remove_module_channels_base
from torch_dag_timm_plugin.channel_removal import CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES, \
    remove_timm_module_channels

logger = logging.getLogger(__name__)


def remove_module_channels(
        module: nn.Module,
        vertex: InnerVertex,
        predecessors_masks: List[List[torch.Tensor]],
        output_masks: List[torch.Tensor],
):
    if isinstance(module, CUSTOM_TIMM_CHANNEL_REMOVAL_MODULES):
        return remove_timm_module_channels(module, vertex, predecessors_masks, output_masks)
    else:
        return remove_module_channels_base(module, vertex, predecessors_masks, output_masks)
