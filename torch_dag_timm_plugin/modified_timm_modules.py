try:
    import timm
except ImportError:
    raise ImportError('`timm` package is missing!')

import logging

import torch
from torch.utils.checkpoint import checkpoint

from torch_dag.core.dag_tracer import register_notrace_module
from torch_dag.core.module_handling_for_pruning import register_unprunable_module

logger = logging.getLogger(__name__)

try:
    from timm.models.beit import Beit
    from timm.models.efficientformer_v2 import Attention2d, Attention2dDownsample
    from timm.models.vgg import ConvMlp


    @register_unprunable_module
    @register_notrace_module
    class ModifiedAttention2d(Attention2d):
        """
        Original implementation with caching of biases does not support deepcopy in some cases
        """

        def get_attention_biases(self, device: torch.device) -> torch.Tensor:
            return self.attention_biases[:, self.attention_bias_idxs]


    @register_unprunable_module
    @register_notrace_module
    class ModifiedAttention2dDownsample(Attention2dDownsample):
        """
        Original implementation with caching of biases does not support deepcopy in some cases
        """

        def get_attention_biases(self, device: torch.device) -> torch.Tensor:
            return self.attention_biases[:, self.attention_bias_idxs]


    class ModifiedBeit(Beit):
        def forward_features(self, x):
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
                else:
                    x = blk(x, shared_rel_pos_bias=rel_pos_bias)
            x = self.norm(x)
            return x


    class ModifiedConvMlp(ConvMlp):
        def forward(self, x):
            x = self.fc1(x)
            x = self.act1(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.act2(x)
            return x


    ALL_MODIFIED_TIMM_MODULES = [
        ModifiedAttention2d,
        ModifiedAttention2dDownsample,
        ModifiedConvMlp,
        ModifiedBeit,
    ]
except ImportError:
    ALL_MODIFIED_TIMM_MODULES = []
