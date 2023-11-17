import logging

import torch
from timm.models.vision_transformer import Attention
from torch import nn

from torch_dag import structured_modules as smodules

logger = logging.getLogger(__name__)

prunable_modules = set()


def register_prunable_module(module: nn.Module):
    """
    Decorator for custom modules which can be pruned inside
    """
    prunable_modules.add(module)
    return module


@register_prunable_module
class BaselineAttention(torch.nn.Module):
    def __init__(
            self,
            dim,
            num_heads: int = 8,
            use_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            use_out_bias: bool = True,
    ):
        super().__init__()
        if use_bias is False:
            logger.warning(f'Not using bias in `BaselineAttention` implementation of attenion will'
                           f'result in issues with `tflite` deployment!')
        self.dim = dim
        self.num_heads = num_heads
        dim_per_head = dim // num_heads
        self.q = torch.nn.ModuleList([torch.nn.Linear(dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])
        self.k = torch.nn.ModuleList([torch.nn.Linear(dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])
        self.v = torch.nn.ModuleList([torch.nn.Linear(dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])
        self.out = torch.nn.Linear(self.num_heads * dim_per_head, dim, bias=use_out_bias)
        self.concat = smodules.ConcatModule(dim=-1)
        self.attention_matmul = smodules.TfMatmulModule(transpose=True, normalize=True)
        self.final_matmul = smodules.TfMatmulModule(transpose=False, normalize=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        carriers = []
        for k in range(self.num_heads):
            x_q = self.q[k](x)
            x_k = self.k[k](x)
            x_v = self.v[k](x)
            qk = self.attention_matmul([x_q, x_k])
            qk = self.softmax(qk)
            qk = self.attn_drop(qk)
            qkv = self.final_matmul([qk, x_v])
            carriers.append(qkv)
        x = self.concat(carriers)
        x = self.out(x)
        x = self.proj_drop(x)
        return x

    @classmethod
    def convert_from_timm(cls, module: Attention):
        dim = module.head_dim * module.num_heads
        hdim = module.head_dim
        result = cls(
            dim=dim,
            num_heads=module.num_heads,
            use_bias=module.qkv.bias is not None,
            use_out_bias=module.proj.bias is not None,
        )
        q_weight = module.qkv.weight[:dim, :]
        k_weight = module.qkv.weight[dim: 2 * dim, :]
        v_weight = module.qkv.weight[2 * dim:3 * dim, :]
        if module.qkv.bias is not None:
            q_bias = module.qkv.bias[:dim]
            k_bias = module.qkv.bias[dim: 2 * dim]
            v_bias = module.qkv.bias[2 * dim:3 * dim]
        for head_id in range(result.num_heads):
            result.q[head_id].weight.data = q_weight[head_id * hdim: (head_id + 1) * hdim, :]
            result.k[head_id].weight.data = k_weight[head_id * hdim: (head_id + 1) * hdim, :]
            result.v[head_id].weight.data = v_weight[head_id * hdim: (head_id + 1) * hdim, :]
            if module.qkv.bias is not None:
                result.q[head_id].bias.data = q_bias[head_id * hdim: (head_id + 1) * hdim]
                result.k[head_id].bias.data = k_bias[head_id * hdim: (head_id + 1) * hdim]
                result.v[head_id].bias.data = v_bias[head_id * hdim: (head_id + 1) * hdim]

        result.out.weight.data = module.proj.weight
        if module.proj.bias is not None:
            result.out.bias = module.proj.bias

        return result


@register_prunable_module
class CrossCovarianceAttention(torch.nn.Module):
    # TODO: add pruning patterns for this
    def __init__(
            self,
            dim,
            num_heads: int = 8,
            use_bias: bool = True,
    ):
        super().__init__()
        if use_bias is False:
            logger.warning(f'Not using bias in `CrossCovarianceAttention` implementation of attenion will'
                           f'result in issues with `tflite` deployment!')
        self.dim = dim
        self.num_heads = num_heads
        dim_per_head = dim // num_heads
        self.q = torch.nn.ModuleList([torch.nn.Linear(dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])
        self.k = torch.nn.ModuleList([torch.nn.Linear(dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])
        self.v = torch.nn.ModuleList([torch.nn.Linear(dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])
        self.out = torch.nn.Linear(self.num_heads * dim_per_head, dim)
        self.concat = smodules.ConcatModule(dim=-1)
        self.attention_matmul = smodules.TfMatmulModule(transpose=True, normalize=False)
        self.final_matmul = smodules.TfMatmulModule(transpose=False, normalize=False)
        self.transpose_0 = smodules.PermuteModule(perm=(0, 2, 1))
        self.transpose_1 = smodules.PermuteModule(perm=(0, 2, 1))
        self.normalize = smodules.NormalizeModule(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperatures = torch.nn.ParameterList([nn.Parameter(torch.ones(1, 1, 1)) for _ in range(self.num_heads)])

    def forward(self, x):
        carriers = []
        for k in range(self.num_heads):
            x_q = self.q[k](x)
            x_k = self.k[k](x)
            x_v = self.v[k](x)
            x_q = self.transpose_0(x_q)
            x_k = self.transpose_0(x_k)
            x_v = self.transpose_0(x_v)

            x_q = self.normalize(x_q)
            x_k = self.normalize(x_k)

            qk = self.attention_matmul([x_q, x_k])
            temperature = self.temperatures[k]
            qk = qk * temperature
            qk = self.softmax(qk)
            qkv = self.final_matmul([qk, x_v])
            qkv = self.transpose_1(qkv)
            carriers.append(qkv)
        x = self.concat(carriers)
        x = self.out(x)
        return x
