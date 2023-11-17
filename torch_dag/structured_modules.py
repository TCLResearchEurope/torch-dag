import logging
import math
from typing import List, Tuple, Union, Optional, Dict, Callable, get_args

import torch
import torch.nn.functional as F
from torch import nn

from torch_dag.core.common_tools import per_channel_noise_to_signal_ratio
from torch_dag.core.dag_tracer import register_notrace_module
from torch_dag.core.module_handling_for_pruning import register_unprunable_module

logger = logging.getLogger(__name__)

ACTIVATION_MODULES_T = Union[
    nn.ReLU,
    nn.ReLU6,
    nn.SiLU,
    nn.Softmax,
    nn.Sigmoid,
    nn.Hardswish,
    nn.Hardsigmoid,
    nn.GELU,
    nn.LeakyReLU,
    nn.ELU,
    nn.Tanh,
    nn.Identity,
]

ACTIVATION_MODULES = get_args(ACTIVATION_MODULES_T)  # -ish...


def space_to_depth(x: torch.Tensor, block_size: int):
    output = x.permute(0, 2, 3, 1)
    (batch_size, s_height, s_width, s_depth) = output.size()
    d_depth = s_depth * block_size ** 2
    d_width = s_width // block_size
    d_height = s_height // block_size
    t_1 = output.split(block_size, 2)
    stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
    output = torch.stack(stack, 1)
    output = output.permute(0, 2, 1, 3)
    return output.permute(0, 3, 1, 2)


def depth_to_space(x: torch.Tensor, block_size: int):
    b, c, h, w = x.shape
    out_channels = c // block_size ** 2
    perm = [i + out_channels * j for i in range(out_channels) for j in range(block_size ** 2)]
    x = x[:, perm]
    return torch.nn.functional.pixel_shuffle(x, block_size)


class ActivationModuleBuilder:
    name_to_activation_dict = {
        # 'leaky_relu':   tf.nn.leaky_relu,
        'relu':         torch.nn.ReLU(),
        'relu6':        torch.nn.ReLU6(),
        'sigmoid':      torch.nn.Sigmoid(),
        'tanh':         torch.nn.Tanh(),
        # 'exp':          tf.exp,
        # 'softmax_v2':   tf.nn.softmax,  # added for backward compatibility
        'swish':        torch.nn.SiLU(),
        'hard_swish':   torch.nn.Hardswish(),
        'hard_sigmoid': torch.nn.Hardsigmoid(),
        'identity':     torch.nn.Identity(),
        'none':         torch.nn.Identity(),
        # 'crelu':        tf.nn.crelu,
        # 'elu':          tf.nn.elu,
        # 'gelu':         tf.nn.gelu,
        # 'log_softmax':  tf.nn.log_softmax,
        # 'selu':         tf.nn.selu,
        'silu':         torch.nn.SiLU(),
        # 'softsign':     tf.nn.softsign,
        'keras_relu':   torch.nn.ReLU(),
        # 'gelu':         tf.nn.gelu,
        None:           torch.nn.Identity(),
    }

    @classmethod
    def build_activation_module(cls, activation_name):
        return cls.name_to_activation_dict[activation_name]


@register_notrace_module
class EmptyModule(torch.nn.Module):
    def forward(self, inputs: List[torch.Tensor]):
        return


@register_notrace_module
class AddModule(torch.nn.Module):
    def forward(self, inputs: List[torch.Tensor]):
        if isinstance(inputs, torch.Tensor):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        return inputs[0] + inputs[1]


@register_notrace_module
class SubModule(torch.nn.Module):
    def forward(self, inputs: List[torch.Tensor]):
        return inputs[0] - inputs[1]


@register_notrace_module
class MulModule(torch.nn.Module):
    def forward(self, inputs: List[torch.Tensor]):
        if isinstance(inputs, torch.Tensor):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        return inputs[0] * inputs[1]


@register_notrace_module
class DivModule(torch.nn.Module):
    def forward(self, inputs: List[torch.Tensor]):
        return inputs[0] / inputs[1]


@register_notrace_module
class ConcatModule(torch.nn.Module):
    def __init__(
            self,
            dim: int,
    ):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: List[torch.Tensor]):
        # degenerate case of one input
        if len(inputs) == 1 and isinstance(inputs, List):
            return inputs[0]
        elif isinstance(inputs, torch.Tensor):
            return inputs
        else:
            return torch.cat(inputs, dim=self.dim)


@register_unprunable_module
@register_notrace_module
class PermuteModule(torch.nn.Module):
    def __init__(self, perm: Tuple[int, ...]):
        super().__init__()
        self.perm = perm

    def forward(self, inputs: torch.Tensor):
        return torch.permute(inputs, self.perm)


@register_unprunable_module
@register_notrace_module
class TransposeModule(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, inputs: torch.Tensor):
        return torch.transpose(inputs, self.dim0, self.dim1)


@register_notrace_module
class GlobalMeanPool2DModule(torch.nn.Module):

    def __init__(
            self,
            dim,
            keepdim: bool,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, inputs: torch.Tensor):
        return torch.mean(inputs, dim=self.dim, keepdim=self.keepdim)


@register_unprunable_module
@register_notrace_module
class SpaceToDepthModule(torch.nn.Module):

    def __init__(
            self,
            block_size: int,
    ):
        super().__init__()
        self.block_size = block_size

    def forward(self, inputs: torch.Tensor):
        # https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/11
        return space_to_depth(inputs, self.block_size)


@register_unprunable_module
@register_notrace_module
class ReshapeModule(torch.nn.Module):

    def __init__(
            self,
            target_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, inputs: torch.Tensor, target_shape=None):
        if self.target_shape is not None:
            batch_size = inputs.shape[0]
            return torch.reshape(inputs, shape=(batch_size,) + tuple(self.target_shape))
        else:
            return inputs.reshape(target_shape)


@register_unprunable_module
@register_notrace_module
class ReshapeModuleV2(torch.nn.Module):
    """Similar as in `ReshapeModule` but `target_shape` includes batch size. """

    def __init__(
            self,
            target_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, inputs: torch.Tensor, target_shape=None):
        if self.target_shape is not None:
            return torch.reshape(inputs, tuple(self.target_shape))
        else:
            return inputs.reshape(target_shape)


@register_unprunable_module
@register_notrace_module
class PatchifyModule(torch.nn.Module):

    def forward(self, inputs: torch.Tensor):
        b, c, h, w = inputs.shape
        x = torch.reshape(inputs, (b, c, h * w))
        return x.permute(0, 2, 1)


@register_unprunable_module
@register_notrace_module
class DePatchifyModule(torch.nn.Module):

    def forward(self, inputs: List[torch.Tensor]):
        x, reference_tensor = inputs  # x.shape (b, t, dim)
        b, c, h, w = reference_tensor.shape
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)
        return x


@register_unprunable_module
@register_notrace_module
class TensorMergerModule(torch.nn.Module):
    def forward(self, inputs: List[torch.Tensor]):
        return inputs


@register_notrace_module
class TensorExtractorModule(torch.nn.Module):

    def __init__(
            self,
            index: Union[int, Tuple[int, ...]],
    ):
        super().__init__()
        self.index = index

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]):
        if not isinstance(inputs, (torch.Tensor, list)):
            return inputs
        if isinstance(self.index, int):
            return inputs[self.index]
        else:
            return [x for k, x in inputs if k in self.index]


@register_notrace_module
class Conv2DSameModule(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        # max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0) original
        i = torch.tensor(i).to(torch.int32)
        k = torch.tensor(k).to(torch.int32)
        s = torch.tensor(s).to(torch.int32)
        d = torch.tensor(d).to(torch.int32)
        x = ((torch.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i).to(torch.int32)
        return torch.maximum(x, torch.tensor(0, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, padding)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


@register_unprunable_module
@register_notrace_module
class SliceModule(torch.nn.Module):
    def __init__(self, slice_spec):
        super().__init__()
        self.slice_spec = slice_spec

    @staticmethod
    def replace_ellipses_by_slices(slice_spec):
        """
        `torch.compile` does not support Ellipsis objects
        """
        result = []
        for el in slice_spec:
            if isinstance(el, type(...)):
                result.append(slice(None, None, None))
            else:
                result.append(el)
        return tuple(result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.slice_spec) == 3 == len(x.shape):
            slice_spec = self.replace_ellipses_by_slices(self.slice_spec)
            return x[slice_spec[0], slice_spec[1], slice_spec[2]]
        if len(self.slice_spec) == 4 == len(x.shape):
            slice_spec = self.replace_ellipses_by_slices(self.slice_spec)
            return x[slice_spec[0], slice_spec[1], slice_spec[2], slice_spec[2]]

        return x[self.slice_spec]


@register_notrace_module
class GetShapeModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(x.shape)


@register_notrace_module
class GetShapeModuleV2(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.shape


@register_notrace_module
class TfMatmulModule(torch.nn.Module):
    def __init__(self, transpose: bool, normalize: bool = True):
        super().__init__()
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x, y = x
        dim = x.shape[-1]
        if self.transpose:
            y = torch.permute(y, (0, 2, 1))
        output = torch.matmul(x, y)
        if self.normalize:
            output /= torch.sqrt(torch.tensor(dim).to(torch.float32))
        return output


@register_unprunable_module
@register_notrace_module
class MatmulModule(torch.nn.Module):

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x, y = x
        return x @ y


@register_notrace_module
class ChannelAffineModule(torch.nn.Module):
    def __init__(self, num_channels: int, use_bias: bool, weight_init_value: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.weight = torch.nn.Parameter(data=torch.ones(size=(num_channels,)) * weight_init_value)
        if self.use_bias:
            self.bias = torch.nn.Parameter(data=torch.zeros(size=(num_channels,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            shape = (1, self.num_channels, 1, 1)
        elif len(x.shape) == 3:
            shape = (1, 1, self.num_channels)
        else:
            raise NotImplementedError
        x = self.weight.view(shape) * x
        if self.use_bias:
            x += self.bias.view(shape)
        return x


@register_unprunable_module
@register_notrace_module
class TfTokenizeModule(torch.nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size > 1:
            x = space_to_depth(x, self.patch_size)
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W)
            x = x.permute(0, 2, 1)
            return x
        else:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            return torch.reshape(x, (B, H * W, C))


@register_unprunable_module
@register_notrace_module
class TfDetokenizeModule(torch.nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # reshape with predecessor shape
        if isinstance(x, List):
            tensor, reference_tensor = x
            b, c, h, w = reference_tensor.shape
            if self.patch_size == 1:
                return tensor.permute(0, 2, 1).reshape(b, c, h, w)
            else:
                b, c, h, w = reference_tensor.shape
                x = tensor
                b, t, dim = x.shape
                x = x.reshape(b, h // self.patch_size, w // self.patch_size, dim)
                x = x.permute(0, 3, 1, 2)
                return depth_to_space(x, self.patch_size)
        else:
            if self.patch_size == 1:
                B, N, C = x.shape
                H = W = math.isqrt(N)
                return x.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                b, t, dim = x.shape
                h = w = math.isqrt(t)
                x = x.reshape(b, h, w, dim)
                x = x.permute(0, 3, 1, 2)
                return depth_to_space(x, self.patch_size)


@register_notrace_module
class TfBatchNorm1d(torch.nn.Module):
    def __init__(self, bn: torch.nn.BatchNorm1d):
        super().__init__()
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3
        B, N, C = x.shape
        x = torch.permute(x, (0, 2, 1))
        x = self.bn(x)
        x = torch.permute(x, (0, 2, 1))
        return x


@register_notrace_module
class ScalarMul(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scalar * x


@register_notrace_module
class ParameterModule(torch.nn.Module):
    def __init__(self, param: torch.nn.Parameter):
        super().__init__()
        self.param = param

    def forward(self, x) -> torch.Tensor:
        return self.param


@register_notrace_module
class NormModule(torch.nn.Module):
    # wraps torch.norm https://pytorch.org/docs/stable/generated/torch.norm.html
    def __init__(self, p: str = 'fro', dim=None, keepdim=False):
        super().__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x) -> torch.Tensor:
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)


@register_notrace_module
class MeanModule(torch.nn.Module):
    # wraps torch.mean
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x) -> torch.Tensor:
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


@register_notrace_module
class SumModule(torch.nn.Module):
    # wraps torch.sum
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x) -> torch.Tensor:
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)


@register_unprunable_module
@register_notrace_module
class SplitModule(torch.nn.Module):
    def __init__(self, split_size_or_sections, dim=0):
        super().__init__()
        if not isinstance(split_size_or_sections, int):
            self.split_size_or_sections = tuple(split_size_or_sections)  # this is a tuple so that `torch.compile` works
        else:
            self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, x) -> List[torch.Tensor]:
        return list(torch.split(x, split_size_or_sections=self.split_size_or_sections, dim=self.dim))


@register_unprunable_module
@register_notrace_module
class ReshapeWithSpecModule(torch.nn.Module):
    """
    TODO: this is deprecated, try to remove this
    """
    SPECS = (
        '(B,C,H,W)->(B,H*W,C)',
        '(B,N,C)->(B,N,target)',
    )
    PREDECESSOR_KEYWORD = 'predecessor'

    def __init__(
            self,
            spec: Union[str, Dict],
            target_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.spec = spec
        self.target_shape = target_shape

    def clear_custom_buffers(self):
        self._buffers.clear()

    def forward(self, x) -> torch.Tensor:
        if self.spec == self.SPECS[0]:
            B, C, H, W = x.shape
            return x.reshape(B, H * W, C)
        elif self.spec == self.SPECS[1]:
            B, N, C = x.shape
            return x.reshape(B, N, *self.target_shape)
        elif isinstance(self.spec, dict):
            if isinstance(x, torch.Tensor):
                target_shape = tuple(self.spec.values())
                return torch.reshape(x, target_shape)
            else:
                x, *input_shapes = x
            if hasattr(self, 'compiled_target_shape'):
                return torch.reshape(x, tuple([int(e) for e in self.compiled_target_shape]))
            pd_shape_counter = 0
            target_shape_list = []
            for k, v in self.spec.items():
                if v == self.PREDECESSOR_KEYWORD:
                    target_shape_list.append(input_shapes[pd_shape_counter])
                    pd_shape_counter += 1
                else:
                    target_shape_list.append(int(v))
            if not hasattr(self, 'compiled_target_shape'):
                compiled_target_shape_tensor = torch.tensor(tuple([int(e) for e in target_shape_list]), device=x.device)
                self.register_buffer('compiled_target_shape', compiled_target_shape_tensor, persistent=False)
            return torch.reshape(x, tuple([int(e) for e in self.compiled_target_shape]))
        else:
            raise NotImplementedError


@register_unprunable_module
@register_notrace_module
class ReshapeWithSpecModuleV2(torch.nn.Module):
    SPECS = (
        '(B,C,H,W)->(B,H*W,C)',
        '(B,N,C)->(B,N,target)',
    )
    PREDECESSOR_KEYWORD = 'predecessor'

    def __init__(
            self,
            spec: Union[str, Dict],
            target_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.spec = spec
        self.target_shape = target_shape

    def forward(self, x) -> torch.Tensor:
        if self.spec == self.SPECS[0]:
            B, C, H, W = x.shape
            return x.reshape(B, H * W, C)
        elif self.spec == self.SPECS[1]:
            B, N, C = x.shape
            return x.reshape(B, N, *self.target_shape)
        elif isinstance(self.spec, dict):
            if isinstance(x, torch.Tensor):
                target_shape = tuple(self.spec.values())
                return torch.reshape(x, target_shape)
            else:
                x, *input_shapes = x
            pd_shape_counter = 0
            target_shape_list = []
            for k, v in self.spec.items():
                if v == self.PREDECESSOR_KEYWORD:
                    target_shape_list.append(input_shapes[pd_shape_counter])
                    pd_shape_counter += 1
                else:
                    target_shape_list.append(v)
            return torch.reshape(x, target_shape_list)
        else:
            raise NotImplementedError


@register_unprunable_module
@register_notrace_module
class TokenizeModule(torch.nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size > 1:
            x = space_to_depth(x, self.patch_size)
            B, C, H, W = x.shape
            x = torch.reshape(x, (B, C, H * W))
            return x.permute(0, 2, 1)
        else:
            B, C, H, W = x.shape
            x = torch.reshape(x, (B, C, H * W))
            return x.permute(0, 2, 1)


@register_unprunable_module
@register_notrace_module
class DetokenizeModule(torch.nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size > 1:
            raise NotImplementedError
        else:
            B, N, C = x.shape
            H = W = math.isqrt(N)
            return x.permute(0, 2, 1).reshape(B, C, H, W)


@register_notrace_module
class UnbindModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return torch.unbind(x, dim=self.dim)


@register_unprunable_module
@register_notrace_module
class ChunkModule(torch.nn.Module):
    def __init__(self, chunks, dim: int):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(torch.chunk(x, self.chunks, self.dim))


@register_notrace_module
class AuxiliaryTokenModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.token = torch.nn.Parameter(data=torch.zeros(size=(dim,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        token_ = self.token.view(1, 1, -1).expand(bs, -1, -1)
        return torch.cat([token_, x], dim=1)


@register_unprunable_module
@register_notrace_module
class ExpandAsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        input, target = x
        return input.expand_as(target)


@register_unprunable_module
@register_notrace_module
class FlattenModule(torch.nn.Module):
    def __init__(self, start_dim: int = 0, end_dim: int = - 1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


@register_unprunable_module
@register_notrace_module
class DepthToSpaceModule(torch.nn.Module):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depth_to_space(x, self.block_size)


@register_unprunable_module
@register_notrace_module
class SpaceToDepthModule(torch.nn.Module):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return space_to_depth(x, block_size=self.block_size)


@register_notrace_module
class InterpolateModule(torch.nn.Module):
    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,

        )


@register_notrace_module
class NormalizeModule(torch.nn.Module):
    def __init__(self, p=2.0, dim=1, ):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim)


@register_notrace_module
class PadModule(torch.nn.Module):
    def __init__(self, pad: tuple, mode: str = 'constant', value: int = None):
        super().__init__()

        assert mode in ['constant', 'reflect', 'replicate',
                        'circular'], f'mode has to be one of [constant, reflect, replicate, circular]. For more information refer to https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html'

        self.pad = pad
        self.mode = mode
        self.value = value

    def forward(self, input):
        return torch.nn.functional.pad(input, self.pad, self.mode, self.value)


@register_notrace_module
class LayerNormWithOptionalBias(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@register_notrace_module
class GeluGPT(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@register_notrace_module
class PositionalEmbeddingGPT(nn.Module):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        return torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)


@register_notrace_module
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            logger.warning("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


@register_notrace_module
class BilinearUpsampling(nn.Module):
    # https://github.com/pytorch/pytorch/issues/10604
    # half_pixel_centers = False and align_corners = False
    def __init__(self, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        scale_factor = self.scale_factor
        b, c, h, w = x.shape
        assert h == w

        N = int(h * scale_factor)
        delta = (1.0 / h)
        p = int(scale_factor) - 1

        xs = torch.linspace(-1.0 + delta, 1.0 - delta, N - p)
        ys = torch.linspace(-1.0 + delta, 1.0 - delta, N - p)
        grid = torch.meshgrid(xs, ys)
        gridy = grid[1]
        gridx = grid[0]
        gridx = torch.nn.functional.pad(gridx.unsqueeze(0), (0, p, 0, p), mode='replicate')[0]
        gridy = torch.nn.functional.pad(gridy.unsqueeze(0), (0, p, 0, p), mode='replicate')[0]
        grid = torch.stack([gridy, gridx], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1).to(x.device)
        output = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros')

        return output


@register_notrace_module
class EfficientAttention(nn.Module):
    """
    An implementation of attention node based on https://arxiv.org/abs/2102.12122 and https://arxiv.org/abs/2105.15203
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 output_dropout_rate: float = 0.0,
                 include_reshapings: bool = False,
                 ):
        super().__init__()
        assert dim % num_heads == 0

        self.query = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=use_bias)
        self.key = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=use_bias)
        self.value = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=use_bias)
        # output projection
        self.output = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=use_bias)

        self.dropout_rate = dropout_rate
        self.output_dropout_rate = output_dropout_rate

        self.include_reshapings = include_reshapings
        self.num_heads = num_heads
        self.dim = dim

    def forward(self, x: List[torch.Tensor]):
        q, k, v = x
        B, C_q, H_q, W_q = q.size()
        B, C_k, H_k, W_k = k.size()

        if self.include_reshapings:
            q = q.reshape(-1, C_q, H_q * W_q, 1)
            k = k.reshape(-1, C_k, H_k * W_k, 1)
            v = v.reshape(-1, C_k, H_k * W_k, 1)

        B, T, C, _ = q.size()

        q_emb = self.query(q)
        q_emb = q_emb.permute((0, 3, 2, 1))
        q_emb = torch.reshape(q_emb, (B, -1, self.num_heads, self.dim // self.num_heads))
        q_emb = torch.permute(q_emb, dims=(0, 2, 1, 3))

        v_emb = self.value(v)
        v_emb = v_emb.permute((0, 3, 2, 1))
        v_emb = torch.reshape(v_emb, (B, -1, self.num_heads, self.dim // self.num_heads))
        v_emb = torch.permute(v_emb, dims=(0, 2, 1, 3))

        k_emb = self.key(k)
        k_emb = k_emb.permute((0, 3, 2, 1))
        k_emb = torch.reshape(k_emb, (B, -1, self.num_heads, self.dim // self.num_heads))
        k_emb = torch.permute(k_emb, dims=(0, 2, 3, 1))

        qk = (q_emb @ k_emb) / (self.dim // self.num_heads) ** (1 / 2)
        scores = torch.nn.functional.softmax(qk, dim=-1)

        if self.training:
            scores = torch.nn.functional.dropout(scores, p=self.dropout_rate)

        sv = scores @ v_emb
        sv = torch.permute(sv, dims=(0, 2, 1, 3))
        sv = sv.reshape((B, 1, -1, self.dim))
        sv = torch.permute(sv, dims=(0, 3, 2, 1))

        out = self.output(sv)
        if self.training:
            out = torch.nn.functional.dropout(out, p=self.output_dropout_rate)

        if self.include_reshapings:
            out = out.reshape(B, C_q, H_q, W_q)

        return out


@register_notrace_module
class AdjustableQueryKeyMatmul(nn.Module):

    def __init__(self, in_features: int, out_features):
        super().__init__()
        self.query = nn.Linear(in_features=in_features, out_features=out_features)
        self.key = nn.Linear(in_features=in_features, out_features=out_features)
        self.matmul = TfMatmulModule(transpose=True, normalize=True)
        self.query_projection = nn.Linear(in_features=in_features, out_features=out_features)
        self.key_projection = nn.Linear(in_features=in_features, out_features=out_features)
        self.loss = None
        self.logits = nn.Parameter(data=3.0 * torch.ones(size=(self.query.out_features,)))

    def forward(self, x):
        q_out = self.query(x)
        key_out = self.key(x)
        attention = self.matmul([q_out, key_out])
        mask = self.sample_from_logits(logits=self.logits)
        q_projected = self.query_projection(x) * mask
        k_projected = self.key_projection(x) * mask
        attention_projected = self.matmul([q_projected, k_projected])
        if len(x.shape) == 3:
            non_channel_dim = (0, 1)
        else:
            raise NotImplementedError
        self.loss = per_channel_noise_to_signal_ratio(y=attention, x=attention_projected,
                                                      non_channel_dim=non_channel_dim)
        return attention

    @staticmethod
    def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
        logits_ = torch.cat([logits[:, None], torch.zeros_like(logits)[:, None]], dim=1)
        gs_sample = torch.nn.functional.gumbel_softmax(
            logits_,
            tau=0.5,
            hard=False,
        )[:, 0]
        return torch.where(logits < 0.0, 0.0, gs_sample)

    @property
    def proportion(self):
        return torch.nn.functional.sigmoid(self.logits).mean()

    def entropy_loss(self, epsilon: float = 0.01):
        probs_ = torch.sigmoid(self.logits)[..., None]
        probs = torch.cat([probs_, 1. - probs_], axis=1)
        return torch.maximum(- (probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon))

    @property
    def trainable_params(self):
        return list(self.query_projection.parameters()) + list(self.key_projection.parameters()) + [self.logits]

    def fuse(self):
        indices = torch.where(self.logits > 0)[0]
        new_weight_query = torch.take_along_dim(self.query_projection.weight, dim=0, indices=indices.view(-1, 1))
        self.query_projection.weight.data = new_weight_query
        if self.query_projection.bias is not None:
            new_bias_query = torch.take_along_dim(self.query_projection.bias, dim=0, indices=indices)
            self.query_projection.bias.data = new_bias_query

        new_weight_key = torch.take_along_dim(self.key_projection.weight, dim=0, indices=indices.view(-1, 1))
        self.key_projection.weight.data = new_weight_key
        if self.key_projection.bias is not None:
            new_bias_key = torch.take_along_dim(self.key_projection.bias, dim=0, indices=indices)
            self.key_projection.bias.data = new_bias_key


@register_notrace_module
class PreFusedAdjustableQueryKeyMatmul(nn.Module):

    def __init__(self, in_features: int, out_features, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(in_features=in_features, out_features=out_features)
        self.key = nn.Linear(in_features=in_features, out_features=out_features)
        self.matmul = TfMatmulModule(transpose=True, normalize=True)
        self.query_projection = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.key_projection = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.loss = None

    def forward(self, x):
        q_out = self.query(x)
        key_out = self.key(x)
        attention = self.matmul([q_out, key_out])
        q_projected = self.query_projection(x)
        k_projected = self.key_projection(x)
        attention_projected = self.matmul([q_projected, k_projected])
        if len(x.shape) == 3:
            non_channel_dim = (0, 1)
        else:
            raise NotImplementedError
        self.loss = per_channel_noise_to_signal_ratio(y=attention, x=attention_projected,
                                                      non_channel_dim=non_channel_dim)
        return attention

    @property
    def trainable_params(self):
        return list(self.query_projection.parameters()) + list(self.key_projection.parameters())


@register_notrace_module
class FusedAdjustableQueryKeyMatmul(nn.Module):

    def __init__(self, in_features: int, out_features):
        super().__init__()
        self.query = nn.Linear(in_features=in_features, out_features=out_features)
        self.key = nn.Linear(in_features=in_features, out_features=out_features)
        self.matmul = TfMatmulModule(transpose=True, normalize=True)

    def forward(self, x):
        q_out = self.query(x)
        key_out = self.key(x)
        attention = self.matmul([q_out, key_out])
        return attention


@register_notrace_module
class HalfPixelCentersFalseBilinearUpsample(nn.Module):
    def __init__(self, scale_factor: int, align_corners: bool = False):
        super(HalfPixelCentersFalseBilinearUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        src_n, src_c, src_h, src_w = x.shape
        dst_n, dst_c, dst_h, dst_w = src_n, src_c, self.scale_factor * src_h, self.scale_factor * src_w

        if src_h == dst_h and src_w == dst_w:
            return x.copy()

        hd = torch.arange(0, dst_h).to(x.device)
        wd = torch.arange(0, dst_w).to(x.device)

        if self.align_corners:
            h = float(src_h) / dst_h * (hd + 0.5) - 0.5
            w = float(src_w) / dst_w * (wd + 0.5) - 0.5
        else:
            h = float(src_h) / dst_h * hd
            w = float(src_w) / dst_w * wd

        h = torch.clamp(h, 0, src_h - 1)
        w = torch.clamp(w, 0, src_w - 1)

        h = h.view(dst_h, 1)
        w = w.view(1, dst_w)

        h = h.repeat(1, dst_w)
        w = w.repeat(dst_h, 1)

        h0 = torch.clamp(torch.floor(h), 0, src_h - 2)
        w0 = torch.clamp(torch.floor(w), 0, src_w - 2)

        h0 = h0.long().to(x.device)
        w0 = w0.long().to(x.device)
        h1 = h0 + 1
        w1 = w0 + 1

        q00 = x[..., h0, w0]
        q01 = x[..., h0, w1]
        q10 = x[..., h1, w0]
        q11 = x[..., h1, w1]

        r0 = (w1 - w) * q00 + (w - w0) * q01
        r1 = (w1 - w) * q10 + (w - w0) * q11

        dst = (h1 - h) * r0 + (h - h0) * r1

        return dst


@register_unprunable_module
@register_notrace_module
class MakeHeadsModule(torch.nn.Module):
    def __init__(
            self,
            num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, input: torch.Tensor):
        # original shape is (B, T, dim)
        assert len(input.shape) == 3
        b, t, dim = input.shape
        x = input.reshape(b, t, dim // self.num_heads, self.num_heads)
        x = x.permute(0, 3, 1, 2)
        return x


@register_unprunable_module
@register_notrace_module
class UnmakeHeadsModule(torch.nn.Module):
    def __init__(
            self,
            num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, input: torch.Tensor):
        # original shape is (B, heads, T, dim_per_head)
        assert len(input.shape) == 4
        b, h, t, dim = input.shape
        x = input.permute(0, 2, 3, 1)
        return x.reshape(b, t, dim * self.num_heads)


@register_notrace_module
class SparseAdjustableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.original_linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.sparse_linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.logits = nn.Parameter(data=3.0 * torch.ones(size=(in_features, out_features)))
        self.loss = None

    @property
    def trainable_params(self):
        return list(self.sparse_linear.parameters()) + [self.logits]

    @property
    def non_logits_params(self):
        return list(self.sparse_linear.parameters())

    @staticmethod
    def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clip(logits, -6.0, 6.0)
        c_in, c_out = logits.shape
        logits_ = logits.view(-1, )
        logits_ = torch.cat([logits_[:, None], torch.zeros_like(logits_)[:, None]], dim=1)
        gs_sample = torch.nn.functional.gumbel_softmax(
            logits_,
            tau=0.5,
            hard=False,
        )[:, 0]
        gs_sample = gs_sample.view(c_in, c_out)
        return torch.where(gs_sample > 0., gs_sample, 0.0)

    @property
    def proportion(self):
        return torch.nn.functional.sigmoid(torch.flatten(self.logits)).mean()

    def forward(self, x):
        y = self.original_linear(x)
        weight_mask = self.sample_from_logits(self.logits)
        weight_masked = self.sparse_linear.weight * weight_mask.permute(1, 0)

        y0 = torch.matmul(x, weight_masked.permute(1, 0)) + self.sparse_linear.bias
        if len(x.shape) == 3:
            non_channel_dim = (0, 1)
        elif len(x.shape) == 4:
            non_channel_dim = (0, 1, 2)
        else:
            raise NotImplementedError
        self.loss = per_channel_noise_to_signal_ratio(y=y, x=y0, non_channel_dim=non_channel_dim)
        return y

    def entropy_loss(self, epsilon: float = 0.01):
        probs_ = torch.sigmoid(torch.flatten(self.logits))[:, None]
        probs = torch.cat([probs_, 1. - probs_], axis=1)
        return torch.maximum(- (probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon))

    def fuse(self):
        indices = torch.where(self.logits > 0)[0]
        print(f'Leaving: {len(indices) / len(self.logits)}')
        indices_lin0 = indices.view(-1, 1)
        new_weight_lin0 = torch.take_along_dim(self.lin0.weight, dim=0, indices=indices_lin0)
        self.lin0.weight.data = new_weight_lin0
        if self.lin0.bias is not None:
            new_bias_lin0 = torch.take_along_dim(self.lin0.bias, dim=0, indices=indices)
            self.lin0.bias.data = new_bias_lin0

        indices_lin1 = indices.view(1, -1)
        new_weight_lin1 = torch.take_along_dim(self.lin1.weight, dim=1, indices=indices_lin1)
        self.lin1.weight.data = new_weight_lin1
        return self.lin0, self.lin1


@register_notrace_module
class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.mask = nn.Parameter(
            data=torch.ones(size=(in_features, out_features), device=self.weight.device),
            requires_grad=False)

    def forward(self, x):
        weight_masked = self.weight * self.mask.permute(1, 0)
        y = torch.matmul(x, weight_masked.permute(1, 0))
        if self.bias is not None:
            y += self.bias
        return y


@register_notrace_module
class DecomposedSparseLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.original_linear = nn.Linear(in_features=in_features, out_features=out_features)
        min_features = min(in_features, out_features)
        self.l0 = nn.Linear(in_features=in_features, out_features=min_features, bias=False)
        self.l1 = nn.Linear(in_features=min_features, out_features=out_features, bias=True)
        self.logits = nn.Parameter(data=3.0 * torch.ones(size=(in_features, out_features)))
        self.loss = None

    @property
    def trainable_params(self):
        return list(self.l0.parameters()) + list(self.l1.parameters()) + [self.logits]

    @staticmethod
    def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clip(logits, -6.0, 6.0)
        c_in, c_out = logits.shape
        logits_ = logits.view(-1, )
        logits_ = torch.cat([logits_[:, None], torch.zeros_like(logits_)[:, None]], dim=1)
        gs_sample = torch.nn.functional.gumbel_softmax(
            logits_,
            tau=0.5,
            hard=False,
        )[:, 0]
        gs_sample = gs_sample.view(c_in, c_out)
        return torch.where(gs_sample > 0., gs_sample, 0.0)

    @property
    def proportion(self):
        return torch.nn.functional.sigmoid(torch.flatten(self.logits)).mean()

    def forward(self, x):
        y = self.original_linear(x)
        weight_mask = self.sample_from_logits(self.logits)
        weight = self.l1.weight @ self.l0.weight
        weight_masked = weight * weight_mask.permute(1, 0)

        y0 = torch.matmul(x, weight_masked.permute(1, 0)) + self.l1.bias
        if len(x.shape) == 3:
            non_channel_dim = (0, 1)
        elif len(x.shape) == 4:
            non_channel_dim = (0, 1, 2)
        else:
            raise NotImplementedError
        self.loss = per_channel_noise_to_signal_ratio(y=y, x=y0, non_channel_dim=non_channel_dim)
        return y

    def entropy_loss(self, epsilon: float = 0.01):
        probs_ = torch.sigmoid(torch.flatten(self.logits))[:, None]
        probs = torch.cat([probs_, 1. - probs_], axis=1)
        return torch.maximum(- (probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon))


@register_notrace_module
class StateSpaceAttentionV2(torch.nn.Module):
    def __init__(
            self,
            dim,
            num_ss_tokens: int,
            s_ratio: int = 4,
            use_bias: bool = True,
            activation=nn.ReLU(),
    ):
        super().__init__()
        if use_bias is False:
            logger.warning(f'Not using bias in `BaselineAttention` implementation of attenion will'
                           f'result in issues with `tflite` deployment!')
        self.dim = dim
        self.activation = activation
        self.num_ss_tokens = num_ss_tokens
        self.inner_dim = dim // s_ratio
        self.query = torch.nn.Linear(dim, dim, bias=use_bias)
        self.key = torch.nn.Linear(dim, num_ss_tokens * self.inner_dim, bias=use_bias)
        self.value = torch.nn.Linear(dim, dim, bias=use_bias)
        self.ss = torch.nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.ss_out = torch.nn.Linear(self.inner_dim, dim, bias=use_bias)

    def forward(self, x):
        # K is the number of state space tokens
        ss_tokens = x[:, -self.num_ss_tokens:, :]  # (B, K, dim)
        # projected_ss_tokens = self.ss(ss_tokens) # (B, K, inner_dim)
        regular_tokens = x[:, :-self.num_ss_tokens, :]  # (B, T, dim)
        x_v = self.value(regular_tokens)
        x_q = self.query(regular_tokens)  # (B, T, K * inner_dim)
        x_k = self.key(regular_tokens)
        b, t, dim = regular_tokens.shape

        ss_update_ = torch.mean(x_k, dim=1, keepdim=True).view(b, self.num_ss_tokens,
                                                               self.inner_dim)  # (B, K, inner_dim)
        ss_update = self.ss_out(ss_update_)
        new_ss_tokens = ss_tokens + ss_update  # (B, K, dim)
        similarity = torch.matmul(x_q, new_ss_tokens.permute(0, 2, 1)) \
                     / torch.sqrt(torch.tensor(self.dim, dtype=x.dtype, device=x.device))  # (B, T, K)
        normalized_similarity = torch.softmax(similarity, dim=-1)  # (B, T, K)
        regular_tokens_update = torch.matmul(normalized_similarity, new_ss_tokens)  # (B, T, dim)
        new_regular_tokens = self.activation(x_v + regular_tokens_update)
        y = torch.cat([new_regular_tokens, new_ss_tokens], dim=1)
        return y


@register_notrace_module
class DecomposedConv(nn.Module):
    """
    A module for learnable `nn.Conv2d` decomposition. The basic idea is as follows:
    - given a `conv` with kernel size, sya 3x3,  in chanels `in` and out channels `out` we train to convolutions `conv1`
    and `conv2`
    (B, in, H, W) -> [conv1](ks=1x1) -> (B, middle_features, H, W) -> [conv2](ks=3x3) -> (B, out, H, W)
    to mimmick the output of the original `conv`. here, `middle_features` << min(in, out). We get a FLOPs reduction by
    trying to minimize `middle_features`, which is done by masking and differentiably penalizing the size of `middle_features`
    """

    def __init__(self, in_features: int, out_features: int, ks: int, padding, stride, bias: bool):
        super().__init__()
        self.middle_features = min(out_features, in_features)
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=ks, padding=padding, stride=stride,
            bias=bias)
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=self.middle_features, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.middle_features, out_channels=out_features, kernel_size=ks,
                               padding=padding, stride=stride, bias=bias)

        self.logits = nn.Parameter(data=3.0 * torch.ones(size=(self.middle_features,)))
        self.loss = None

    @property
    def non_logits_params(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    @property
    def trainable_params(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + [self.logits]

    @staticmethod
    def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
        logits_ = torch.cat([logits[:, None], torch.zeros_like(logits)[:, None]], dim=1)
        gs_sample = torch.nn.functional.gumbel_softmax(
            logits_,
            tau=0.5,
            hard=False,
        )[:, 0]
        return torch.where(logits < 0.0, 0.0, gs_sample)

    @property
    def proportion(self) -> torch.Tensor:
        return torch.nn.functional.sigmoid(self.logits).mean()

    def forward(self, x):
        y0 = self.conv(x)
        mask = self.sample_from_logits(self.logits)
        z = self.conv1(x)
        z = mask.view(1, -1, 1, 1) * z
        z = self.conv2(z)
        self.loss = per_channel_noise_to_signal_ratio(y=y0, x=z, non_channel_dim=(0, 2, 3))
        return y0

    def entropy_loss(self, epsilon: float = 0.01) -> torch.Tensor:
        probs_ = torch.sigmoid(self.logits)[..., None]
        probs = torch.cat([probs_, 1. - probs_], axis=1)
        return torch.maximum(- (probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon))

    def fuse(self) -> Tuple[nn.Conv2d, nn.Conv2d]:
        indices = torch.where(self.logits > 0)[0]
        if len(indices) == 0:
            max_logit = self.logits.max()
            indices = torch.where(self.logits >= max_logit)[0]
        logger.info(f'Leaving: {len(indices) / len(self.logits)}')
        indices_conv1 = indices.view(-1, 1, 1, 1)
        indices_conv2 = indices.view(1, -1, 1, 1)
        new_weight_conv1 = torch.take_along_dim(self.conv1.weight, dim=0, indices=indices_conv1)
        self.conv1.weight.data = new_weight_conv1
        self.conv1.out_channels = len(indices)
        self.conv2.in_channels = len(indices)
        new_weight_conv2 = torch.take_along_dim(self.conv2.weight, dim=1, indices=indices_conv2)
        self.conv2.weight.data = new_weight_conv2
        return self.conv1, self.conv2

    @classmethod
    def build_from_conv(cls, module: nn.Conv2d) -> "DecomposedConv":
        new_module = cls(
            in_features=module.in_channels,
            out_features=module.out_channels,
            ks=module.kernel_size[0],
            padding=module.padding,
            stride=module.stride,
            bias=module.bias is not None
        )
        new_module.conv.weight.data = module.weight
        if module.bias is not None:
            new_module.conv.bias.data = module.bias
        return new_module


@register_notrace_module
class DecomposedLinear(nn.Module):
    """
        A module for learnable `nn.Linear` decomposition. The basic idea is as follows:
        - given a `linear` with in chanels `in` and out channels `out` we train two linear layers `lin0`
        and `lin1`
        (B, in) -> [lin0] -> (B, middle_features) -> [lin1] -> (B, out)
        to mimmick the output of the original `conv`. here, `middle_features` << min(in, out). We get a FLOPs reduction by
        trying to minimize `middle_features`, which is done by masking and differentiably penalizing the size of `middle_features`
        """

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.hidden_features = min(in_features, out_features)
        self.original_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.lin0 = nn.Linear(in_features=in_features, out_features=self.hidden_features, bias=False)
        self.lin1 = nn.Linear(in_features=self.hidden_features, out_features=out_features, bias=bias)
        self.logits = nn.Parameter(data=3.0 * torch.ones(size=(self.hidden_features,)))
        self.loss = None

    @property
    def trainable_params(self):
        return list(self.lin0.parameters()) + list(self.lin1.parameters()) + [self.logits]

    @staticmethod
    def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
        logits_ = torch.cat([logits[:, None], torch.zeros_like(logits)[:, None]], dim=1)
        gs_sample = torch.nn.functional.gumbel_softmax(
            logits_,
            tau=0.5,
            hard=False,
        )[:, 0]
        return torch.where(logits < 0.0, 0.0, gs_sample)

    @property
    def proportion(self) -> torch.Tensor:
        return torch.nn.functional.sigmoid(self.logits).mean()

    def forward(self, x):
        y = self.original_linear(x)
        hidden = self.lin0(x)
        #mask = self.sample_from_logits(self.logits)
        #hidden_masked = mask * hidden
        y0 = self.lin1(hidden)
        if len(x.shape) == 3:
            non_channel_dim = (0, 1)
        else:
            raise NotImplementedError
        self.loss = per_channel_noise_to_signal_ratio(y=y, x=y0, non_channel_dim=non_channel_dim)
        return y

    def entropy_loss(self, epsilon: float = 0.01) -> torch.Tensor:
        probs_ = torch.sigmoid(self.logits)[..., None]
        probs = torch.cat([probs_, 1. - probs_], axis=1)
        return torch.maximum(- (probs * torch.log(probs)).sum(dim=1).mean(), torch.tensor(epsilon))

    def fuse(self) -> Tuple[nn.Linear, nn.Linear]:
        indices = torch.where(self.logits > 0)[0]
        logger.info(f'Leaving: {len(indices) / len(self.logits)}')
        indices_lin0 = indices.view(-1, 1)
        new_weight_lin0 = torch.take_along_dim(self.lin0.weight, dim=0, indices=indices_lin0)
        self.lin0.weight.data = new_weight_lin0
        indices_lin1 = indices.view(1, -1)
        new_weight_lin1 = torch.take_along_dim(self.lin1.weight, dim=1, indices=indices_lin1)
        self.lin1.weight.data = new_weight_lin1
        self.lin0.out_features = len(indices)
        self.lin1.in_features = len(indices)
        return self.lin0, self.lin1

    @classmethod
    def build_from_linear(cls, module: nn.Linear) -> "DecomposedLinear":
        in_channels = module.in_features
        out_channels = module.out_features
        new_module = cls(in_features=in_channels, out_features=out_channels, bias=module.bias is not None)
        new_module.original_linear.weight.data = module.weight
        if module.bias is not None:
            new_module.original_linear.bias.data = module.bias
        return new_module


@register_notrace_module
class PowerModule(torch.nn.Module):

    def __init__(self, pow: Union[float, int]):
        super().__init__()
        self.pow = pow

    def forward(self, inputs: torch.Tensor):
        return torch.pow(inputs, self.pow)


@register_notrace_module
class UnsqueezeModule(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor):
        return torch.unsqueeze(inputs, self.dim)


@register_unprunable_module
@register_notrace_module
class ExpandTokenModule(torch.nn.Module):
    """
    Supports `cait/crossvit` models.
    """

    def __init__(self, sizes: Union[torch.Size, int]):
        super().__init__()
        self.sizes = sizes

    def forward(self, inputs: List[torch.Tensor]):
        x, b = inputs
        return x.expand(b, *self.sizes)


@register_notrace_module
class AddcmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]):
        input, tensor1, tensor2 = inputs[:3]
        if len(inputs) == 4:
            value = inputs[3]
        else:
            value = 1
        return torch.addcmul(input, tensor1, tensor2, value=value)


@register_unprunable_module
@register_notrace_module
class ScaledDotProductAttentionModule(torch.nn.Module):
    def __init__(self, dropout_p=0.0, is_causal: bool = False):
        super().__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal

    def forward(self, inputs: List[torch.Tensor]):
        query, key, value = inputs[:3]
        if len(inputs) == 4:
            attn_mask = inputs[3]
        else:
            attn_mask = None
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            is_causal=self.is_causal,
        )


@register_unprunable_module
@register_notrace_module
class AutoWrapFunctionModule(torch.nn.Module):
    PREDECESSOR_KEYWORD = 'predecessor'

    def __init__(self, function: Callable, spec: Dict):
        super().__init__()
        self.spec = spec
        self.function = function

    def _build_inputs(self, spec: Dict, inputs=None, inputs_counter: int = 0):
        """
        At this point we suppport only inputs that are pickled Pytthon primitives, predecessors,
        or tuples/lists with elements of these two types
        """
        result = []
        for key, value in spec.items():
            if value == self.PREDECESSOR_KEYWORD:
                _input = inputs[inputs_counter]
                inputs_counter += 1
                result.append(_input)
            elif isinstance(value, Dict):
                result.append(self._build_inputs(value, inputs, inputs_counter))
            else:
                result.append(value)

        return result

    def forward(self, inputs):
        if not isinstance(inputs, List):
            inputs = [inputs]
        _inputs = self._build_inputs(self.spec, inputs)
        return self.function(*_inputs)


@register_unprunable_module
@register_notrace_module
class StackModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: List[torch.Tensor]):
        if isinstance(inputs, List) and len(inputs) == 1:
            return torch.unsqueeze(inputs[0], self.dim)
        if isinstance(inputs, torch.Tensor):
            return torch.unsqueeze(inputs, self.dim)
        return torch.stack(inputs, self.dim)


class ArgModule(torch.nn.Module):
    """
    This module is designed to hold some hyperparameters/tensors that need to be serialized.
    For example if a model works only for specific input shapes and was build given a predefined
    input_shape, then it may be necessary to hold that information somewhere.
    """

    def __init__(self, arg):
        super().__init__()
        if isinstance(arg, torch.Tensor):
            self.register_buffer('arg', arg, persistent=True)
        else:
            self.arg = arg

    def forward(self, x):
        return self.arg


@register_unprunable_module
@register_notrace_module
class RollModule(torch.nn.Module):
    def __init__(self, shifts: Union[int, Tuple[int, ...]], dims: Union[int, Tuple[int, ...]] = None):
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, inputs: torch.Tensor):
        return torch.roll(inputs, shifts=self.shifts, dims=self.dims)


@register_notrace_module
class GetItemModule(torch.nn.Module):

    def __init__(
            self,
            index: Union[int, Tuple[int, ...]],
    ):
        super().__init__()
        self.index = index

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]):
        if not isinstance(inputs, (torch.Tensor, list)):
            return inputs
        if isinstance(self.index, int):
            return inputs[self.index]
        else:
            return [x for k, x in inputs if k in self.index]


class StageZeroSllrcAttention(torch.nn.Module):
    def __init__(
            self,
            dim,
            num_heads: int = 8,
            use_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            use_out_bias: bool = True,
            width_multiplier: float = 0.25,
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
        self.concat = ConcatModule(dim=-1)
        self.attention_matmul = TfMatmulModule(transpose=True, normalize=True)
        self.final_matmul = TfMatmulModule(transpose=False, normalize=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.inner_dim = int(dim_per_head * width_multiplier)
        self.width_multiplier = width_multiplier

        # additional modules

        self.shared_q0 = torch.nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.q1s = torch.nn.ModuleList(
            [torch.nn.Linear(self.inner_dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])

        self.shared_k0 = torch.nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.k1s = torch.nn.ModuleList(
            [torch.nn.Linear(self.inner_dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])

        self.shared_v0 = torch.nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.v1s = torch.nn.ModuleList(
            [torch.nn.Linear(self.inner_dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])

    def forward(self, x):
        carriers = []
        x_q_shared = self.shared_q0(x)  # (B, T, dim')
        x_k_shared = self.shared_k0(x)  # (B, T, dim')
        x_v_shared = self.shared_v0(x)  # (B, T, dim')
        for k in range(self.num_heads):
            x_q = self.q[k](x)
            x_q1 = self.q1s[k](x_q_shared)
            x_k = self.k[k](x)
            x_k1 = self.k1s[k](x_k_shared)
            x_v = self.v[k](x)
            x_v1 = self.v1s[k](x_v_shared)
            qk = self.attention_matmul([x_q, x_k])
            qk = self.softmax(qk)
            qk = self.attn_drop(qk)
            qkv = self.final_matmul([qk, x_v])
            carriers.append(qkv)
        x = self.concat(carriers)
        x = self.out(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BatchedAttention(torch.nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attention_matmul = TfMatmulModule(transpose=True, normalize=False)
        self.value_matmul = TfMatmulModule(transpose=False, normalize=False)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @classmethod
    def convert_from_timm(cls, module: Attention):
        dim = module.head_dim * module.num_heads
        result = cls(
            dim=dim,
            num_heads=module.num_heads,
            qkv_bias=module.qkv.bias is not None,
        )
        result.qkv = module.qkv
        result.proj = module.proj
        return result


class SllrcAttention(torch.nn.Module):
    def __init__(
            self,
            dim,
            num_heads: int = 8,
            use_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            use_out_bias: bool = True,
            width_multiplier: float = 0.25,
    ):
        super().__init__()
        if use_bias is False:
            logger.warning(f'Not using bias in `BaselineAttention` implementation of attenion will'
                           f'result in issues with `tflite` deployment!')
        self.dim = dim
        self.num_heads = num_heads
        dim_per_head = dim // num_heads
        self.out = torch.nn.Linear(self.num_heads * dim_per_head, dim, bias=use_out_bias)
        self.concat = ConcatModule(dim=-1)
        self.attention_matmul = TfMatmulModule(transpose=True, normalize=True)
        self.final_matmul = TfMatmulModule(transpose=False, normalize=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.inner_dim = int(dim_per_head * width_multiplier)

        # additional modules

        self.shared_q0 = torch.nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.q1s = torch.nn.ModuleList(
            [torch.nn.Linear(self.inner_dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])

        self.shared_k0 = torch.nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.k1s = torch.nn.ModuleList(
            [torch.nn.Linear(self.inner_dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])

        self.shared_v0 = torch.nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.v1s = torch.nn.ModuleList(
            [torch.nn.Linear(self.inner_dim, dim_per_head, bias=use_bias) for _ in range(num_heads)])

    def forward(self, x):
        carriers = []
        x_q_shared = self.shared_q0(x)  # (B, T, dim')
        x_k_shared = self.shared_k0(x)  # (B, T, dim')
        x_v_shared = self.shared_v0(x)  # (B, T, dim')
        for k in range(self.num_heads):
            x_q1 = self.q1s[k](x_q_shared)
            x_k1 = self.k1s[k](x_k_shared)
            x_v1 = self.v1s[k](x_v_shared)
            qk = self.attention_matmul([x_q1, x_k1])
            qk = self.softmax(qk)
            qk = self.attn_drop(qk)
            qkv = self.final_matmul([qk, x_v1])
            carriers.append(qkv)
        x = self.concat(carriers)
        x = self.out(x)
        x = self.proj_drop(x)
        return x


class MultiQueryAttention(torch.nn.Module):
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
        self.k = torch.nn.Linear(dim, dim_per_head, bias=use_bias)
        self.v = torch.nn.Linear(dim, dim_per_head, bias=use_bias)
        self.out = torch.nn.Linear(self.num_heads * dim_per_head, dim, bias=use_out_bias)
        self.concat = ConcatModule(dim=-1)
        self.attention_matmul = TfMatmulModule(transpose=True, normalize=True)
        self.final_matmul = TfMatmulModule(transpose=False, normalize=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        carriers = []
        x_k = self.k(x)
        x_v = self.v(x)
        for k in range(self.num_heads):
            x_q = self.q[k](x)
            qk = self.attention_matmul([x_q, x_k])
            qk = self.softmax(qk)
            qk = self.attn_drop(qk)
            qkv = self.final_matmul([qk, x_v])
            carriers.append(qkv)
        x = self.concat(carriers)
        x = self.out(x)
        x = self.proj_drop(x)
        return x


class SvdLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.u = nn.Parameter(torch.zeros(size=(out_features, self.rank)))
        self.log_s = nn.Parameter(torch.ones(size=(self.rank,)))
        self.v_t = nn.Parameter(torch.zeros(size=(self.rank, out_features)))
        self.bias = nn.Parameter(torch.zeros(size=(out_features,))) if use_bias else None
        self.original = None

    @property
    def rank(self) -> int:
        return min(self.in_features, self.out_features)

    @property
    def s(self):
        return torch.exp(self.log_s)

    @classmethod
    def build_from_linear(cls, module: nn.Linear):
        with torch.no_grad():
            U, S, V_t = torch.linalg.svd(module.weight, full_matrices=False)

        result_module = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            use_bias=module.bias is not None,
        )
        result_module.u.data = U
        result_module.v_t.data = V_t
        result_module.log_s.data = torch.log(S)
        if module.bias is not None:
            result_module.bias.data = module.bias
        return result_module

    def forward(self, x):
        y = x @ self.v_t.T
        y = y * self.s
        y = y @ self.u.T
        if self.bias is not None:
            y += self.bias
        return y
