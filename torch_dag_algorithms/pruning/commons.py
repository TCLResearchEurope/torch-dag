from typing import Dict

from torch import nn

from torch_dag import structured_modules as smodules
from torch_dag_algorithms.pruning.modules import MaskModule
from torch_dag_timm_plugin.mask_propagation import PASS_THROUGH_CHANNELS_TIMM_CLASSES

PASS_THROUGH_CHANNELS_CLASSES = (
    smodules.ChannelAffineModule,
    smodules.NormalizeModule,
    smodules.LayerNormWithOptionalBias,
    smodules.TfBatchNorm1d,
    nn.BatchNorm2d,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Dropout,
    nn.Upsample,
    nn.LayerNorm,
    nn.BatchNorm1d,
    MaskModule,
    smodules.PowerModule,
    smodules.AddcmulModule,
    smodules.HalfPixelCentersFalseBilinearUpsample,
    smodules.BilinearUpsampling,
    smodules.PadModule,
    smodules.NormalizeModule,
    smodules.InterpolateModule,
    smodules.ScalarMul,
    smodules.MeanModule,

)
PASS_THROUGH_CHANNELS_CLASSES += smodules.ACTIVATION_MODULES

PASS_THROUGH_CHANNELS_CLASSES += PASS_THROUGH_CHANNELS_TIMM_CLASSES

ELEMENWISE_CHANNEL_CLASSES = (
    smodules.AddModule,
    smodules.MulModule,
    smodules.SubModule,
)


def is_depthwise_conv(module: nn.Module) -> bool:
    return isinstance(module, (
        nn.Conv2d, nn.ConvTranspose2d)) and module.in_channels == module.groups and module.in_channels > 1


def is_conv_source(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if module.in_channels == 1:
            return True
        if module.groups < module.in_channels:
            return True
    return False


def is_linear_source(module: nn.Module):
    if isinstance(module, nn.Linear):
        return True
    return False


def is_source(module: nn.Module):
    return is_linear_source(module) or is_conv_source(module)


def get_source_out_channels(module: nn.Module) -> int:
    assert is_source(module)
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return module.out_channels
    elif isinstance(module, nn.Linear):
        return module.out_features
    else:
        raise NotImplementedError


def get_source_in_channels(module: nn.Module) -> int:
    assert is_source(module)
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return module.in_channels
    elif isinstance(module, nn.Linear):
        return module.in_features
    else:
        raise NotImplementedError


def is_sink(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.groups < module.in_channels:
        return True
    if isinstance(module, nn.Linear):
        return True
    return False


def get_orbits_dict(dag) -> Dict:
    all_orbit_modules = set([v.module.orbit for v in dag.inner_vertices if isinstance(v.module, MaskModule)])
    return {orbit.name: orbit for orbit in all_orbit_modules}


TRUNCATE_ON = smodules.ConcatModule


class Skipped:
    """
    Class used for marking orbit as skipped durning orbitalization process due to lower channels_num than block_size.
    """
    pass
