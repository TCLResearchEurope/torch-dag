import logging
from functools import singledispatch

from torch import nn

from torch_dag import structured_modules as smodules
from torch_dag.core.dag_tracer import _autowrap_modules
from torch_dag.core.module_handling_for_pruning import unprunable_modules
from torch_dag_timm_plugin.modified_timm_modules import ALL_MODIFIED_TIMM_MODULES

logger = logging.getLogger(__name__)

ALLOWED_BUILT_IN_MODULES = [
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Flatten,
    nn.Linear,
    nn.Dropout,
    nn.Upsample,
    nn.LayerNorm,
    nn.Embedding,
    nn.BatchNorm1d,
    nn.Conv1d,
    nn.ConvTranspose2d,
]

ALLOWED_BUILT_IN_MODULES += list(smodules.ACTIVATION_MODULES)

ALLOWED_CUSTOM_MODULES = list(_autowrap_modules) + list(unprunable_modules) + list(ALL_MODIFIED_TIMM_MODULES)


def is_handled_module(module: nn.Module):
    builtin = any([type(module) == module_class for module_class in ALLOWED_BUILT_IN_MODULES])
    custom = any([type(module) == module_class for module_class in ALLOWED_CUSTOM_MODULES])
    if not builtin and not custom:
        logger.warning(f'The module: {module} of type: {type(module)} is not covered by `torch-dag`. '
                       f'by the DagModule. In particular, pruning support is not guaranteed.')
        # raise NotImplementedError(f'The module: {module} of type: {type(module)} is not handled '
        #                           f'by the DagModule. In particular, pruning will not be supported.')



