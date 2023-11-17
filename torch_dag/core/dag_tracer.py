import logging
import math
from types import ModuleType
from typing import Tuple, Callable, Type

import torch

from torch_dag_timm_plugin.timm_modules_handling import initial_autowrap_timm_modules, initial_autowrap_timm_functions

logger = logging.getLogger(__name__)

_autowrap_modules = set(initial_autowrap_timm_modules)
_autowrap_functions = set(initial_autowrap_timm_functions)


def register_notrace_module(module: torch.nn.Module):
    """
    Decorator for modules which ought not to be traced through
    """
    _autowrap_modules.add(module)
    return module


def register_notrace_function(function: Callable):
    """
    Decorator for modules which ought not to be traced through
    """
    _autowrap_functions.add(function)
    return function


class DagTracer(torch.fx.Tracer):

    def __init__(
            self,
            autowrap_modules: Tuple[ModuleType] = (math,),
            autowrap_functions: Tuple[Callable, ...] = tuple(_autowrap_functions),
            param_shapes_constant: bool = False,
            custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),
    ) -> None:
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
            param_shapes_constant=param_shapes_constant,
        )
        self.custom_autowrap_torch_module_classes = set(custom_autowrap_torch_module_classes)
        _autowrap_modules.update(custom_autowrap_torch_module_classes)
        self.custom_leaf_modules = set().union(
            *[_autowrap_modules, self.custom_autowrap_torch_module_classes])

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        _result = super().is_leaf_module(m, module_qualified_name)
        _custum_no_trace = type(m) in self.custom_leaf_modules
        if _custum_no_trace:
            logger.debug(f'Module of type: {type(m)} is considered a leaf module.')
            return True
        return _result
