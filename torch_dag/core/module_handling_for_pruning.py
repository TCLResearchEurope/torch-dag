from torch import nn

from torch_dag_timm_plugin.timm_modules_handling import unprunable_timm_modules

# a list of external leaf modules that we know we can't prune
# (can't remove input and output channels easily)
unprunable_modules = set(unprunable_timm_modules)


def register_unprunable_module(module: nn.Module):
    """
    Decorator for custom modules that cannot be pruned
    """
    unprunable_modules.add(module)
    return module
