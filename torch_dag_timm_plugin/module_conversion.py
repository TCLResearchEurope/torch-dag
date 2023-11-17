from functools import singledispatch

from torch import nn

from torch_dag import structured_modules as smodules
from torch_dag.core import prunable_modules as pmodules
from torch_dag.core.dag_module import DagModule, InputVertex


@singledispatch
def maybe_convert_timm_module(module: nn.Module) -> nn.Module:
    """
    A function to convert/handle modules that require special care because one or more of the following reasons:
    - their implementation is not algorithm-friendly, e.g., they are not easily prunable
    - they are custom modules that can be easily expressed as other modules so that there is no duplication in
    classes used
    """
    return module


try:
    from timm.models.efficientformer_v2 import LayerScale2d, Attention2d, Attention2dDownsample
    from torch_dag_timm_plugin.modified_timm_modules import ModifiedAttention2d, ModifiedAttention2dDownsample


    @maybe_convert_timm_module.register
    def _(module: LayerScale2d):
        smodule = smodules.ChannelAffineModule(num_channels=module.gamma.shape[0], use_bias=False)
        smodule.weight = module.gamma
        return smodule


    @maybe_convert_timm_module.register
    def _(module: Attention2d):
        module.__class__ = ModifiedAttention2d
        return module


    @maybe_convert_timm_module.register
    def _(module: Attention2dDownsample):
        module.__class__ = ModifiedAttention2dDownsample
        return module

except ImportError:
    pass

try:
    from timm.models.layers import BatchNormAct2d


    @maybe_convert_timm_module.register
    def _(module: BatchNormAct2d):
        """
                Handling the nasty case of unreasonable implementation of `BatchNormAct2d` trom timm
                """
        bn = nn.BatchNorm2d(num_features=module.num_features)
        bn.weight.data = module.weight
        bn.bias.data = module.bias
        bn.running_mean.data = module.running_mean
        bn.running_var.data = module.running_var
        bn.eps = module.eps
        if not isinstance(module.drop, nn.Identity):
            raise NotImplementedError
        act = module.act
        if hasattr(act, 'inplace'):
            act.inplace = False

        input_vertex = InputVertex(name='x')
        dag = DagModule(
            name=module.__class__.__name__,
            vertices=[input_vertex],
        )

        vertex = dag.add_vertex(
            name=f'{dag.name}/bn',
            module=bn,
            predecessors=[input_vertex],
        )
        vertex = dag.add_vertex(
            name=f'{dag.name}/act',
            module=act,
            predecessors=[vertex],
        )
        dag.output_vertex = vertex
        return dag
except ImportError:
    pass

try:
    from timm.models.vision_transformer import Attention


    @maybe_convert_timm_module.register
    def _(module: Attention):
        smodule = pmodules.BaselineAttention.convert_from_timm(module)
        # smodule = smodules.BatchedAttention.convert_from_timm(module)
        return smodule
except ImportError:
    pass

try:
    from timm.models.metaformer import Attention as MetaFormerAttention


    @maybe_convert_timm_module.register
    def _(module: MetaFormerAttention):
        return pmodules.BaselineAttention.convert_from_timm(module)
except ImportError:
    pass
