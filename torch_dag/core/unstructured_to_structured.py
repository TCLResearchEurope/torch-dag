import copy
import logging
import operator
from collections.abc import Iterable
from functools import singledispatch
from typing import Tuple, Optional, Callable, Dict, Union, List, Type

import torch
from torch import fx

from torch_dag import structured_modules
from torch_dag.core import dag_module
from torch_dag.core.dag_module_utils import insert_between
from torch_dag.core.dag_module_utils import recursively_remove_redundant_vertices
from torch_dag.core.dag_module_utils import remove_identity_traverser
from torch_dag.core.dag_tracer import DagTracer, _autowrap_functions
from torch_dag.core.module_handling import is_handled_module
from torch_dag.core.prunable_modules import prunable_modules
from torch_dag_timm_plugin.module_conversion import maybe_convert_timm_module

logger = logging.getLogger(__name__)


def convert_reshape_node(node: torch.fx.node.Node):
    if isinstance(node.args[1], Tuple):
        if len(node.args) > 2:
            raise NotImplementedError
        args = node.args[1]
    else:
        args = node.args[1:]
    reshape_spec = {}
    for k, arg in enumerate(args):
        if isinstance(arg, torch.fx.Node):
            reshape_spec[k] = structured_modules.ReshapeWithSpecModuleV2.PREDECESSOR_KEYWORD
        else:
            reshape_spec[k] = arg

    return structured_modules.ReshapeWithSpecModuleV2(spec=reshape_spec)


def convert_dot_node(node: torch.fx.node.Node):
    if node.target == 'permute':
        permutation = tuple(node.args[1:])
        return structured_modules.PermuteModule(perm=permutation)

    if node.target == 'transpose':
        return structured_modules.TransposeModule(
            dim0=node.args[1],
            dim1=node.args[2]
        )

    elif node.target == 'view':
        return convert_reshape_node(node)

    elif node.target == 'norm':
        assert len(node.args) == 1
        return structured_modules.NormModule(**node.kwargs)

    elif node.target == 'mean':
        return structured_modules.MeanModule(*node.args[1:], **node.kwargs)

    elif node.target == 'split':
        args = node.args[1:]
        kwargs = node.kwargs
        return structured_modules.SplitModule(*args, **kwargs)

    elif node.target == 'contiguous':
        return torch.nn.Identity()

    elif node.target == 'mul' or node.target == 'mul_':
        if not isinstance(node.args[1], torch.fx.node.Node):
            return structured_modules.ScalarMul(node.args[1])
        return structured_modules.MulModule()

    elif node.target == 'reshape':
        return convert_reshape_node(node)


    elif node.target == 'flatten':
        if isinstance(node.args[1], int) and len(node.args) == 2:
            return structured_modules.FlattenModule(start_dim=node.args[1])
        else:
            raise NotImplementedError

    elif node.target == 'shape':
        raise NotImplementedError
    elif node.target == 'unbind':
        return structured_modules.UnbindModule(*node.args[1:], **node.kwargs)

    elif node.target == 'softmax':
        return torch.nn.Softmax(*node.args[1:], **node.kwargs)

    elif node.target == 'sigmoid':
        return torch.nn.Sigmoid(*node.args[1:], **node.kwargs)

    elif node.target == 'chunk':
        return structured_modules.ChunkModule(*node.args[1:], **node.kwargs)

    elif node.target == 'flatten':
        return structured_modules.FlattenModule(*node.args[1:], **node.kwargs)
    elif node.target == 'unsqueeze':
        return structured_modules.UnsqueezeModule(dim=node.args[1])
    elif node.target == 'expand':
        if isinstance(node.args[1], fx.Node):
            sizes = node.args[2:]
            return structured_modules.ExpandTokenModule(sizes=sizes)
        else:
            raise NotImplementedError

    elif node.target == 'expand_as':
        return structured_modules.ExpandAsModule()

    else:
        raise NotImplementedError(f'Not implemented dot node conversion for: {node.target}')


def _build_autowrap_function_spec(args) -> Dict:
    spec = {}
    for k, arg in enumerate(args):
        if isinstance(arg, torch.fx.Node):
            spec[k] = structured_modules.AutoWrapFunctionModule.PREDECESSOR_KEYWORD
        elif isinstance(arg, Tuple):
            inner_spec = _build_autowrap_function_spec(arg)
            spec[k] = inner_spec
        elif isinstance(arg, Iterable):
            raise NotImplementedError
        else:
            spec[k] = arg
    return spec


def convert_autowrap_function(node: torch.fx.Node):
    """
    EXPERIMENTAL SUPPORT
    """
    logger.info(f'Auto wrapping non-traceable function: {node.target}')
    logger.info(f'node.args: {node.args}')
    assert len(node.kwargs) == 0
    spec = _build_autowrap_function_spec(node.args)
    return structured_modules.AutoWrapFunctionModule(node.target, spec=spec)


def convert_getitem_node(node: torch.fx.Node):
    if len(node.args[1:]) == 1 and isinstance(node.args[1], int):
        return structured_modules.TensorExtractorModule(index=node.args[1])
    elif isinstance(node.args[1], Tuple):
        return structured_modules.SliceModule(slice_spec=node.args[1])
    elif len(node.args) == 2 and isinstance(node.args[1], slice):
        return structured_modules.SliceModule(slice_spec=node.args[1])
    else:
        raise NotImplementedError


class DagBuilder:

    def __init__(
            self,
            model: torch.nn.Module,
            autowrap_functions: Tuple[Callable] = (),
            custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),
    ):
        self.model = model
        self.autowrap_functions = tuple(_autowrap_functions) + tuple(
            autowrap_functions)  # add custom autowrap functions
        self.custom_autowrap_torch_module_classes = custom_autowrap_torch_module_classes

    def add_leaf_modules_with_arg_modules(self, dag: dag_module.DagModule, name: str, modules, node: torch.fx.Node):
        """
        TODO try to make this work for some edge cases where a given leaf module has some non-Node inputs
        """
        arg_modules = modules
        arg_vertices = [
            dag.add_vertex(
                name=f'{name}_arg_{k}',
                module=m,
                predecessors=[],
            ) for k, m in enumerate(arg_modules)
        ]
        predecessor_nodes = [input_node for input_node in node._input_nodes]
        predecessors = [dag.get_vertex_by_name(node.name) for node in predecessor_nodes]
        return dag.add_vertex(
            name=name,
            module=modules[0],
            predecessors=predecessors + arg_vertices,
        )

    @staticmethod
    def get_named_buffers_names(module: torch.nn.Module):
        return [e[0] for e in module.named_buffers()]

    @staticmethod
    def get_tensor_from_named_buffers_by_name(module: torch.nn.Module, name: str):
        return [e[1] for e in module.named_buffers() if e[0] == name][0]

    def build_dag_by_tracing(self) -> dag_module.DagModule:
        """
        This is the core function that builds a `DagMOdule` from `torch.nn.Module` by first tracing the model using `torch.FX`
        and then unifying the model structure using a finite set of `unified` modules.
        """
        model = copy.deepcopy(self.model)
        model.eval()
        tracer = DagTracer(
            autowrap_functions=self.autowrap_functions,
            custom_autowrap_torch_module_classes=self.custom_autowrap_torch_module_classes,
        )
        graph = tracer.trace(model)
        modules_dict = dict(model.named_modules())
        input_counter = 0

        # create input vertices

        dag = dag_module.DagModule(
            model.__class__.__name__,
            vertices=[],
        )

        input_vertices = []

        # add input nodes

        for node in graph.nodes:
            num_predecessors = len(node._input_nodes)
            if num_predecessors == 0:  # input vertex or input parameter
                if node.target in model.state_dict():  # free parameter node
                    pass
                elif node.target in [e[0] for e in model.named_buffers()]:  # free arg node
                    pass
                else:
                    vertex = dag_module.InputVertex(name=f'{node.name}')
                    input_vertices.append(vertex)
                    input_counter += 1

        dag.vertices.extend(input_vertices)

        # crate free parameter modules
        for node in graph.nodes:
            num_predecessors = len(node._input_nodes)
            if num_predecessors == 0 and node.target in model.state_dict():  # free parameter node
                module = structured_modules.ParameterModule(param=torch.nn.Parameter(model.state_dict()[node.target]))
                vertex = dag.add_vertex(
                    name=node.name,
                    module=module,
                    predecessors=[],
                )

        # crate free arg modules
        for node in graph.nodes:
            num_predecessors = len(node._input_nodes)
            if num_predecessors == 0 and node.target in self.get_named_buffers_names(model):
                tensor = self.get_tensor_from_named_buffers_by_name(model, name=node.target)
                module = structured_modules.ArgModule(arg=tensor)
                vertex = dag.add_vertex(
                    name=node.name,
                    module=module,
                    predecessors=[],
                )

        output_vertices = []
        for node in graph.nodes:
            num_predecessors = len(node._input_nodes)
            if num_predecessors > 0:  # inner vertex
                module = self.convert_node(node, modules_dict, model.state_dict())
                if module is not None:
                    name = node.name
                    if not isinstance(module, torch.nn.Module):
                        # vertex = self.add_leaf_modules_with_arg_modules(dag=dag, name=name, modules=module)
                        raise NotImplementedError
                    else:
                        # if isinstance(node.args[0], (List, Tuple)):
                        #     args = node.args[0]
                        # else:
                        #     args = node.args
                        # predecessor_nodes = [e for e in list(args) if isinstance(e, torch.fx.Node)]
                        predecessor_nodes = [input_node for input_node in node._input_nodes]
                        predecessors = [dag.get_vertex_by_name(node.name) for node in predecessor_nodes]
                        vertex = dag.add_vertex(
                            name=name,
                            module=module,
                            predecessors=predecessors,
                        )

                    if 'output' in [node.name for node in node.users]:
                        output_vertices.append(vertex)

        if len(output_vertices) == 1:
            dag.output_vertex = output_vertices[0]
        else:
            output_vertex = dag.add_vertex(
                name='final_merger',
                module=structured_modules.TensorMergerModule(),
                predecessors=output_vertices,
            )
            dag.output_vertex = output_vertex

        recursively_remove_redundant_vertices(dag)
        return dag

    def convert_node(self, node: torch.fx.node.Node, modules_dict, state_dict):
        logger.debug(f'Converting node: {node} with target: {node.target}')
        logger.debug(f'input_nodes: {node._input_nodes}')
        logger.debug(f'args: {node.args}')
        logger.debug(f'kwargs: {node.kwargs}')
        logger.debug(f'next: {node.next}')

        if node.target == operator.add:
            return structured_modules.AddModule()
        elif node.target == operator.mul:
            # tensor * tensor
            if len(node.args) == 2 and all([isinstance(arg, fx.Node) for arg in node.args]):
                return structured_modules.MulModule()
            # scalar * tensor
            elif len(node.args) == 2 and isinstance(node.args[0], fx.Node):
                assert isinstance(node.args[1], (float, int))
                return structured_modules.ScalarMul(scalar=node.args[1])
            # tensor * scalar
            elif len(node.args) == 2 and isinstance(node.args[1], fx.Node):
                assert isinstance(node.args[0], (float, int))
                return structured_modules.ScalarMul(scalar=node.args[0])
            else:
                raise NotImplementedError(f'{operator.mul} conversion not implemented'
                                          f' for args {node.args}')
        elif node.target == operator.sub:
            return structured_modules.SubModule()

        elif node.target == operator.truediv:
            if len(node.args) == 2 and isinstance(node.args[0], fx.Node) and isinstance(node.args[1], fx.Node):
                return structured_modules.DivModule()
            scalar = node.args[1]
            return structured_modules.ScalarMul(scalar=1.0 / scalar)

        elif node.target == operator.pow:
            return structured_modules.PowerModule(node.args[1])

        elif node.target == operator.getitem:
            return convert_getitem_node(node)

        elif node.target == operator.matmul:
            return structured_modules.MatmulModule()

        elif node.target == torch.cat:
            if node.kwargs.get('dim') == 1:
                return structured_modules.ConcatModule(dim=1)
            elif node.kwargs.get('dim') == 2:
                return structured_modules.ConcatModule(dim=2)
            elif len(node.args) == 2:
                return structured_modules.ConcatModule(dim=node.args[1])
            elif len(node.args) == 1:
                return structured_modules.ConcatModule(dim=0)
            else:
                raise NotImplementedError

        elif node.target == torch.nn.functional.pad:
            return structured_modules.PadModule(*node.args[1:], **node.kwargs)

        elif node.target == torch.nn.functional.gelu:
            return torch.nn.GELU()

        elif node.target == torch.nn.functional.relu6:
            return torch.nn.ReLU6()

        elif node.target == getattr:
            attribute_name = node.args[1]
            if attribute_name == 'shape':
                return structured_modules.GetShapeModuleV2()
            else:
                raise NotImplementedError

        elif node.target == torch.permute:
            if isinstance(node._args[1], Tuple):
                permutation = node._args[1]
                return structured_modules.PermuteModule(perm=permutation)
            else:
                raise NotImplementedError

        elif node.target == torch.reshape:
            return convert_reshape_node(node)

        elif node.target == torch.flatten:
            # TODO handle start dim and end dim
            return structured_modules.ReshapeModule(target_shape=[-1])

        elif node.target == torch.mean:
            assert len(node.args) == 1
            return structured_modules.MeanModule(**node.kwargs)

        elif node.target == torch.sum:
            raise NotImplementedError

        elif node.target == torch.split:
            return structured_modules.SplitModule(*node.args[1:], **node.kwargs)

        elif node.target == torch.square:
            return structured_modules.PowerModule(pow=2)

        elif node.target == torch.addcmul:
            return structured_modules.AddcmulModule()

        elif node.target == torch.stack:
            if len(node.args) == 1:
                dim = 0
            elif len(node.args) == 2:
                dim = node.args[1]
            else:
                raise NotImplementedError

            return structured_modules.StackModule(dim=dim)

        elif node.target == torch.roll:
            if len(node.args) != 1 or len(node.kwargs) != 2:
                raise NotImplementedError
            return structured_modules.RollModule(**node.kwargs)

        elif node.target == torch.nn.functional.interpolate:
            assert len(node.args) == 1
            return structured_modules.InterpolateModule(**node.kwargs)

        elif node.target == torch.nn.functional.dropout:
            assert len(node.args) == 1
            return torch.nn.Dropout(p=node.kwargs['p'], inplace=node.kwargs['inplace'])

        elif node.target == torch.nn.functional.adaptive_avg_pool2d:
            assert len(node.args) == 2
            assert len(node.kwargs) == 0
            size = node.args[1]
            return torch.nn.AdaptiveAvgPool2d(output_size=size)

        elif node.target == torch.nn.functional.max_pool2d:
            kernel_size = node.args[1]
            return torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                **node.kwargs
            )

        elif node.target == torch.nn.functional.log_softmax:
            if len(node.args) == 2:
                dim = node.args[1]
            else:
                dim = node.kwargs['dim']
            return torch.nn.LogSoftmax(dim=dim)

        elif node.target == torch.nn.functional.scaled_dot_product_attention:
            if len(node.args) == 3:
                return structured_modules.ScaledDotProductAttentionModule()
            else:
                raise NotImplementedError

        elif isinstance(node.target, str):
            if node.target == 'output':  # output node in torch.FX graph
                return
            if node.target in modules_dict:  # node represets a torch module
                return self.convert_leaf_module(node, modules_dict)
            else:
                return convert_dot_node(node)
        elif node.target in self.autowrap_functions:
            return convert_autowrap_function(node)
        else:
            raise NotImplementedError(f'Node target: {type(node.target)} not implemented.')

    def are_all_args_nodes(self, args):
        # args can be nested like (arg0, (arg1, arg2)) -> we have to take care of that
        for arg in args:
            if isinstance(arg, torch.fx.Node):
                pass
            elif isinstance(arg, (Tuple, List)):
                if not self.are_all_args_nodes(arg):
                    return False
            else:
                return False
        return True

    def convert_leaf_module(self, node: torch.fx.Node, modules_dict) -> Union[torch.nn.Module, Tuple[torch.nn.Module]]:
        module = modules_dict[node.target]
        module = maybe_convert_timm_module(module)
        logger.debug(f'Using in built torch module: {type(module)}')
        # if all node.args are `torch.fx.Node` objects return module
        if self.are_all_args_nodes(node.args):
            return module
        else:
            raise NotImplementedError(f'At this point leaf modules with non node inputs are not supported')
        #     # arg_modules = []
        #     # for arg in node.args:
        #     #     if not isinstance(arg, torch.fx.Node):
        #     #         arg_module = structured_modules.ArgModule(arg)
        #     #         arg_modules.append(arg_module)
        # return module, *arg_modules


def unroll_prunable_modules(vertex: dag_module.InnerVertex):
    if any([isinstance(vertex.module, class_) for class_ in prunable_modules]):
        logger.debug(f'Unrolling prunable module : {vertex.name}, of class: {vertex.module.__class__.__name__}')
        vertex.module = build_from_unstructured_module(vertex.module)


@singledispatch
def build_dag_by_tracing(
        module: torch.nn.Module,
        autowrap_functions: Tuple[Callable] = (),
        custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),
) -> dag_module.DagModule:
    return _build_dag_by_tracing(
        module,
        autowrap_functions=autowrap_functions,
        custom_autowrap_torch_module_classes=custom_autowrap_torch_module_classes,
    )


try:
    from timm.models.vision_transformer import VisionTransformer


    @build_dag_by_tracing.register
    def _(
            module: VisionTransformer,
            autowrap_functions: Tuple[Callable] = (),
            custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),
    ):
        return _build_dag_by_tracing(module)
        # """
        # TODO: Remove this after adding handling of `torch.expand`
        # Special method to handle deit3 family of models. We need to take care of cls token and head unbatching for
        # ease of pruning.
        # """
        # cls_token = module.cls_token
        # module.cls_token = None
        # dag = _build_dag_by_tracing(module)
        # data = cls_token[0, 0, :]
        # token_mod = structured_modules.AuxiliaryTokenModule(dim=data.shape[0])
        # token_mod.token.data = data
        # if module.no_embed_class:
        #     insert_between(
        #         dag=dag,
        #         name='cls_token',
        #         new_module=token_mod,
        #         before_vertex=dag.inner_vertices[3],
        #         after_vertex=dag.inner_vertices[2],
        #     )
        # else:
        #     insert_between(
        #         dag=dag,
        #         name='cls_token',
        #         new_module=token_mod,
        #         before_vertex=dag.inner_vertices[2],
        #         after_vertex=dag.inner_vertices[1],
        #     )
        # flat_dag = dag.flatten()
        # return flat_dag
except ImportError:
    pass

try:
    from timm.models.beit import Beit
    from torch_dag_timm_plugin.modified_timm_modules import ModifiedBeit


    @build_dag_by_tracing.register
    def _(
            module: Beit,
            autowrap_functions: Tuple[Callable] = (),
            custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),
    ):
        """
        Special method to handle beit family of models. We need to take care of cls token.
        """
        cls_token = module.cls_token
        module.__class__ = ModifiedBeit
        dag = _build_dag_by_tracing(module)
        data = cls_token[0, 0, :]
        token_mod = structured_modules.AuxiliaryTokenModule(dim=data.shape[0])
        token_mod.token.data = data
        if module.pos_embed is None:
            patch_embed_vertex = dag.get_vertex_by_name('patch_embed')
            pos_drop_vertex = dag.get_vertex_by_name('pos_drop')
            insert_between(
                dag=dag,
                name='cls_token',
                new_module=token_mod,
                before_vertex=pos_drop_vertex,
                after_vertex=patch_embed_vertex,
            )
        else:
            raise NotImplementedError
        flat_dag = dag.flatten()
        return flat_dag
except ImportError:
    pass

try:
    from torch_dag_timm_plugin.modified_timm_modules import ModifiedConvMlp
    from timm.models.vgg import VGG


    @build_dag_by_tracing.register
    def _(
            module: VGG,
            autowrap_functions: Tuple[Callable] = (),
            custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),
    ):
        """
        Needed to fix untraceable `ConvMlp` module in timm
        """
        pre_logits_module = module.pre_logits
        pre_logits_module.__class__ = ModifiedConvMlp
        return _build_dag_by_tracing(module)
except ImportError:
    pass


def _build_dag_by_tracing(
        model: torch.nn.Module,
        autowrap_functions: Tuple[Callable] = (),
        custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),

) -> dag_module.DagModule:
    # TODO: this method is deprecated -> remove it after refactor
    """
    This is the core function that builds a `DagMOdule` from `torch.nn.Module` by first tracing the model using `torch.FX`
    and then unifying the model structure using a finite set of `unified` modules.
    """
    model = copy.deepcopy(model)
    model.eval()
    dag_builder = DagBuilder(
        model,
        autowrap_functions=autowrap_functions,
        custom_autowrap_torch_module_classes=custom_autowrap_torch_module_classes,
    )
    return dag_builder.build_dag_by_tracing()


def build_from_unstructured_module(
        model: torch.nn.Module,
        name: Optional[str] = None,
        remove_identity_vertices: bool = True,
        do_unroll_prunable_modules: bool = True,
        autowrap_functions: Tuple[Callable] = (),
        custom_autowrap_torch_module_classes: Tuple[Type[torch.nn.Module]] = (),
) -> dag_module.DagModule:
    dag = build_dag_by_tracing(
        copy.deepcopy(model),
        autowrap_functions=autowrap_functions,
        custom_autowrap_torch_module_classes=custom_autowrap_torch_module_classes,
    )
    if name:
        dag.name = name

    if remove_identity_vertices:
        dag.traverse(remove_identity_traverser)

    if do_unroll_prunable_modules:
        dag.traverse(unroll_prunable_modules)
    for module in dag.inner_modules:
        is_handled_module(module)
    return dag
