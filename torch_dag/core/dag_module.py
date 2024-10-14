import json
import logging
import os
from collections import Counter
from copy import deepcopy
from typing import List, Union, Optional, Set, Dict, Callable, Tuple, Type

import torch

from torch_dag.core.common_tools import per_channel_noise_to_signal_ratio
from torch_dag.core.dag_tracer import register_notrace_module

logger = logging.getLogger(__name__)


def _postprocess_module_output(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]):
    if isinstance(x, Tuple):
        x = list(x)
    if isinstance(x, List):
        result = []
        for el in x:
            if isinstance(el, (List, Tuple)):
                result.append(torch.stack([torch.tensor(t) for t in el]))
            else:
                result.append(el)
        return result
    return x


class Vertex:
    MAX_LEN_REPR = None

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        if self.MAX_LEN_REPR is not None and len(self.name) > self.MAX_LEN_REPR:
            return f'{self.name[:self.MAX_LEN_REPR // 2]}...{self.name[-self.MAX_LEN_REPR // 2:]}'
        return self.name

    def config_dict(self, atomic_modules: List[torch.nn.Module]):
        return {
            'name': self.name,
        }


class InputVertex(Vertex):
    def config_dict(self, atomic_modules: List[torch.nn.Module]):
        return {
            'name': self.name,
            'type': 'input',
        }


class InnerVertex(Vertex):
    def __init__(
            self,
            name: str,
            module: torch.nn.Module,
            predecessors: List[Vertex],
    ):
        super().__init__(name=name)
        self._module = module
        self._predecessors = list(predecessors)
        self.dag_module: "DagModule" = None
        self.orbit = None

    @property
    def successors(self) -> List['InnerVertex']:
        if self.dag_module is None:
            logger.error(f'Trying to get successors of an InnerVertex that has not been assigned to any DagModule.')
        return [vertex for vertex in self.dag_module.inner_vertices if self in vertex.predecessors]

    @property
    def predecessors(self) -> List[Vertex]:
        return self._predecessors

    @property
    def predecessor_indices(self) -> List[Vertex]:
        return [self.dag_module.vertices.index(pd) for pd in self.predecessors]

    @predecessors.setter
    def predecessors(self, new_predecessors: List[Vertex]):
        if not isinstance(new_predecessors, list):
            logger.error(f'Predecessors is expected to be a list. Got {type(new_predecessors)} except.')
        self._predecessors = new_predecessors

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    @module.setter
    def module(self, module: torch.nn.Module):
        self._module = module
        # TODO: Remove after validation
        self.dag_module.update_inner_modules()

    def config_dict(self, atomic_modules: List[torch.nn.Module]):
        is_atomic = not isinstance(self.module, DagModule)
        result = {
            'name':                self.name,
            'predecessor_indices': self.predecessor_indices,
            'is_atomic':           is_atomic,
            'type':                'inner',
            'orbit':               self.orbit,
        }
        if not is_atomic:
            result['module_dict'] = self.module.config_dict(atomic_modules)
        else:
            result['module_index'] = atomic_modules.index(self.module)
        return result


VertexProcessor = Callable[[InnerVertex], None]


@register_notrace_module
class DagModule(torch.nn.Module):
    MAX_LEN_REPR = None

    def __init__(
            self,
            name: str,
            vertices: Optional[List[Vertex]] = None,
            output_vertex: Optional[InnerVertex] = None,
    ):
        super().__init__()
        self.name = name
        self.vertices = vertices if vertices is not None else []
        self.output_vertex = output_vertex
        self.forward_dict = None
        self.inputs_dict = None
        self.cache_forward_dict = False
        self._inner_modules = None
        self.forward_scaffold = {}
        self.output_index = None
        self.compiled = False
        self.update_inner_modules()

    def compile(self, inputs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None):
        """
        In general `forward` method of DagModule is not `torch.compile` friendly. To overcome that
        we need to use a modified implementation of the forward pass, with no cacheing of intermediate tensors.
        Additionally, some modules may require a compile-type step for `torch.compile` usage.
        :param inputs: optional input (a dummy tensor for a single forward pass)
        """
        if inputs is not None:
            is_training = self.training
            if is_training:
                self.eval()
            _ = self(inputs)
            if is_training:
                self.train()

        self.forward_scaffold, self.output_index = self.get_forward_scaffold()
        for v in self.inner_vertices:
            if isinstance(v.module, DagModule):
                v.module.compile()
        self.compiled = True

    @property
    def inner_modules(self) -> torch.nn.ModuleList:
        self._inner_modules = torch.nn.ModuleList([vertex.module for vertex in self.inner_vertices])
        return self._inner_modules

    @property
    def input_vertices(self) -> List[InputVertex]:
        return [vertex for vertex in self.vertices if isinstance(vertex, InputVertex)]

    @property
    def inner_vertices(self) -> List[InnerVertex]:
        return [vertex for vertex in self.vertices if isinstance(vertex, InnerVertex)]

    def update_inner_modules(self):
        self._inner_modules = torch.nn.ModuleList([vertex.module for vertex in self.inner_vertices])
        for iv in self.inner_vertices:
            if isinstance(iv.module, DagModule):
                iv.module.update_inner_modules()

    def get_vertex_by_name(self, name: str) -> Union[InnerVertex, InputVertex]:
        result = [vertex for vertex in self.vertices if vertex.name == name]
        if len(result) == 1:
            return result[0]
        elif len(result) > 1:
            raise AssertionError(f'Multiple vertices found with name: {name} -> {result}')
        else:
            return

    def get_forward_scaffold(self):
        # a mapping between vertices index and its predecessors indices
        forward_scaffold = {}
        for k, vertex in enumerate(self.vertices):
            if isinstance(vertex, InputVertex):
                pass
            elif isinstance(vertex, InnerVertex):
                predecessors = vertex.predecessors
                predecessors_indices = [
                    self.vertices.index(pd) for pd in predecessors
                ]
                forward_scaffold[k] = predecessors_indices

        output_index = self.vertices.index(self.output_vertex)

        return forward_scaffold, output_index

    def compiled_forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[
        InnerVertex, Union[torch.Tensor, List[torch.Tensor]]]:

        assert self.compiled

        if not isinstance(inputs, List):
            inputs = [inputs]
        if len(inputs) != len(self.input_vertices):
            raise AssertionError

        forward_list = [None for _ in range(len(self.vertices))]

        for k, input in enumerate(inputs):
            forward_list[k] = input

        num_inputs = len(inputs)

        for k in range(len(self.vertices)):
            if k < num_inputs:
                pass
            else:

                pd_indices = self.forward_scaffold[k]
                module_inputs = [forward_list[pd_index] for pd_index in pd_indices]
                if len(module_inputs) == 1:
                    module_inputs = module_inputs[0]
                try:
                    result = self.vertices[k].module(module_inputs)
                except (TypeError, AttributeError):
                    result = self.vertices[k].module(*module_inputs)
                result = _postprocess_module_output(result)

                forward_list[k] = result

        return forward_list[self.output_index]

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[
        InnerVertex, Union[torch.Tensor, List[torch.Tensor]]]:
        # this is for `torch.compile` usage
        if self.compiled:
            return self.compiled_forward(inputs)

        if not isinstance(inputs, List):
            inputs = [inputs]
        if len(inputs) != len(self.input_vertices):
            raise AssertionError

        forward_dict = {}
        for k, v in enumerate(self.input_vertices):
            forward_dict[v] = inputs[k]

        # forward_dict = {vertex: tensor for vertex, tensor in zip(self.input_vertices, inputs)}
        inputs_dict = {}

        for vertex in self.vertices:
            if isinstance(vertex, InputVertex):
                pass
            elif isinstance(vertex, InnerVertex):
                predecessors = vertex.predecessors
                module_inputs = [forward_dict[pd] for pd in predecessors]
                inputs_dict[vertex] = module_inputs

                if len(module_inputs) == 1:
                    module_inputs = module_inputs[0]

                try:
                    result = vertex.module(module_inputs)
                except (TypeError, AttributeError):
                    result = vertex.module(*module_inputs)
                # if isinstance(result, Tuple):
                result = _postprocess_module_output(result)

                forward_dict[vertex] = result
        if self.cache_forward_dict:
            self.forward_dict = forward_dict
            self.inputs_dict = inputs_dict
        return forward_dict[self.output_vertex]

    def traverse(
            self,
            processor: VertexProcessor = None,
    ):
        if processor is None:
            inner_vertices = []
            for inner_vertex in self.inner_vertices:
                if isinstance(inner_vertex.module, DagModule):
                    inner_vertices.extend(inner_vertex.module.traverse())
                inner_vertices.append(inner_vertex)
            return inner_vertices
        else:
            for inner_vertex in self.traverse():
                processor(inner_vertex)
        # TODO: Remove after validation
        # self._update_inner_modules()

    def _check_if_name_unique(self, name: str):
        if name in [v.name for v in self.vertices]:
            raise ValueError(
                f'{self.name} already has an Vertex with name {name}. Please use different name.'
            )

    def add_input_vertex(self, name: str) -> InputVertex:
        self._check_if_name_unique(name)
        input_vertex = InputVertex(name)
        self.vertices.append(input_vertex)
        return input_vertex

    def add_vertex(
            self,
            name: str,
            module: torch.nn.Module,
            predecessors: List[Vertex],
    ) -> InnerVertex:
        self._check_if_name_unique(name)
        assert isinstance(module, torch.nn.Module)

        inner_vertex = InnerVertex(
            name=name,
            module=module,
            predecessors=predecessors,
        )
        for predecessor in predecessors:
            if predecessor not in self.vertices:
                raise ValueError(f'The predecessor: {predecessor} of InnerVertex: {InnerVertex} is not in '
                                 f'the DagModule: {self.name}')
        self.vertices.append(inner_vertex)
        self.inner_modules.append(module)
        inner_vertex.dag_module = self
        return inner_vertex

    def __repr__(self):
        representation = f'{self.__class__.__name__}[{self.name}]'
        if len(self.vertices) == 0:
            return representation
        for inner_vertex in self.inner_vertices:
            inner_vertex.MAX_LEN_REPR = self.MAX_LEN_REPR

        for vertex in self.vertices:
            if isinstance(vertex, InputVertex):
                representation += f'\n << {vertex.name} '
            else:
                index = self.inner_vertices.index(vertex)
                prefix = '>>' if vertex == self.output_vertex else '*'
                if isinstance(vertex.module, DagModule):
                    prefix += '#'
                representation += f"\n{prefix} {index}: {vertex} " \
                                  f"--> predecessors: {vertex.predecessors}, " \
                                  f"successors: {vertex.successors}"
                representation += f' {self.add_custom_module_info(vertex)}'
        for vertex in self.inner_vertices:
            vertex.MAX_LEN_REPR = None
        return representation

    def add_custom_module_info(self, vertex: InnerVertex):
        m = vertex.module
        if isinstance(m, torch.nn.Conv2d):
            return f'Conv2d(in={m.in_channels}, out={m.out_channels}, ks={m.kernel_size}, padding={m.padding})'
        if isinstance(m, torch.nn.Linear):
            return f'Linear(in={m.in_features}, out={m.out_features})'
        return ''

    def mark_current_top_vertex_as_output(self):
        if not self.inner_vertices:
            raise ValueError(f'One cannot mark top node in an empty {self}')
        if self.output_vertex is not None:
            logger.warning(f'{self} already has an output vertex. Replacing...')
        self.output_vertex = self.inner_vertices[-1]

    @property
    def module_classes(self) -> Set:
        return set([m.__class__ for m in self.inner_modules])

    def unroll_inner_modules(self) -> List[torch.nn.Module]:
        result = []
        for m in self.inner_modules:
            if not isinstance(m, DagModule):
                result.append(m)
            else:
                result.extend(m.unroll_inner_modules())
        return result

    def save(self, path: str):
        # TODO: Remove after validation
        # self._update_inner_modules()
        self.enforce_names_uniqueness()
        os.makedirs(path, exist_ok=True)
        atomic_modules = self.unroll_inner_modules()
        self.clear_custom_buffers()
        torch.save(torch.nn.ModuleList(atomic_modules), os.path.join(path, 'modules.pt'))
        with open(os.path.join(path, 'config.dict.json'), 'w') as f:
            json.dump(self.config_dict(), f)

    def clear_custom_buffers(self):
        for module in self.unroll_inner_modules():
            if hasattr(module, 'clear_custom_buffers'):
                module._buffers.clear()

    @classmethod
    def load(
            cls,
            path: str,
            map_location='cpu',
            custom_module_classes: Tuple[Type[torch.nn.Module]] = (),
    ) -> "DagModule":
        """

        :param path: directory from which model should be loaded
        :param map_location: defaults to `cpu`
        :param custom_module_classes: custom torch module classes needed for loading a `DagModule` that was built
        using these modules
        """
        with open(os.path.join(path, 'config.dict.json'), 'r') as f:
            config_dict = json.load(f)
        m = torch.load(os.path.join(path, 'modules.pt'), map_location=map_location)
        return cls.load_from_config_dict_and_atomic_modules(
            config_dict=config_dict,
            atomic_modules=m
        )

    @classmethod
    def load_from_config_dict_and_atomic_modules(
            cls,
            config_dict: Dict,
            atomic_modules: List[torch.nn.Module]
    ) -> "DagModule":
        output_index = config_dict.pop('output_index')
        name = config_dict.pop('name')
        if 'class' in config_dict:
            class_name = config_dict.pop('class')
        else:
            class_name = cls.__name__
        dag = None
        if class_name == cls.__name__:
            dag = cls(name)
        for subclass in cls.__subclasses__():
            if subclass.__name__ == class_name:
                dag = subclass(name)

        if dag is None:
            raise NotImplementedError(f'There is no subclass with name: {class_name}.')

        for k, (key, config) in enumerate(config_dict.items()):
            if config['type'] == 'input':
                dag.add_input_vertex(name=config['name'])
            else:
                predecessors = [dag.vertices[index] for index in config['predecessor_indices']]
                if config['is_atomic']:
                    module = atomic_modules[config['module_index']]
                else:
                    module = cls.load_from_config_dict_and_atomic_modules(
                        config_dict=config['module_dict'],
                        atomic_modules=atomic_modules,
                    )
                vertex = dag.add_vertex(
                    name=config['name'],
                    module=module,
                    predecessors=predecessors,
                )
                orbit = config.get('orbit')
                if orbit:
                    vertex.orbit = orbit
                if k == output_index:
                    dag.output_vertex = vertex

        return dag

    def config_dict(self, atomic_modules: List[torch.nn.Module] = None) -> Dict:
        if atomic_modules is None:
            atomic_modules = self.unroll_inner_modules()
        config_dict = {}
        for k, vertex in enumerate(self.vertices):
            _config = vertex.config_dict(atomic_modules)
            config_dict[k] = _config

        config_dict['name'] = self.name
        config_dict['class'] = self.__class__.__name__
        config_dict['output_index'] = self.vertices.index(self.output_vertex)
        return config_dict

    def _get_inner_vertex_predecessor_indices(self, inner_vertex: InnerVertex) -> List[int]:
        result = [
            self.vertices.index(predecessor)
            for predecessor in inner_vertex.predecessors
        ]
        return result

    @property
    def flat(self) -> bool:
        for v in self.inner_vertices:
            if isinstance(v.module, DagModule):
                return False
        return True

    def flatten(self, input_shape_for_verification: Optional[Tuple[int, ...]] = None) -> "DagModule":
        """
        This method will switch the `dag` to `eval` mode if `input_shape_for_verification` is provided.
        :param input_shape_for_verification:
        :return:
        """
        dag_copy = deepcopy(self)
        if self.flat:
            return dag_copy

        if input_shape_for_verification:
            dag_copy.eval()
            x = torch.normal(mean=torch.zeros(size=input_shape_for_verification))
            reference_output = dag_copy(x)

        # builds a new cell (not in place flatten)
        dag = self.__class__(name=dag_copy.name, vertices=dag_copy.input_vertices)
        for v in dag_copy.inner_vertices:
            if not isinstance(v.module, DagModule):
                dag.vertices.append(v)
                v.dag_module = dag
                if v == dag_copy.output_vertex:
                    dag.output_vertex = v
            else:
                inner_dag_predecessors = v.predecessors
                inner_dag_successors = v.successors
                inner_dag = v.module.flatten()
                for iv in inner_dag.inner_vertices:
                    for pd in iv.predecessors:  # remap predecessors where needed
                        if isinstance(pd, InputVertex):
                            pd_index_in_inner_dag = inner_dag.input_vertices.index(pd)
                            index = iv.predecessors.index(pd)
                            iv.predecessors[index] = inner_dag_predecessors[pd_index_in_inner_dag]
                    if inner_dag.output_vertex == iv:  # remap output of inner dag
                        for suc in inner_dag_successors:
                            index = suc.predecessors.index(v)
                            suc.predecessors[index] = iv
                    iv.dag_module = dag
                    dag.vertices.append(iv)
                    if v == dag_copy.output_vertex:
                        dag.output_vertex = iv
                    assert all([e in dag.vertices for e in iv.predecessors])

        if input_shape_for_verification:
            dag.eval()
            new_output = dag(x)
            if isinstance(reference_output, (list, tuple)):
                assert isinstance(new_output, (list, tuple))
                assert len(reference_output) == len(new_output)
                for ref, new in zip(reference_output, new_output):
                    assert torch.abs(ref - new).sum() == 0.0
            else:
                assert torch.abs(reference_output - new_output).sum() == 0.0


        # TODO: Remove after validation
        # self._update_inner_modules()
        dag.enforce_names_uniqueness()

        return dag

    def enforce_names_uniqueness(self):
        names = [v.name for v in self.vertices]
        while len(names) != len(set(names)):
            names_counter = Counter()
            for v in self.vertices:
                name = v.name
                names_counter[name] += 1
                if names_counter[name] > 1:
                    new_name = f'{name}_{names_counter[name] - 1}'
                    logger.debug(f'Renaming: {name} -> {new_name}')
                    v.name = new_name
            names = [v.name for v in self.vertices]

    def clear_tensor_dicts(self):
        self.forward_dict = None
        self.inputs_dict = None

    @property
    def device(self):
        # https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/10
        # useful, but may be dangerous
        self.update_inner_modules()
        device_ = next(iter(self.parameters())).device
        if not all([p.device == device_ for p in self.parameters()]):
            raise AssertionError(f'Not all parameters of {self.name} are on the same device')
        return device_


class TeacherModule(DagModule):
    pass


class StudentModule(DagModule):
    pass


@register_notrace_module
class BkdDagModule(DagModule):

    def __init__(
            self,
            name: str,
            vertices: Optional[List[Vertex]] = None,
            output_vertex: Optional[Vertex] = None,
            channel_dim: int = 1,
    ):
        """
        :param channel_dim: is needed for computing teacher-student losses between `TeacherModule` and
        `CandidateModule`s outputs.
        """
        super().__init__(name=name, vertices=vertices, output_vertex=output_vertex)
        self.channel_dim = channel_dim
        self.bkd_loss = {}

    @property
    def teacher_vertex(self) -> InnerVertex:
        return [vertex for vertex in self.inner_vertices if isinstance(vertex.module, TeacherModule)][0]

    @property
    def students_vertices(self) -> List[InnerVertex]:
        return [vertex for vertex in self.inner_vertices if isinstance(vertex.module, StudentModule)]

    @classmethod
    def build_for_teacher_and_students(
            cls,
            name: str,
            teacher: TeacherModule,
            students: List[StudentModule],
    ):
        input_vertices = [InputVertex(name=f'input_{k}') for k in range(len(teacher.inner_vertices))]
        dag = cls(name=name)

        teacher_vertex = dag.add_vertex(
            name=teacher.name,
            module=teacher,
            predecessors=input_vertices,
        )
        dag.output_vertex = teacher_vertex
        for candidate in students:
            _ = dag.add_vertex(
                name=candidate.name,
                module=candidate,
                predecessors=input_vertices,
            )

        return dag

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]):
        teacher_output = self.teacher_vertex.module(inputs)
        non_channel_dim_ = [e for e in range(len(teacher_output.shape))]
        non_channel_dim_.remove(self.channel_dim)
        non_channel_dim = tuple(non_channel_dim_)
        student_outputs = []
        for vertex in self.students_vertices:
            s_out = vertex.module[inputs]
            self.bkd_loss[vertex] = per_channel_noise_to_signal_ratio(
                y=teacher_output,
                x=s_out,
                non_channel_dim=non_channel_dim,
            )
            student_outputs.append(vertex.module[inputs])

        return teacher_output
