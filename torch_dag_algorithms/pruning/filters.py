import logging
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple, Type
import torch

from torch import nn

from torch_dag import structured_modules as smodule
from torch_dag_algorithms.pruning.orbit import Orbit
from torch_dag.core.dag_module import DagModule, InnerVertex, InputVertex
from torch_dag.core.module_handling_for_pruning import unprunable_modules

logger = logging.getLogger(__name__)


class OrbitFilter(ABC):
    def __repr__(self):
        return self.__class__.__name__

    def filter(self, orbit: Orbit) -> bool:
        # True if orbit is valid else False
        include = self._filter(orbit=orbit)

        if not include:
            logger.info(f'[\033[1m\033[96m{self.__class__.__name__}\033[0m] Removing orbit {orbit}')

        return include

    @abstractmethod
    def _filter(self, orbit: Orbit) -> bool:
        raise NotImplementedError


class StemPresentFilter(OrbitFilter):
    STEM_TYPES = (nn.Conv2d,)

    def __init__(self, dag: DagModule, stem_types: Tuple[nn.Module] = None):
        self.stem_types = stem_types if stem_types else self.STEM_TYPES
        self.stem = self.find_stem(dag)

    def find_stem(self, dag: DagModule) -> List[InnerVertex]:
        stem = []
        for vertex in dag.inner_vertices:
            if isinstance(vertex.module, self.stem_types) and any(
                    [isinstance(p, InputVertex) for p in vertex.predecessors]):
                stem += [vertex]

        return stem

    def _filter(self, orbit: Orbit) -> bool:
        if self.stem:
            for vertex in orbit.sources:
                if isinstance(vertex, InnerVertex) and vertex in self.stem:
                    return False

        return True


class OutputInScopeFilter(OrbitFilter):
    def __init__(self, output_vertex: InnerVertex):
        self.output_vertex = output_vertex

    def _filter(self, orbit: Orbit) -> bool:
        for vertex in orbit.vertices_in_scope:
            if vertex == self.output_vertex:
                return False

        return True


class InputInScopeFilter(OrbitFilter):
    def __init__(self, input_vertices: List[InputVertex]):
        self.input_vertices = input_vertices

    def _filter(self, orbit: Orbit) -> bool:
        for vertex in orbit.vertices_in_scope:
            if vertex in self.input_vertices:
                return False

        return True


class JoinOpAfterConcatPresentFilter(OrbitFilter):
    FORBIDDEN_AFTER_CONCAT = (smodule.SubModule, smodule.AddModule, smodule.MulModule)

    def __init__(self, forbidden_after_concat: Tuple[nn.Module] = None):
        self.forbidden_after_concat = forbidden_after_concat if forbidden_after_concat else self.FORBIDDEN_AFTER_CONCAT

    def _search_for_join_op(self, orbit: Orbit, vertex: InnerVertex):
        if vertex in orbit.vertices_in_scope and isinstance(vertex.module, self.forbidden_after_concat):
            return False

        if vertex in orbit.sinks:
            return True

        if vertex.successors:
            for successor in vertex.successors:
                if not self._search_for_join_op(orbit, successor):
                    return False

        return True

    def _filter(self, orbit: Orbit) -> bool:
        for vertex in orbit.non_border:
            if isinstance(vertex, InnerVertex) and isinstance(vertex.module, smodule.ConcatModule):
                if not self._search_for_join_op(orbit, vertex):
                    return False

        return True


class NonPrunableCustomModulesFilter(OrbitFilter):

    def __init__(self, custom_unprunable_modules: Tuple[Type[torch.nn.Module]] = (),):
        self.custom_unprunable_modules = custom_unprunable_modules

    def _filter(self, orbit: Orbit) -> bool:
        if any([
            isinstance(vertex, InnerVertex) and isinstance(vertex.module, tuple(unprunable_modules) + self.custom_unprunable_modules)
            for vertex in orbit.vertices_in_scope
        ]):
            return False

        return True


class ProperGroupedConvolutionPresent(OrbitFilter):
    """Filter out 'proper' grouped convolutions (the ones where the number of groups is not 1 and not
    the full number of input channels
    """

    def _filter(self, orbit: Orbit) -> bool:
        for sink in orbit.sinks:
            if isinstance(sink.module, nn.Conv2d):
                if 1 < sink.module.groups < sink.module.in_channels:
                    return False
        for source in orbit.sources:
            if isinstance(source.module, nn.Conv2d):
                if 1 < source.module.groups < source.module.in_channels:
                    return False

        return True
