from typing import List
from typing import Set
from typing import Tuple

from torch_dag_algorithms.pruning.orbits_search_stage import OrbitsDiscoveryStage
from torch_dag.core.dag_module import InnerVertex


class Orbit:
    def __init__(self, color: int):
        """Basic orbit object that can represent either extended or final orbit. If orbit has `allow_for_further_processing` set to True then it can be processed by Orbitalizer by it's general mechanism. If set to False orbit won't be processed in any way and will be passed to orbitalization algorithm in unchanged state.

        `_found_by` - indicates what stage lead to orbit being found. It's used in testing handling custom known patterns that are handled by hand. It also holds information that can be usefull durning debugging. 

        Args:
            color (int): orbit color. has to be unique
            allow_for_further_processing (bool, optional): If False orbit won't be process in any way. Defaults to True.
        """
        self.color = color

        self.vertices_in_scope: Set[InnerVertex] = set()
        self.sources: List[InnerVertex] = []
        self.sinks: List[InnerVertex] = []
        self.end_path: List[Tuple[InnerVertex, InnerVertex]] = []

        self.kmapps = None
        self._discovery_stage = None

    @property
    def discovery_stage(self) -> OrbitsDiscoveryStage:
        return self._discovery_stage

    @discovery_stage.setter
    def discovery_stage(self, val: OrbitsDiscoveryStage):
        if not self._discovery_stage:
            self._discovery_stage = val
        else:
            raise AttributeError('_found_by property is already set and cannot be changed.')

    @property
    def non_border(self) -> Set[InnerVertex]:
        return self.vertices_in_scope - (set(self.sources) | set(self.sinks))

    def __repr__(self):
        return f'\033[1m\033[95mOrbit\033[0m[\033[1m\033[93mcolor\033[0m={self.color}, \033[1m\033[93mdiscovery_stage\033[0m={self.discovery_stage}, \033[1m\033[93msources\033[0m={self.sources}, \033[1m\033[93msinks\033[0m={self.sinks}, \033[1m\033[93mnon_border\033[0m={self.non_border}, \033[1m\033[93mend_path\033[0m={self.end_path}]'

    def __iter__(self):
        yield from self.vertices_in_scope

    def __len__(self):
        return len(self.vertices_in_scope)

    def __eq__(self, other: 'Orbit'):
        sources_equal = (set(self.sources) == set(other.sources))
        sinks_equal = (set(self.sinks) == set(other.sinks))
        non_border_equal = self.non_border == other.non_border

        return sources_equal and sinks_equal and non_border_equal

    def add_to_scope(self, vertex: InnerVertex):
        self.vertices_in_scope.add(vertex)

    def mark_as_source(self, vertex: InnerVertex):
        if vertex not in self.sources:
            self.sources += [vertex]

    def mark_as_sink(self, vertex: InnerVertex):
        if vertex not in self.sinks:
            self.sinks += [vertex]

    def mark_end_path_node_and_sink(self, end_vertex: InnerVertex, sink_vertex: InnerVertex):
        self.end_path += [(end_vertex, sink_vertex)]

    def is_valid(self, orbit_filters: List['OrbitFilter']) -> bool:
        for orbit_filter in orbit_filters:
            include = orbit_filter.filter(self)
            if not include:
                return False

        return True
