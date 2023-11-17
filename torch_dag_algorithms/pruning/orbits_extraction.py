from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from torch import nn

from torch_dag.core.dag_module import InnerVertex, InputVertex, Vertex
from torch_dag_algorithms.pruning.commons import is_sink, is_source
from torch_dag_algorithms.pruning.orbit import Orbit


def create_color_generator():
    i = 1
    while True:
        yield i
        i += 1


COLOR_GENERATOR = create_color_generator()


@dataclass
class NodeColor:
    incoming_color: int = None
    outgoing_color: int = None


class Coloring:
    """
    Class that holds information about incoming and outgoing colors for each node in set of inner_cell_nodes. 
    INCOMING - incoming edges color
    OUTGOING - outcoming edges color
    """

    def __init__(self, vertices: List[InnerVertex]):
        self.colors = self.init_colors(vertices)

    def init_colors(self, vertices: List[InnerVertex]) -> Dict[InnerVertex, NodeColor]:
        return {vertex: NodeColor() for vertex in vertices}

    def has_incoming_color(self, vertex: InnerVertex) -> bool:
        return True if self.colors[vertex].incoming_color else False

    def has_outgoing_color(self, vertex: InnerVertex) -> bool:
        return True if self.colors[vertex].outgoing_color else False

    def in_scope(self, vertex: InnerVertex) -> bool:
        return vertex in self.colors

    def set_incoming_color(self, vertex: InnerVertex, color):
        self.colors[vertex].incoming_color = color

    def set_outgoing_color(self, vertex: InnerVertex, color):
        self.colors[vertex].outgoing_color = color

    def incoming_color_equal_to(self, vertex: InnerVertex, other_color) -> bool:
        return self.colors[vertex].incoming_color == other_color

    def outgoing_color_equal_to(self, vertex: InnerVertex, other_color) -> bool:
        return self.colors[vertex].outgoing_color == other_color

    def incoming_color_is_empty(self, vertex: InnerVertex) -> bool:
        return self.in_scope(vertex) and not self.has_incoming_color(vertex)

    def outgoing_color_is_empty(self, vertex: InnerVertex) -> bool:
        return self.in_scope(vertex) and not self.has_outgoing_color(vertex)

    def can_go_backward(self, vertex: InnerVertex, color: int) -> bool:
        return self.in_scope(vertex) and self.incoming_color_equal_to(vertex, color)

    def can_go_forward(self, vertex: InnerVertex, color: int) -> bool:
        return self.in_scope(vertex) and self.outgoing_color_equal_to(vertex, color)

    def available_predecessors(self, vertex: InnerVertex, color: int) -> List[InnerVertex]:
        if self.can_go_backward(vertex, color):
            return vertex.predecessors

        return []

    def available_successors(self, vertex: InnerVertex, color: int) -> List[InnerVertex]:
        if self.can_go_forward(vertex, color):
            return vertex.successors

        return []


class TypeChecker:
    def __init__(
            self,
            truncate_on: Union[nn.Module, Tuple[nn.Module]]
    ):
        self.truncate_on = truncate_on

    def is_source_type(self, vertex: InnerVertex) -> bool:
        return is_source(vertex.module)

    def is_sink_type(self, vertex: InnerVertex) -> bool:
        return is_sink(vertex.module)

    def should_truncate(self, vertex: InnerVertex) -> bool:
        return self.truncate_on and isinstance(vertex.module, self.truncate_on)

    @staticmethod
    def is_input_type(vertex: Vertex) -> bool:
        return isinstance(vertex, InputVertex)


def find_nodes_in_orbit_scope(
        vertex: Vertex,
        orbit: Orbit,
        coloring: Coloring,
        type_checker: TypeChecker
) -> Orbit:
    """
    Searches for nodes that belong to orbit scope and mark it's sinks and sources. Search begins from node `vertex`.
    """
    for predecessor in coloring.available_predecessors(vertex, orbit.color):
        if type_checker.is_input_type(predecessor):
            orbit.add_to_scope(predecessor)
            continue
        if coloring.outgoing_color_is_empty(predecessor):
            orbit.add_to_scope(predecessor)
            coloring.set_outgoing_color(predecessor, orbit.color)
            if type_checker.is_source_type(predecessor):
                orbit.mark_as_source(predecessor)
            else:
                coloring.set_incoming_color(predecessor, orbit.color)
            find_nodes_in_orbit_scope(predecessor, orbit, coloring, type_checker)

    for successor in coloring.available_successors(vertex, orbit.color):
        if coloring.incoming_color_is_empty(successor):
            if type_checker.should_truncate(successor):
                orbit.mark_end_path_node_and_sink(end_vertex=vertex, sink_vertex=successor)
                continue
            orbit.add_to_scope(successor)
            coloring.set_incoming_color(successor, orbit.color)
            if type_checker.is_sink_type(successor):
                orbit.mark_as_sink(successor)
                orbit.mark_end_path_node_and_sink(end_vertex=vertex, sink_vertex=successor)
            else:
                coloring.set_outgoing_color(successor, orbit.color)
            find_nodes_in_orbit_scope(successor, orbit, coloring, type_checker)

    return orbit


def extract_orbits(
        vertices: List[Union[InnerVertex, InputVertex]],
        sources: Optional[List[InnerVertex]] = None,
        truncate_on: Optional[Union[nn.Module, Tuple[nn.Module]]] = None,
        discovery_stage: str = None
) -> List[Orbit]:
    """Function that extract orbits from given inner cell nodes. 
    If `sources` is specified then only those nodes will be treated as source candidates.
    If `truncate_on` is specified then during search, orbit will be truncated on truncate_on type.

    Args:
        -vertices (List[InnerVertex]): Set of nodes within which orbits will be searched for.
        -source_types (Tuple[nd.nodes.Node]): types of nodes that will be treated as source. Defaults to (nd.ops.Conv2D, nd.ops.Dense).
        -sink_types (Tuple[nd.nodes.Node]): types of nodes that will be treated as sink. Defaults to (nd.ops.Conv2D, nd.ops.Dense).
        -sources (Optional[List[InnerVertex]], optional): Nodes that should be treated as source candidates. Defaults to None. If not specified then all nodes are treated as source candidates.
        -truncate_on (Optional[Union[nd.nodes.Node, Tuple[nd.nodes.Node]]], optional): Type on which orbit should be truncated. Defaults to None. If specified then node of type `truncate_on` will not be included in orbit.
        -discovery_stage (str): If not None, `found_by` attribute in Orbit will be set to `discovery_stage` value. It will hold the information when orbit was found if orbits search in conducted in multiple steps.

    Returns:
        List[Orbit]: List of found orbits
    """
    sources = sources if sources else vertices
    coloring = Coloring(vertices=vertices)
    type_checker = TypeChecker(truncate_on=truncate_on)

    orbits: List[Orbit] = []
    for vertex in sources:
        if not type_checker.is_input_type(vertex) and type_checker.is_source_type(vertex):
            if not coloring.has_outgoing_color(vertex):
                color = next(COLOR_GENERATOR)
                coloring.set_outgoing_color(vertex, color)
                orbit = Orbit(color=color)
                if discovery_stage:
                    orbit.discovery_stage = discovery_stage
                orbit.add_to_scope(vertex)
                orbit.mark_as_source(vertex)
                find_nodes_in_orbit_scope(vertex, orbit, coloring, type_checker)
                orbits += [orbit]

    return orbits
