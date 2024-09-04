import copy
import logging
from typing import List, Tuple

import torch

from torch_dag.core.dag_module import DagModule, InnerVertex, InputVertex
from torch_dag.structured_modules import EmptyModule, ParameterModule, ArgModule

logger = logging.getLogger(__name__)


def remove_vertex(dag: DagModule, vertex: InnerVertex):
    """
    A method to remove an inner vertex from DagModule. If in the computational DAG the `vertex`
    occurs in a patter: prev_vertex -> `vertex` -> next_vertex, then removing `vertex` will result in:
    prev_vertex -> next_vertex. The method will NOT work for a `vertex` with multiple predecessors,
    :param dag:
    :param vertex:
    :return:
    """
    predecessors = vertex.predecessors
    successors = vertex.successors
    if len(predecessors) > 1:
        raise AssertionError(f'Vertex removal works only for single predecessor inner vertices.')
    if predecessors:
        predecessor = predecessors[0]
    for successor in successors:
        index = successor.predecessors.index(vertex)
        if predecessors:
            successor.predecessors[index] = predecessor
        else:
            successor.predecessors.remove(vertex)
    dag.vertices.remove(vertex)
    if vertex == dag.output_vertex:
        dag.output_vertex = vertex.predecessors[0]
    # TODO: Remove after validation
    dag.update_inner_modules()
    return vertex


def check_if_flat_dag_has_redundant_vertices(dag: DagModule) -> bool:
    for vertex in dag.inner_vertices:
        if isinstance(vertex.module, EmptyModule):
            return True
        if len(vertex.successors) == 0 and vertex != dag.output_vertex:
            return True
        if len(vertex.predecessors) == 0 and not isinstance(vertex.module, (ParameterModule, ArgModule)):
            return True
    return False


def recursively_remove_redundant_vertices(dag: DagModule):
    while check_if_flat_dag_has_redundant_vertices(dag):
        for vertex in dag.inner_vertices:
            # no predecessors but not a ParameterModule
            if len(vertex.predecessors) == 0 and not isinstance(vertex.module, (ParameterModule, ArgModule)):
                vertex.module = EmptyModule()
            # no successors and not an output vertex
            if len(vertex.successors) == 0 and vertex != dag.output_vertex:
                vertex.module = EmptyModule()
            if isinstance(vertex.module, EmptyModule):
                for sc in vertex.successors:
                    sc.predecessors.remove(vertex)
                logger.info(f'Deleting vertex: {vertex}')
                if vertex == dag.output_vertex:
                    dag.output_vertex = None
                dag.vertices.remove(vertex)
                # TODO: Remove after validation
                dag.update_inner_modules()

            if isinstance(vertex.module, DagModule):
                recursively_remove_redundant_vertices(vertex.module)

            if isinstance(vertex.module, torch.nn.Identity):
                logger.info(f'Removing identity activation: {vertex}')
                remove_vertex(dag, vertex)


def remove_redundant_vertices_from_flat_dag(dag: DagModule):
    assert dag.flat
    return recursively_remove_redundant_vertices(dag)


def insert_before(
        dag: DagModule,
        reference_vertex: InnerVertex,
        name: str,
        new_module: torch.nn.Module,
):
    if reference_vertex not in dag.vertices:
        raise AssertionError
    if len(reference_vertex.predecessors) > 1:
        raise AssertionError
    logger.debug(f'Inserting new vertex: {name} node with module {new_module} after a '
                 f'vertex: {reference_vertex}.')
    reference_vertex_index = dag.vertices.index(reference_vertex)
    logger.debug(f'Inserting new vertex: {name} node with module {new_module} before a '
                 f'vertex: {reference_vertex}.')
    new_vertex = InnerVertex(
        name=name,
        predecessors=reference_vertex.predecessors,
        module=new_module,
    )
    new_vertex.dag_module = dag
    reference_vertex.predecessors = [new_vertex]
    dag.vertices.insert(reference_vertex_index, new_vertex)
    # TODO: Remove after validation
    dag.update_inner_modules()
    return new_vertex


def insert_after(
        dag: DagModule,
        reference_vertex: InnerVertex,
        name: str,
        new_module: torch.nn.Module,
):
    if reference_vertex not in dag.vertices:
        raise AssertionError(f'Vertex: {reference_vertex} is not in DagModule: {dag.name}')
    logger.debug(f'Inserting new vertex: {name} node with module {new_module} after an '
                 f'vertex: {reference_vertex}.')
    reference_vertex_index = dag.vertices.index(reference_vertex)
    successors = reference_vertex.successors

    new_vertex = InnerVertex(
        name=name,
        predecessors=[reference_vertex],
        module=new_module,
    )
    dag.vertices.insert(reference_vertex_index + 1, new_vertex)
    new_vertex.dag_module = dag
    if reference_vertex == dag.output_vertex:
        dag.output_vertex = new_vertex
    else:
        for successor in successors:
            pd_index = successor.predecessors.index(reference_vertex)
            successor.predecessors[pd_index] = new_vertex
    # TODO: Remove after validation
    dag.update_inner_modules()
    return new_vertex


def insert_between(
        dag: DagModule,
        name: str,
        after_vertex: InnerVertex,
        new_module: torch.nn.Module,
        before_vertex: InnerVertex,
):
    if before_vertex not in after_vertex.successors:
        raise AssertionError(f'Vertex {before_vertex} is not a successor of {after_vertex}')

    vertex = InnerVertex(
        name=name,
        predecessors=[after_vertex],
        module=new_module,
    )
    before_vertex.predecessors[before_vertex.predecessors.index(after_vertex)] = vertex
    vertex.dag_module = dag
    before_index = dag.vertices.index(before_vertex)
    dag.vertices.insert(before_index, vertex)
    # TODO: Remove after validation
    dag.update_inner_modules()
    logger.debug(f'Inserted vertex: {vertex} between {after_vertex} and {before_vertex}.')
    return vertex


def wrap_sequence_in_dag_module(name: str, sequence: List[InnerVertex]) -> DagModule:
    input_vertices = [InputVertex(name=f'x_{k}') for k in range(len(sequence[0].predecessors))]
    dag = DagModule(name=name, vertices=input_vertices)

    # process first vertex in the `sequence`
    new_vertex = dag.add_vertex(name=sequence[0].name, predecessors=input_vertices, module=sequence[0].module)
    for old_vertex in sequence[1:]:
        new_vertex = dag.add_vertex(name=old_vertex.name, predecessors=[new_vertex], module=old_vertex.module)
    dag.output_vertex = new_vertex
    return dag


def wrap_subgraph_of_dag_module(
        dag: DagModule,
        end_vertex: InnerVertex,
        begin_vertex: InnerVertex,
        subgraph_name: str = None,
        input_shape_for_validation: Tuple[int, ...] = None,
        allow_subgraphs_inside: bool = False,
):
    if subgraph_name is None:
        subgraph_name = end_vertex.name
    if input_shape_for_validation:
        dag_copy = copy.deepcopy(dag)
    if end_vertex == dag.output_vertex:
        raise NotImplementedError
    final_paths = []
    backwards_paths = {(end_vertex,)}
    while len(backwards_paths) > 0:
        path = backwards_paths.pop()
        new_paths = {(pd,) + path for pd in path[0].predecessors}
        for new_path in new_paths:
            if new_path[0] == begin_vertex:
                final_paths.append(new_path)

            elif len(new_path[0].predecessors) == 0:  # a module without input
                final_paths.append(new_path)
            # check for invalid paths
            elif isinstance(new_path[0], InputVertex):
                raise AssertionError(f'There are paths ending in `end_vertex`: {end_vertex} '
                                     f'that do not go through `end_vertex`: {end_vertex}')
            else:
                backwards_paths.add(new_path)

    max_path_length = max([len(path) for path in final_paths])

    # validate that we are actually enclosing a self-contained subgraph
    forward_paths = {(begin_vertex,)}

    while len(forward_paths) > 0:
        path = forward_paths.pop()
        new_paths = [path + (suc,) for suc in path[-1].successors]
        for new_path in new_paths:
            if new_path[-1] == end_vertex:
                pass
            elif new_path[-1] == dag.output_vertex or len(new_path) > max_path_length:
                raise AssertionError(f'There are paths starting in `begin_vertex` and ending in output vertex for '
                                     f'the dag that do not contain `end_vertex`.')
            else:
                forward_paths.add(new_path)

    all_vertices = set([v for path in final_paths for v in path])

    topologically_sorted_vertices = sorted(all_vertices, key=lambda x: dag.vertices.index(x), reverse=False)
    if any([isinstance(v.module, DagModule) for v in topologically_sorted_vertices]) and not allow_subgraphs_inside:
        return

    subgraph_predecessors = begin_vertex.predecessors
    subgraph_successors = end_vertex.successors
    index_per_successor = [suc.predecessors.index(end_vertex) for suc in subgraph_successors]
    new_input_vertices = [InputVertex(name=f'{subgraph_name}/input_{k}') for k in range(len(subgraph_predecessors))]

    for vertex in topologically_sorted_vertices:
        vertex.dag_module = None
        dag.vertices.remove(vertex)

    subgraph_insertion_index = max([dag.vertices.index(pd) for pd in begin_vertex.predecessors]) + 1

    subgraph = DagModule(name=subgraph_name, vertices=new_input_vertices)

    begin_vertex.dag_module = subgraph
    begin_vertex.predecessors = subgraph.input_vertices
    end_vertex.dag_module = subgraph
    subgraph.vertices.append(begin_vertex)
    subgraph.output_vertex = end_vertex

    for vertex in topologically_sorted_vertices:
        vertex.dag_module = subgraph
        if vertex not in subgraph.vertices:
            subgraph.vertices.append(vertex)

    subgraph_inner_vertex = InnerVertex(name=subgraph_name, module=subgraph, predecessors=subgraph_predecessors)
    subgraph_inner_vertex.dag_module = dag
    subgraph_inner_vertex.module = subgraph
    dag.vertices.insert(subgraph_insertion_index, subgraph_inner_vertex)

    for k, suc in zip(index_per_successor, subgraph_successors):
        suc.predecessors[k] = subgraph_inner_vertex

    # TODO: Remove after validation
    dag.update_inner_modules()

    if input_shape_for_validation:
        dag.eval()
        x = torch.normal(mean=torch.zeros(size=input_shape_for_validation))
        original_output = dag_copy(x)
        new_output = dag(x)
        diff = torch.abs(original_output - new_output).sum()
        if diff > 0.0:
            logger.info(f'MAE loss: {diff}')
            raise AssertionError('Block wrapping resulted in divergent outputs.')

    logger.info(f'Finished wrapping {subgraph_name} into a subgraph.')
    return subgraph


def compare_module_outputs(
        first_module: torch.nn.Module,
        second_module: torch.nn.Module,
        input_shape: Tuple[int, ...],
        atol=1e-6,
):
    first_module.eval()
    second_module.eval()
    x = torch.normal(mean=torch.zeros(size=input_shape))
    y1 = first_module(x)
    y2 = second_module(x)
    if not isinstance(y1, (list, tuple)):
        y1 = [y1]
        y2 = [y2]

    for t1, t2 in zip(y1, y2):
        diff = torch.abs(t1 - t2).mean()
        if diff > atol:
            raise AssertionError(f'The outputs of first_module and second_module '
                                 f'are not equal. atol: {atol}, diff: {diff}')


def in_place_remove_traverser(v: InnerVertex):
    if isinstance(v.module, (torch.nn.Hardswish, torch.nn.ReLU6, torch.nn.ReLU)):
        if hasattr(v.module, 'inplace'):
            logger.info(f'Setting `inplace` to False in {v}')
            v.module.inplace = False


def remove_identity_traverser(v: InnerVertex):
    if isinstance(v.module, torch.nn.Identity):
        remove_vertex(dag=v.dag_module, vertex=v)
