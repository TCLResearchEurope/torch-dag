import logging
from typing import Dict, Any, Optional, List

import torch.nn

from torch_dag.core.dag_module import InnerVertex, DagModule, InputVertex
from torch_dag.core.dag_module_utils import insert_before, wrap_sequence_in_dag_module


logger = logging.getLogger(__name__)


class VertexSpec:
    def __init__(
            self,
            module_class,
            name: Optional[str] = None,
            spec_dict: Optional[Dict[str, Any]] = None,
            num_predecessors: Optional[int] = None,
            num_successors: Optional[int] = None,
    ):
        self.module_class = module_class
        self.name = name
        self.spec_dict = spec_dict
        self.num_predecessors = num_predecessors
        self.num_successors = num_successors

    def match_vertex(self, vertex: InnerVertex) -> bool:
        module = vertex.module
        if not isinstance(module, self.module_class):
            return False
        if self.spec_dict:
            for k, v in self.spec_dict.items():
                if not hasattr(module, k):
                    return False
                else:
                    module_v = getattr(module, k)
                    if v != module_v:
                        return False
        if self.num_successors:
            if len(vertex.successors) != self.num_successors:
                return False
        if self.num_predecessors:
            if len(vertex.predecessors) != self.num_predecessors:
                return False
        return True

    def find_matched_vertices_in_dag(self, dag: DagModule) -> List[InnerVertex]:
        return self.find_matched_vertices_in_vertex_list(dag.inner_vertices)

    def find_matched_vertices_in_vertex_list(self, vertices: List[InnerVertex]) -> List[InnerVertex]:
        result = []
        for vertex in vertices:
            if self.match_vertex(vertex):
                result.append(vertex)
        return result


class ReverseGraphPattern:
    def __init__(
            self,
            pattern: Dict[str, List[str]],
            named_specs: Dict[str, VertexSpec],
            starting_key: str,
    ):
        self.pattern = pattern
        self.named_specs = named_specs
        self.starting_key = starting_key

    def match_vertex_subpattern(self, vertex_to_check: InnerVertex, subpattern_name: str):
        if any([isinstance(pd, InputVertex) for pd in vertex_to_check.predecessors]):
            return False, None, None
        predecesors_spec_names = self.pattern[subpattern_name]
        vertex_spec = self.named_specs[subpattern_name]
        predecessor_vertex_specs = [self.named_specs[name] for name in predecesors_spec_names]
        vertex_match_result = vertex_spec.match_vertex(vertex_to_check)
        predecessors_match_result = all([
            spec.match_vertex(v) for spec, v in zip(predecessor_vertex_specs, vertex_to_check.predecessors)
        ])
        matched = vertex_match_result and predecessors_match_result
        return matched, predecesors_spec_names, vertex_to_check.predecessors

    def match_overall_pattern(self, vertex: InnerVertex):
        final_matched = True
        collected_vertices = [vertex, ]
        queue = [(self.starting_key, vertex), ]
        while len(queue) > 0:
            subpattern_name, vertex = queue.pop()
            matched, names, vertices = self.match_vertex_subpattern(
                vertex_to_check=vertex, subpattern_name=subpattern_name)
            if not matched:
                final_matched = False
            else:
                collected_vertices.extend(vertices)
                for name, v in zip(names, vertices):
                    if name in self.pattern:
                        queue.append((name, v))
        if final_matched:
            return collected_vertices


class LinearGraphPattern:
    def __init__(
            self,
            pattern: List[str],
            named_specs: Dict[str, VertexSpec],
    ):
        self.pattern = pattern
        assert len(pattern) > 1
        self.named_specs = named_specs

    def match_overall_pattern(self, vertex: InnerVertex):
        if len(vertex.predecessors) != 1:
            return
        matched = self.named_specs[self.pattern[0]].match_vertex(vertex)
        if not matched or len(vertex.successors) != 1:
            return
        collected_vertices = [vertex]
        current_vertex = vertex.successors[0]
        require_single_successor = True
        counter = 0

        for pattern_name in self.pattern[1:]:
            counter += 1
            if counter == len(self.pattern) - 1:
                require_single_successor = False
            matched = self.named_specs[pattern_name].match_vertex(current_vertex)
            if not matched:
                return
            if len(current_vertex.successors) != 1 and require_single_successor:
                return
            collected_vertices.append(current_vertex)
            current_vertex = current_vertex.successors[0]

        return collected_vertices

    def find_all_patterns(self, dag: DagModule):
        all_patterns = []
        for vertex in dag.inner_vertices:
            result = self.match_overall_pattern(vertex)
            if result:
                all_patterns.append(result)
        logger.info(f'Found: {len(all_patterns)} patterns.')
        return all_patterns


def build_dag_from_sequential(name: str, sequential: List[InnerVertex]) -> DagModule:
    v = InputVertex(name='x')
    dag = DagModule(name=name, vertices=[v])
    for k, vertex in enumerate(sequential):
        v = dag.add_vertex(name=vertex.name, module=vertex.module, predecessors=[v])
    dag.output_vertex = v
    return dag


def replace_linear_pattern(
        name: str,
        pattern: List[InnerVertex],
        module_sequence: List[torch.nn.Module],
):
    """
    * possible multiple successors of pattern[-1]
    * required single predecessor of pattern[0]
    :param name:
    :param pattern:
    :param module_sequence:
    :return:
    """
    dag: DagModule = pattern[0].dag_module
    successors = pattern[-1].successors
    # for vertex in pattern:
    #     remove_vertex(dag, vertex)
    vertex = pattern[0]
    for k, module in enumerate(module_sequence):
        if k == 0:
            vertex = insert_before(
                dag=dag,
                name=f'{name}/{k}',
                reference_vertex=vertex,
                new_module=module,
            )
        else:
            prev_vertex = vertex
            # old_vertex = current_vertex
            vertex = InnerVertex(
                name=f'{name}/{k}',
                predecessors=[prev_vertex],
                module=module,
            )
            vertex.dag_module = dag
            reference_vertex_index = dag.vertices.index(prev_vertex)
            dag.vertices.insert(reference_vertex_index + 1, vertex)
            # vertex.predecessors = [old_vertex]

    for suc in successors:
        pd_index = suc.predecessors.index(pattern[-1])
        suc.predecessors[pd_index] = vertex

    for v in pattern:
        dag.vertices.remove(v)

def replace_linear_pattern_with_dag(
        name: str,
        pattern: List[InnerVertex],
) -> DagModule:
    """
    * possible multiple successors of pattern[-1]
    * required single predecessor of pattern[0]
    :param name:
    :param pattern:
    :param module_sequence:
    :return:
    """
    dag: DagModule = pattern[0].dag_module
    successors = pattern[-1].successors
    reference_vertex_index = dag.vertices.index(pattern[0])
    new_dag = wrap_sequence_in_dag_module(name=name, sequence=pattern)
    new_vertex = InnerVertex(name=name, module=new_dag, predecessors=pattern[0].predecessors)

    dag.vertices.insert(reference_vertex_index, new_vertex)
    new_vertex.dag_module = dag

    for suc in successors:
        pd_index = suc.predecessors.index(pattern[-1])
        suc.predecessors[pd_index] = new_vertex

    for v in pattern:
        dag.vertices.remove(v)
    return new_dag
