import abc
import sys
import copy
import logging
from typing import List, Type
from typing import Tuple
from typing import Union
from typing import Iterator
from typing import Optional

import torch
from torch_dag.core.dag_module import DagModule
from torch_dag.core.dag_module import InnerVertex, InputVertex
from torch_dag.patterns_v2.match import Match, SubgraphMatch, LinearMatch
from torch_dag.patterns_v2.pattern_node import PatternNode


logger = logging.getLogger(__name__)


class Pattern(abc.ABC):
    def __iter__(self) -> Iterator[PatternNode]:
        yield from self.pattern_nodes

    @abc.abstractmethod
    def search_at_node(self, start_search_at: DagModule) -> List[Match]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pattern_nodes(self) -> List[PatternNode]:
        raise NotImplementedError

    @property
    def pattern_node_names(self) -> List[str]:
        return [pn.name for pn in self.pattern_nodes]

    def search(self, inner_vertexes: InnerVertex) -> List[Match]:
        matches = []
        for iv in inner_vertexes:
            match = self.search_at_node(start_search_at=iv)
            if match:
                logger.info(f'[+] Found match {match} at {iv}')
                matches += [match]

        return matches


class LinearPattern(Pattern):
    def __init__(self, pattern: List[PatternNode]):
        self.pattern = pattern[::-1]

    @classmethod
    def from_ops(cls, ops: List[Type[torch.nn.Module]]) -> "LinearPattern":
        pattern = [PatternNode(op_type=op) for op in ops]

        return cls(pattern=pattern)

    def __repr__(self):
        return f'{self.__class__.__name__}{self.pattern}'

    @property
    def pattern_nodes(self):
        return self.pattern

    def is_complete(self, current_match: LinearMatch) -> bool:
        return len(current_match) == len(self.pattern)

    def op_matches_to_pattern(self, current_match: LinearMatch, op: torch.nn.Module) -> Tuple[bool, bool]:
        pattern_node = self.pattern[len(current_match)]
        type_match = isinstance(op, pattern_node.op_type)
        if pattern_node.extra_spec:
            return type_match and pattern_node.spec_compliant(op), pattern_node.mandatory

        return type_match, pattern_node.mandatory

    def _search_for_pattern_backward(
            self,
            cn: Union[InputVertex, InnerVertex],
            current_match: LinearMatch
    ) -> Union[LinearMatch, None]:
        complete = self.is_complete(current_match)

        if isinstance(cn, InputVertex):
            pattern_left = self.pattern_nodes[len(current_match):]
            only_non_mandatory_left = all([not pn.mandatory for pn in pattern_left])
            if complete:
                return current_match
            elif only_non_mandatory_left:
                for _ in pattern_left:
                    current_match.add(None)
                return current_match
            return

        if not complete:
            type_match, mandatory = self.op_matches_to_pattern(current_match, cn.module)

            if not mandatory and not type_match:
                current_match.add(None)
                self._search_for_pattern_backward(cn, current_match)

            if type_match:
                current_match.add(cn)
                for predecessor in cn.predecessors:
                    self._search_for_pattern_backward(predecessor, current_match)

        # re-evaluate match after recursion
        if self.is_complete(current_match):
            return current_match

    def search_at_node(self, start_search_at: InnerVertex) -> Optional[LinearMatch]:
        if isinstance(start_search_at, InnerVertex):
            current_match = LinearMatch(names=self.pattern_node_names)
            type_match, mandatory = self.op_matches_to_pattern(current_match, start_search_at.module)
            if type_match and mandatory:
                current_match.add(start_search_at)
                for predecessor in start_search_at.predecessors:
                    match = self._search_for_pattern_backward(predecessor, current_match)
                    if match:
                        match.remove_missing_match_for_non_mandatory()

                        return match
            elif not type_match and not mandatory:
                current_match.add(None)
                match = self._search_for_pattern_backward(start_search_at, current_match)
                if match:
                    match.remove_missing_match_for_non_mandatory()

                    return match


class SubgraphPattern(Pattern):
    def __init__(self, pattern: List[PatternNode], inputs: Optional[List['SubgraphPattern']] = None):
        if inputs is None:
            inputs = []
        self.pattern = LinearPattern(pattern)
        self.inputs = inputs

    @classmethod
    def from_ops(cls, pattern_ops: List[Type[torch.nn.Module]], input_ops: Optional[List[Type[torch.nn.Module]]] = None):
        if input_ops is None:
            input_ops = []
        pattern = [PatternNode(op) for op in pattern_ops]
        inputs = [SubgraphPattern.from_ops([input_op]) for input_op in input_ops]

        return cls(pattern, inputs)

    @classmethod
    def from_ops_and_input_patterns(
        cls, 
        pattern_ops: List[Type[torch.nn.Module]],
        inputs: Optional[List['SubgraphPattern']] = None
    ):
        pattern = [PatternNode(op) for op in pattern_ops]

        return cls(pattern, inputs)

    def __repr__(self):
        return f'{self.__class__.__name__}[pattern={self.pattern}, inputs={self.inputs}]'

    @property
    def pattern_nodes(self) -> List[PatternNode]:
        return [pn for p in self.all_subpatterns for pn in p.pattern]

    @property
    def all_subpatterns(self):
        def traverse(current: SubgraphPattern, subpatterns: List[SubgraphPattern]):
            subpatterns += [current]
            for inp in current.inputs:
                traverse(inp, subpatterns)

            return subpatterns

        return traverse(self, [])

    def is_complete(self, current_match: SubgraphMatch) -> bool:
        return len(current_match.all_submatches) == len(self.all_subpatterns)

    def _search_for_pattern_backward(self, current_match: SubgraphMatch) -> SubgraphMatch:
        current_match.init_inputs(self.inputs)
        predecessors_to_check = copy.copy(current_match.match.start[0].predecessors)
        for i, inp in enumerate(self.inputs):
            for predecessor in predecessors_to_check:
                match = inp.pattern.search_at_node(predecessor)
                if match:
                    current_match.inputs[i].set_match(match)
                    inp._search_for_pattern_backward(current_match.inputs[i])
                    predecessors_to_check.remove(predecessor)
                    break

        return current_match

    def search_at_node(self, start_search_at: Union[InputVertex, InnerVertex]) -> Optional[SubgraphMatch]:
        current_match = SubgraphMatch(names=self.pattern_node_names)
        match = self.pattern.search_at_node(start_search_at)

        if match:
            current_match.set_match(match)
            self._search_for_pattern_backward(current_match)

            if self.is_complete(current_match):
                return current_match
