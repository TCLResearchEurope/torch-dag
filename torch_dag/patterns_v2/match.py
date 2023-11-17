import abc
from typing import Dict
from typing import List
from typing import Iterator
from typing import Union

import torch
from torch_dag.core.dag_module import InnerVertex, InputVertex


class Match(abc.ABC):
    def __init__(self, names: List[str] = None):
        self._names = [] if not names else names

    def __iter__(self) -> Iterator[InnerVertex]:
        yield from self.matched_ivs

    def __len__(self):
        return len(self.matched_ivs)

    def __eq__(self, other: 'Match'):
        return len(self) == len(other) and all([other_icn in self.matched_ivs for other_icn in other])

    def __hash__(self):
        return hash(tuple(self.matched_ivs))

    def get_op_by_pattern_node_name(self, pattern_node_name: str) -> torch.nn.Module:
        for iv, pn_name in zip(self.matched_ivs, self.names):
            if pn_name == pattern_node_name:
                return iv.module

    @property
    def names(self) -> List[str]:
        return self._names

    @property
    @abc.abstractmethod
    def matched_ivs(self) -> List[InnerVertex]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def start(self) -> List[InnerVertex]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def end(self) -> List[InnerVertex]:
        raise NotImplementedError

    @property
    def can_be_rolled(self) -> bool:
        allowed_successors = [*self.end[0].successors, *self.matched_ivs]
        for iv in self:
            if any([successor not in allowed_successors for successor in iv.successors]):
                return False

        if len(self.start) >= 2:
            return False

        return True


class LinearMatch(Match):
    def __init__(self, ivs: List[InnerVertex] = None, names: List[str] = None):
        super().__init__(names=names)
        self._matched_ivs = [] if not ivs else ivs

    def __repr__(self):
        return f'{self.__class__.__name__}{self.matched_ivs}'

    @property
    def matched_ivs(self):
        return self._matched_ivs

    @property
    def start(self):
        return self.matched_ivs[-1:]

    @property
    def end(self):
        return self.matched_ivs[:1]

    def add(self, iv: Union[InnerVertex, None]):
        self._matched_ivs += [iv]

    def remove_missing_match_for_non_mandatory(self):
        self._names = [name for iv, name in zip(self._matched_ivs, self._names) if iv]
        self._matched_ivs = [iv for iv in self._matched_ivs if iv]


class SubgraphMatch(Match):
    def __init__(self, names: List[str] = None):
        super().__init__(names=names)
        self.match: LinearMatch = LinearMatch()
        self.inputs: List[SubgraphMatch] = None

    def __repr__(self):
        return f'{self.__class__.__name__}[match={self.match}, inputs={self.inputs}]'

    @property
    def matched_ivs(self) -> List[InnerVertex]:
        ivs = self.match.matched_ivs
        for submatch in self.all_submatches:
            for iv in submatch.match.matched_ivs:
                if iv not in ivs:
                    ivs += [iv]

        return ivs

    @property
    def start(self) -> List[InnerVertex]:
        start = []
        if self.inputs:
            for submatch in self.all_submatches:
                if not submatch.inputs:
                    for iv in submatch.match.start:
                        if iv not in start:
                            start += [iv]

        return start

    @property
    def end(self) -> List[InnerVertex]:
        return self.match.end

    @property
    def all_submatches(self):
        def traverse(current: SubgraphMatch, submatches: List[SubgraphMatch]):
            submatches += [current]
            if current.inputs:
                for inp in current.inputs:
                    traverse(inp, submatches)

            return submatches

        return traverse(self, [])

    def set_match(self, match: LinearMatch):
        for iv in match.matched_ivs:
            self.match.add(iv)

    def init_inputs(self, inputs):
        self.inputs = [SubgraphMatch(names=inp.pattern_node_names) for inp in inputs]
