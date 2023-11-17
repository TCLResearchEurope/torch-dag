from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple

from torch_dag.core.dag_module import InnerVertex
from torch_dag_algorithms.pruning.orbit import Orbit


class MaskingInsertionStrategy(ABC):
    """
    Interface to find a spot within orbit where masking should be inserted. 
    """

    @abstractmethod
    def find_reference_nodes(self, orbit: Orbit) -> List[Tuple[InnerVertex, InnerVertex]]:
        """
        Function that return list of tuples (start_icn, end_icn) where masking should be inserted. End_icn can be either a proper sink like Conv2D/Dense or "dummy" sink like Concat. Masking has to be inserted in each unique end within orbit.  
        """
        raise NotImplementedError


class AtTheEndOfPathStrategy(MaskingInsertionStrategy):
    """
    This strategy simply return list of tuples of last non-sink nodes in orbit and proper/dummy sink node. Masking will be inserted between them.
    """

    def find_reference_nodes(self, orbit: Orbit) -> List[Tuple[InnerVertex, InnerVertex]]:
        return orbit.end_path
