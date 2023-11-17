from typing import Any, Type
from typing import Dict
from typing import Union
from typing import Tuple

import torch


class PatternNode:
    def __init__(
        self,
        op_type: Union[Type[torch.nn.Module], Tuple[Type[torch.nn.Module], ...]],
        extra_spec: Dict[str, Any] = None,
        transform: Dict[str, Any] = None,
        name: str = None,
        mandatory: bool = True
    ):
        self.op_type = op_type if type(op_type) is tuple else (op_type,)
        self.extra_spec = extra_spec
        self.transform = transform
        self.name = name
        self.mandatory = mandatory

    def __repr__(self):
        return f'{self.__class__.__name__}[op_type={[op_type.__name__ for op_type in self.op_type]}, extra_spec={self.extra_spec}, transform={self.transform}, name={self.name}, mandatory={self.mandatory}]'

    def spec_compliant(self, op: torch.nn.Module) -> bool:
        if self.extra_spec:
            for param, value in self.extra_spec.items():
                if hasattr(op, param):
                    param_compliant = getattr(op, param) == value if not callable(value) else value(getattr(op, param))
                    if not param_compliant:
                        return False

        return True
