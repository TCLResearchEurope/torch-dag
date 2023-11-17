from typing import Tuple, Type

import torch

from torch_dag.core.dag_module import DagModule


def load_dag_from_path(path: str, custom_module_classes: Tuple[Type[torch.nn.Module]] = (), ) -> DagModule:
    dag = DagModule.load(path, custom_module_classes=custom_module_classes)
    return dag
