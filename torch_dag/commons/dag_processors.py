import logging

import torch.nn

from torch_dag.core.dag_module import InnerVertex, DagModule
from torch_dag.core.unstructured_to_structured import unroll_prunable_modules

logger = logging.getLogger(__name__)


def atomic_module_printer(iv: InnerVertex):
    if not isinstance(iv.module, DagModule):
        logger.info(f'Inner vertex: {iv.name}, module: {iv.module.__class__.__name__}')


def set_bn_to_eval(iv: InnerVertex):
    if isinstance(iv.module, torch.nn.BatchNorm2d):
        iv.module.eval()
