import logging

import torch

import torch_dag as td

logger = logging.getLogger(__name__)


def look_for_dagable_modules(model: torch.nn.Module):
    """
    A function that recursively goes through all modules in `model` (including the `model` itself) and checks 
    whether these modules can be converted to `DagModule`.
    :param model: any torch.nn.Module instance
    :return: dagable_classes, undagable_classes a tuple of sets of classes that can and can't be converted to
    `DagModule`
    """
    dagable_submodules = []
    other_submodules = []

    for k, nested_module in enumerate(list(model.modules())):
        num_submodules = len(list(nested_module.modules()))
        if num_submodules > 1:
            logger.info(nested_module.__class__)
            try:
                nested_dag = td.build_from_unstructured_module(nested_module)
                if len(nested_dag.inner_modules) > 1:
                    logger.info(f'SUCCESS: {nested_module.__class__}')
                    dagable_submodules.append(nested_module)
            except Exception as e:
                other_submodules.append(nested_module)
                logger.info(f'FAILURE: {nested_module.__class__}')
                logger.info(e)

    dagable_classes = set([m.__class__ for m in dagable_submodules])
    undagable_classes = set([m.__class__ for m in other_submodules])
    logger.info(f'Dagable classes:')
    for el in dagable_classes:
        logger.info(el)

    logger.info(f'Undagable classes:')
    for el in undagable_classes:
        logger.info(el)

    return dagable_classes, undagable_classes
