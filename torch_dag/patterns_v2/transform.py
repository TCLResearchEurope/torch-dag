import sys
import enum
import copy
import logging
import numpy as np
import torch
from typing import Any, Type, Union
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable


from torch_dag.core.dag_module import DagModule
from torch_dag.core.dag_module import InnerVertex, InputVertex
from torch_dag.patterns_v2.pattern import Pattern
from torch_dag.patterns_v2.match import Match

logger = logging.getLogger(__name__)


class SelectionType(enum.Enum):
    FIRST = 'first'
    LAST = 'last'
    ALL = 'all'
    ALL_BUT_FIRST = 'all_but_first'
    ALL_BUT_LAST = 'all_but_last'


class Transform:
    """
    For more information on patterns please refer to https://gitlab.com/tcl-research/auto-ml/blog/-/blob/main/2022-06-16-patterns/README.md or https://gitlab.com/tcl-research/auto-ml/node-api/-/blob/master/tutorials/advanced/pattens.ipynb
    """
    PRESERVE_PARAMS_FROM = 'preserve_old_params_from'

    @staticmethod
    def _assert_forward_pass_equal(
        cell1: DagModule,
        cell2: DagModule,
        validate_same_output: bool = False,
        input_size: Union[Tuple[int, ...], List[Tuple[int, ...]]] = (1, 3, 288, 512),
    ):
        """Asserts whether forward pass on cell1 and cell2 works. If `validate_same_output` is set to True then outputs are matched with atol=1e-9. If `input_size` specified then forward pass is run with given shape.

        Args:
            cell1 (Cell): First cell to validate
            cell2 (Cell): Second cell to validate
            validate_same_output (bool, optional): If True then output tensors will be compared with atol=1e-9. Defaults to False.
            input_size (Union[Tuple[int, ...], List[Tuple[int, ...]]], optional): Input shape that will be used to generate input tensor. Defaults to [1, 512, 512, 3].
        """
        input_ = [torch.ones(input_size)]
        cell1.predict()
        output1 = cell1(input_).output_tensors[0]

        cell2.predict()
        output2 = cell2(input_).output_tensors[0]

        if validate_same_output:
            assert np.allclose(output1, output2, atol=1e-9), 'Outputs after surgery dont match original outputs'

    def find(
        self,
        flatten_dag: DagModule,
        pattern: Pattern,
        selection: Union[SelectionType, Callable[[List[Match]], List[Match]]] = SelectionType.ALL
    ) -> List[Match]:
        logger.info(f'Searching for pattern {pattern} in {flatten_dag.name}...')
        matches = pattern.search(inner_vertexes=flatten_dag.inner_vertices)
        logger.info(f'Found [{len(matches)}] matches.')

        if callable(selection):
            logger.info(f'Custom selection passed.')
            # matches = selection(matches)
            matches = [match for match in matches if selection(match)]
            logger.info(f'{len(matches)} matches remain')
        elif selection == SelectionType.FIRST:
            logger.info(f'Selection type - only first')
            matches = matches[:1]
        elif selection == SelectionType.LAST:
            logger.info(f'Selection type - only last')
            matches = matches[-1:]
        elif selection == SelectionType.ALL:
            logger.info(f'Selection type - all')
            pass
        elif selection == SelectionType.ALL_BUT_FIRST:
            logger.info(f'Selection type - all but first')
            matches = matches[1:]
        elif selection == SelectionType.ALL_BUT_LAST:
            logger.info(f'Selection type - all but last')
            matches = matches[:-1]

        return matches