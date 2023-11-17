import logging
import warnings
from copy import deepcopy
from pprint import pprint
from typing import Dict
from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import nn

from dev_tools.constants import SUCCESS_RESULT, FAILURE_RESULT, CHANNEL_PRUNING_NAME
from torch_dag.commons.flops_computation import build_full_flops_list, compute_static_kmapp
from torch_dag.core.dag_module import DagModule
from torch_dag.core.dag_module_utils import compare_module_outputs
from torch_dag.core.unstructured_to_structured import build_from_unstructured_module
from torch_dag.visualization.visualize_dag import DagVisualizer
from torch_dag_algorithms.commons.flops_computation import compute_kmapp
from torch_dag_algorithms.pruning import constants
from torch_dag_algorithms.pruning import dag_orbitalizer
from torch_dag_algorithms.pruning import remove_channels
from torch_dag_algorithms.pruning.commons import get_orbits_dict

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

TESTED_ALGORITHMS = [
    CHANNEL_PRUNING_NAME,
]


def test_conversion_to_dag(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[Union[DagModule, None], str]:
    model_copy = deepcopy(model)
    model.eval()

    try:
        dag = build_from_unstructured_module(model=model)
        compare_module_outputs(
            first_module=model_copy,
            second_module=dag,
            input_shape=input_shape,
            atol=1e-5,
        )
        msg = ''
    except Exception as e:
        logger.info(f'Failed to dagify. Error message: {e}')
        msg = str(e)
        dag = None

    return dag, msg


def orbitalize_model(
        dag: DagModule,
        pruning_mode: str,
        input_shape: Tuple[int, ...],
) -> Tuple[DagModule, float]:
    block_size = np.random.randint(1, 17) if pruning_mode == constants.PRUNING_BLOCK_SNPE_MODE_NAME else None
    orbitalizer = dag_orbitalizer.GeneralOrbitalizer(
        pruning_mode=pruning_mode,
        block_size=block_size,
    )
    dag_orb, orbits, prunable_kmapp, total_kmapp = orbitalizer.orbitalize(
        dag=dag,
        vis_final_orbits=False,
        prune_stem=True,
        input_shape=input_shape,
        return_stats=True,
    )
    prunable_fraction = prunable_kmapp / total_kmapp
    return dag_orb, prunable_fraction


def test_orbitalization_and_channel_removal(
        dag: DagModule,
        input_shape: Tuple[int, ...],
        pruning_mode: str = constants.PRUNING_BLOCK_SNPE_MODE_NAME,
        prob_removal: float = 0.0,
        saving_path: Optional[str] = None,
        kmapp_computation_rtol=0.3,
) -> Dict:
    dag.eval()
    prunable_fraction = 0.0
    msg = None
    try:
        dag_orbitalized, prunable_fraction = orbitalize_model(
            dag=dag,
            pruning_mode=pruning_mode,
            input_shape=input_shape,
        )
        dag_orbitalized.cache_forward_dict = True
        if saving_path:
            dag_orbitalized.save(saving_path)
            vis = DagVisualizer(dag_orbitalized)
            vis.visualize(max_depth=0, input_shape=input_shape, saving_path=f'{saving_path}/vis')
    except Exception as e:
        logger.warning(f'Orbitalization failed! ')
        pprint(e)
        return {
            'status':            FAILURE_RESULT,
            'reason':            str(e),
            'prunable_fraction': prunable_fraction,
        }

    # try channel removal
    try:
        orbits_dict = get_orbits_dict(dag_orbitalized)
        for k, v in orbits_dict.items():
            num_channels = v.num_channels
            v.debug_logits = torch.normal(mean=torch.zeros(size=(num_channels,)))
            p = np.random.uniform()
            if p < prob_removal:
                v.debug_logits = - torch.ones(size=(num_channels,))

        flops_list = build_full_flops_list(dag_orbitalized, input_shape_without_batch=input_shape[1:])
        pre_kmapp = compute_kmapp(dag_orbitalized, input_shape_without_batch=input_shape[1:],
                                  full_flops_list=flops_list)

        dag_pruned = remove_channels.remove_channels_in_dag(
            dag=dag_orbitalized,
            input_shape=input_shape
        )
        if dag_pruned.output_vertex:
            post_kmapp = compute_static_kmapp(dag_pruned, input_shape_without_batch=input_shape[1:])

            diff = (post_kmapp - pre_kmapp) / post_kmapp
            abs_diff = abs(diff)
            warnings.warn(
                f'Pre kmapp: {pre_kmapp}, post kmapp: {post_kmapp}, rdiff: {abs_diff}')
            success = True
            # if `prob_removal` is zero we want to `post_kmapp` to be close to `pre_kmapp`
            if prob_removal == 0.0 and pruning_mode != constants.PRUNING_WHOLE_BLOCK_MODE_NAME:
                if abs_diff > kmapp_computation_rtol:
                    logger.warning(
                        f'Pre kmapp: {pre_kmapp}, post kmapp: {post_kmapp}, rdiff: {abs_diff} is bigger than rtol: '
                        f'{kmapp_computation_rtol}.')
                    success = False

            # if `prob_removal` is bigger than zero or `pruning model == constants.PRUNING_WHOLE_BLOCK_MODE_NAME we
            # want `post_kmapp` to be lower than `pre_kmapp` plus some relative relaxation threshold
            else:
                if diff > kmapp_computation_rtol:
                    logger.warning(
                        f'Pre kmapp: {pre_kmapp}, post kmapp: {post_kmapp}, rdiff: {abs_diff} is bigger than rtol: '
                        f'{kmapp_computation_rtol}.')
                    success = False

            if not success:
                return {
                    'status':            FAILURE_RESULT,
                    'reason':            f'kmapp computation relative diff: {diff}',
                    'prunable_fraction': prunable_fraction,
                }

        else:
            post_kmapp = 0.0
        logger.info(f'kmapp with orbits randomized {pre_kmapp}, kmapp after channel removal {post_kmapp}')
    except Exception as e:
        logger.warning(f'Channel removal failed! ')
        pprint(e)
        return {
            'status':            FAILURE_RESULT,
            'reason':            str(e),
            'prunable_fraction': prunable_fraction,
        }

    return {
        'status':            SUCCESS_RESULT,
        'reason':            msg,
        'prunable_fraction': prunable_fraction,
    }
