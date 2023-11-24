#
# Copyright © TCL Research Europe. All rights reserved.
#
import itertools
import warnings
from copy import deepcopy
from typing import Tuple

import numpy as np
import pytest
import timm
import torch

from torch_dag.commons.flops_computation import compute_static_kmapp, build_full_flops_list
from torch_dag.core.dag_module import DagModule
from torch_dag.core.unstructured_to_structured import build_from_unstructured_module
from torch_dag_algorithms.commons.flops_computation import compute_kmapp
from torch_dag_algorithms.pruning import constants
from torch_dag_algorithms.pruning import dag_orbitalizer
from torch_dag_algorithms.pruning import remove_channels
from torch_dag_algorithms.pruning.commons import get_orbits_dict
from .test_constants import test_cases

MODEL_NAMES = [case.timm_name for case in test_cases]


def orbitalize_model(
        dag: DagModule,
        pruning_mode: str,
        input_shape: Tuple[int, ...],
) -> DagModule:
    block_size = np.random.randint(1, 17) if pruning_mode == constants.PRUNING_BLOCK_SNPE_MODE_NAME else None
    orbitalizer = dag_orbitalizer.GeneralOrbitalizer(
        pruning_mode=pruning_mode,
        block_size=block_size,
    )
    dag_orb, orbits = orbitalizer.orbitalize(
        dag=dag,
        vis_final_orbits=False,
        prune_stem=True,
        input_shape=input_shape,
        force_log_stats=False,
    )
    return dag_orb


probs_removal = [0.1]
pruning_modes = [
    constants.PRUNING_BLOCK_SNPE_MODE_NAME,
    constants.PRUNING_DEFAULT_MODE_NAME,
    constants.PRUNING_WHOLE_BLOCK_MODE_NAME,
]

list_of_params = [probs_removal, pruning_modes, MODEL_NAMES]

test_data = []

for tc in test_cases:
    for pm in pruning_modes:
        if tc.prob_removal == 0.0:
            test_data.append((0.0, pm, tc.timm_name))
        else:
            for pr in probs_removal:
                test_data.append((pr, pm, tc.timm_name))


@pytest.mark.ioheavy
@pytest.mark.parametrize(
    [
        "prob_removal",
        "pruning_mode",
        "model"
    ],
    test_data,
)
def test_orbitalization_channel_removal_and_dynamic_kmapp(
        prob_removal: float,
        pruning_mode: str,
        model: str,
        kmapp_computation_rtol=0.2
):
    test_case = [case for case in test_cases if case.timm_name == model][0]
    if test_case.prob_removal == 0.0:
        prob_removal = 0.0  # TODO rewrite
    timm_model = timm.create_model(model, pretrained=False)
    timm_model.eval()
    input_shape = timm_model.default_cfg['input_size']
    input_shape = (1,) + input_shape
    dag = build_from_unstructured_module(model=timm_model)
    dag.eval()
    dag_copy = deepcopy(dag)
    dag_copy.eval()
    dag_orbitalized = orbitalize_model(
        dag=dag_copy,
        pruning_mode=pruning_mode,
        input_shape=input_shape,
    )
    dag_orbitalized.cache_forward_dict = True
    # sample logits
    orbits_dict = get_orbits_dict(dag_orbitalized)
    for k, v in orbits_dict.items():
        num_channels = v.num_channels
        v.debug_logits = torch.normal(mean=torch.zeros(size=(num_channels,)))
        p = np.random.uniform()
        if p < prob_removal:
            v.debug_logits = - torch.ones(size=(num_channels,))

    flops_list = build_full_flops_list(dag_orbitalized, input_shape_without_batch=input_shape[1:])
    pre_kmapp = compute_kmapp(dag_orbitalized, input_shape_without_batch=input_shape[1:], full_flops_list=flops_list)

    dag_pruned = remove_channels.remove_channels_in_dag(
        dag=dag_orbitalized,
        input_shape=input_shape
    )
    if dag_pruned.output_vertex:
        post_kmapp = compute_static_kmapp(dag_pruned, input_shape_without_batch=input_shape[1:])
        diff = (post_kmapp - pre_kmapp) / post_kmapp
        abs_diff = abs(diff)
        warnings.warn(
            f'Model: {model}, pre kmapp: {pre_kmapp}, post kmapp: {post_kmapp}, rdiff: {abs_diff}')

        # if `prob_removal` is zero we want to `post_kmapp` to be close to `pre_kmapp`
        if prob_removal == 0.0 and pruning_mode != constants.PRUNING_WHOLE_BLOCK_MODE_NAME:
            if abs_diff > kmapp_computation_rtol:
                raise AssertionError(
                    f'Pre kmapp: {pre_kmapp}, post kmapp: {post_kmapp}, rdiff: {abs_diff} is bigger than rtol: '
                    f'{kmapp_computation_rtol}.')

        # if `prob_removal` is bigger than zero or `pruning model == constants.PRUNING_WHOLE_BLOCK_MODE_NAME we
        # want `post_kmapp` to be lower than `pre_kmapp` plus some relative relaxation threshold
        else:
            if diff > kmapp_computation_rtol:
                raise AssertionError(
                    f'Pre kmapp: {pre_kmapp}, post kmapp: {post_kmapp}, rdiff: {abs_diff} is bigger than rtol: '
                    f'{kmapp_computation_rtol}.')

    else:
        post_kmapp = 0.0
