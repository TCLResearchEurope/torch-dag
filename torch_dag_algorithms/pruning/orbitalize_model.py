#
# Copyright © TCL Research Europe. All rights reserved.
#
import argparse
import logging
import os

from torch_dag.core.dag_module import DagModule
from torch_dag.commons.flops_computation import log_dag_characteristics
from torch_dag_algorithms.pruning import dag_orbitalizer
from torch_dag_algorithms.pruning import constants
from torch_dag.visualization.visualize_dag import DagVisualizer


logger = logging.getLogger(__name__)


PRUNING_MODES = [
    constants.PRUNING_DEFAULT_MODE_NAME,
    constants.PRUNING_BLOCK_SNPE_MODE_NAME,
    constants.PRUNING_WHOLE_BLOCK_MODE_NAME,
]


def parse_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--model_path",
        type=str,
    )
    arg_parser.add_argument(
        "--saving_path",
        type=str,
    )
    arg_parser.add_argument(
        "--block_size",
        default=8,
        type=int,
    )
    arg_parser.add_argument(
        "--pruning_mode",
        type=str,
        default=constants.PRUNING_BLOCK_SNPE_MODE_NAME,
        choices=PRUNING_MODES,
    )
    arg_parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        help="Input shape to the orbitalized model (including batch dimension).",
    )
    args = arg_parser.parse_args()
    return args


def orbitalize_model(
    model_path,
    input_shape,
    pruning_mode,
    block_size,
    saving_path,
):
    path = model_path
    dag = DagModule.load(path)
    dag.eval()

    input_shape = tuple(input_shape)
    log_dag_characteristics(dag, input_shape_without_batch=input_shape[1:])

    orbitalizer = dag_orbitalizer.GeneralOrbitalizer(
        pruning_mode=pruning_mode,
        block_size=block_size,
    )

    dag, found_final_orbits = orbitalizer.orbitalize(
        dag=dag,
        prune_stem=True,
        input_shape=input_shape,
        skip_orbits_with_channels_less_than_block_size=True,
    )
    if not saving_path:
        saving_path = os.path.join(path, "orbitalized")
    else:
        saving_path = saving_path
    dag.save(os.path.join(saving_path))
    visualizer = DagVisualizer(dag=dag)
    visualizer.visualize(
        max_depth=0,
        input_shape=input_shape,
        saving_path=f"{saving_path}/vis",
    )
    logger.info(f"Saving orbitalized model to: {saving_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    orbitalize_model(
        args.model_path,
        args.input_shape,
        args.pruning_mode,
        args.block_size,
        args.saving_path,
    )


if __name__ == "__main__":  # TODO main
    main()
