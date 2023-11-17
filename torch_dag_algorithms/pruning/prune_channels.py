#
# Copyright © TCL Research Europe. All rights reserved.
#
import argparse
import logging
import os

from torch_dag.core.dag_module import DagModule
from torch_dag.commons.flops_computation import log_dag_characteristics
from torch_dag_algorithms.pruning.remove_channels import remove_channels_in_dag
from torch_dag.visualization.visualize_dag import DagVisualizer


logger = logging.getLogger(__name__)


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
        "--input_shape",
        type=int,
        nargs="+",
        help="Input shape to the orbitalized model (including batch dimension).",
    )
    args = arg_parser.parse_args()
    return args


def prune_channels(
    model_path,
    input_shape,
    saving_path,
):
    path = model_path
    if saving_path is None:
        saving_path = os.path.join(model_path, "pruned")
    else:
        saving_path = saving_path
    dag = DagModule.load(path)
    dag = dag.flatten()
    dag.eval()
    input_shape = tuple(input_shape)
    log_dag_characteristics(dag, input_shape_without_batch=input_shape[1:])
    dag = remove_channels_in_dag(dag=dag, input_shape=input_shape)
    log_dag_characteristics(dag, input_shape_without_batch=input_shape[1:])
    dag.save(os.path.join(saving_path))
    logger.info(f"Saving pruned model to: {saving_path}")
    visualizer = DagVisualizer(dag=dag)
    visualizer.visualize(
        max_depth=0,
        input_shape=input_shape,
        saving_path=f"{saving_path}/vis",
    )


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    prune_channels(
        args.model_path,
        args.input_shape,
        args.saving_path,
    )


if __name__ == "__main__":  # TODO main
    main()
