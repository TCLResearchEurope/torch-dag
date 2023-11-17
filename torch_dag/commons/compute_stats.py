#
# Copyright © TCL Research Europe. All rights reserved.
#
import argparse
import logging

from torch_dag.core.dag_module import DagModule
from torch_dag.commons.flops_computation import log_dag_characteristics


def parse_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--model_path",
        type=str,
    )
    arg_parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=(1, 3, 224, 224),
    )
    args = arg_parser.parse_args()
    return args


def compute_stats(model_path, input_shape):
    input_shape = tuple(input_shape)
    dag = DagModule.load(model_path)
    log_dag_characteristics(dag, input_shape_without_batch=input_shape[1:])


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    compute_stats(
        args.model_path,
        args.input_shape,
    )


if __name__ == "__main__":  # TODO main
    main()
