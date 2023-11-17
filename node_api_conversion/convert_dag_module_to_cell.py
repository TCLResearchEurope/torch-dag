#
# Copyright © TCL Research Europe. All rights reserved.
#

import argparse
import logging
import os

import torch

import node_api as nd
from torch_dag.core.dag_module import DagModule
from node_api_conversion.convert_cell_to_dag_module import clean_up
from node_api_conversion.to_nd_converter import convert_dag_module_to_cell

logger = logging.getLogger(__name__)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path')
    arg_parser.add_argument('--saving_path')
    arg_parser.add_argument(
        '--input_shape',
        type=int,
        nargs='+',
        default=(1, 3, 320, 320),
    )
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    dag = DagModule.load(args.model_path)
    if args.saving_path:
        saving_path = args.saving_path
    else:
        saving_path = os.path.join(args.model_path, 'cell')
    input_shape = tuple(args.input_shape)

    logger.info(f'Unique modules:')
    unique_modules = set([type(m) for m in dag.modules()])
    for m in unique_modules:
        print(m.__name__)

    with torch.no_grad():
        cell = convert_dag_module_to_cell(dag, input_shape_without_batch=input_shape[1:])[0]

    clean_up(cell)
    nd.io.save_cell(cell=cell, path=saving_path)
    logger.info(f'Saved to: {saving_path}')


if __name__ == '__main__':
    main()
