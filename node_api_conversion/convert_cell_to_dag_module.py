#
# Copyright © TCL Research Europe. All rights reserved.
#

import argparse
import logging

import node_api as nd
from node_api_conversion import from_nd_converter
from torch_dag.visualization.visualize_dag import DagVisualizer
from node_api_conversion.utils import log_cell_characteristics
import modelhub_client as mh

logger = logging.getLogger(__name__)


def find_icns_to_remove(cell: nd.cells.Cell):
    result = []
    for icn in cell.inner_cell_nodes:
        if isinstance(icn.node, nd.ops.Activation):
            if icn.node.activation_name in (None, 'none', 'identity'):
                result.append(icn)
        if isinstance(icn.node, nd.ops.Sum) and len(icn.predecessors) == 1:
            result.append(icn)
        if isinstance(icn.node, nd.ops.Concat) and len(icn.predecessors) == 1:
            result.append(icn)
        if isinstance(icn.node, nd.ops.Mul) and len(icn.predecessors) == 1:
            result.append(icn)
        if isinstance(icn.node, nd.cells.Cell):
            result.extend(find_icns_to_remove(icn.node))
    return result


def clean_up(cell: nd.cells.Cell):
    to_remove = find_icns_to_remove(cell)
    for icn in to_remove:
        icn.cell.remove_node(icn)
        logger.info(f'Removing {icn} with class {icn.node.__class__.__name__}')


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_path')
    arg_parser.add_argument('--hub_name')
    arg_parser.add_argument('--saving_path')
    arg_parser.add_argument(
        '--input_shape',
        type=int,
        nargs='+',
        default=(1, 320, 320, 3),
    )
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.hub_name:
        model = mh.api.Model.get(args.hub_name)
        cell, _ = model.load_cell()
    else:
        cell, _ = nd.io.load_cell(args.model_path)

    input_shape = tuple(args.input_shape)
    cell = cell.flatten()
    cell.predict()
    clean_up(cell)
    nd.cells_utils.fuse_padding_nodes(cell, input_size=input_shape)
    log_cell_characteristics(cell, input_shape[1:])

    cell = nd.cells_utils.extract_subcell_for_a_given_output(
        original_cell=cell,
        inner_cell_node_index=-1

    )

    dag = from_nd_converter.convert_cell_to_torch_dag_module(
        cell=cell,
        input_shape_without_batch=input_shape[1:],
    )[0]
    dag.save(args.saving_path)
    logger.info(f'dag module saved to: {args.saving_path}')

    visualizer = DagVisualizer(dag=dag)
    visualizer.visualize(
        max_depth=0,
        input_shape=(8, input_shape[3], input_shape[1], input_shape[2]),
        saving_path=f'{args.saving_path}/vis',
    )


if __name__ == '__main__':
    main()
