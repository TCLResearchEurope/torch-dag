#
# Copyright © TCL Research Europe. All rights reserved.
#
import argparse
from functools import singledispatch
from typing import Dict, Union
from typing import List
from typing import Optional
from typing import Tuple

import graphviz
import torch
from torch import nn

from torch_dag import structured_modules as smodules
from torch_dag_algorithms.pruning.modules import MaskModule
from torch_dag.commons.flops_computation import build_full_flops_list
from torch_dag.core.dag_module import DagModule, InputVertex


DEFAULT_COLOR = "white"
INPUT_SHAPE = "invtriangle"
OUTPUT_SHAPE = "oval"
INPUT_COLOR = "green"
OUTPUT_COLOR = "orange"
CELL_SHAPE = "box"
CONV_COLOR = "red"
DEP_CONV_COLOR = "deepskyblue1"
ADD_COLOR = "yellow"
MUL_COLOR = "darkorchid1"
CONCAT_COLOR = "beige"
POOLING_COLOR = "darkgoldenrod"
CELL_FILL_COLOR = "lightblue"
BN_COLOR = "cornsilk"
ACT_COLOR = "aquamarine3"
DEPTH_1_FILL_COLOR = "gray90"
DEPTH_2_FILL_COLOR = "gray80"
NESTED_CELL_FILL_COLOR = "deepskyblue4"
NODE_SHAPE = "box"
NODE_SHAPE_COLOR = "black"
ATTENTION_COLOR = "darkolivegreen4"
LAYER_NORM_COLOR = "darkorange"
DENSE_COLOR = "darkseagreen3"
MATMUL_COLOR = "chocolate"
MASK_COLOR = "brown2"


@singledispatch
def get_vertex_color(module: torch.nn.Module):
    return DEFAULT_COLOR


@get_vertex_color.register(DagModule)
def _(module: DagModule):
    return NESTED_CELL_FILL_COLOR


@get_vertex_color.register(torch.nn.Conv2d)
def _(module: torch.nn.Conv2d):
    return CONV_COLOR if module.groups == 1 else DEP_CONV_COLOR


@get_vertex_color.register(smodules.AddModule)
def _(module: smodules.AddModule):
    return ADD_COLOR


@get_vertex_color.register(smodules.MulModule)
def _(module: smodules.MulModule):
    return MUL_COLOR


@get_vertex_color.register(smodules.ConcatModule)
def _(module: smodules.ConcatModule):
    return CONCAT_COLOR


@get_vertex_color.register(torch.nn.BatchNorm2d)
def _(module: torch.nn.BatchNorm2d):
    return BN_COLOR


# ... I don't see a better way
@get_vertex_color.register(nn.ReLU)
@get_vertex_color.register(nn.ReLU6)
@get_vertex_color.register(nn.SiLU)
@get_vertex_color.register(nn.Softmax)
@get_vertex_color.register(nn.Sigmoid)
@get_vertex_color.register(nn.Hardswish)
@get_vertex_color.register(nn.Hardsigmoid)
@get_vertex_color.register(nn.GELU)
@get_vertex_color.register(nn.LeakyReLU)
@get_vertex_color.register(nn.ELU)
@get_vertex_color.register(nn.Tanh)
@get_vertex_color.register(nn.Identity)
def _(module: smodules.ACTIVATION_MODULES_T):
    return ACT_COLOR


@get_vertex_color.register(torch.nn.LayerNorm)
def _(module: torch.nn.LayerNorm):
    return LAYER_NORM_COLOR


@get_vertex_color.register(torch.nn.Linear)
def _(module: torch.nn.Linear):
    return DENSE_COLOR


MATMUL_T = Union[smodules.MatmulModule, smodules.TfMatmulModule]


@get_vertex_color.register(smodules.TfMatmulModule)
@get_vertex_color.register(smodules.MatmulModule)
def _(module: MATMUL_T):
    return MATMUL_COLOR


POOLING_T = Union[torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d]


@get_vertex_color.register(torch.nn.MaxPool2d)
@get_vertex_color.register(torch.nn.AvgPool2d)
@get_vertex_color.register(torch.nn.AdaptiveAvgPool2d)
def _(module: POOLING_T):
    return POOLING_COLOR


@get_vertex_color.register(MaskModule)
def _(module: MaskModule):
    return MASK_COLOR


class DagVisualizer:

    def __init__(self, dag: DagModule):
        self.dag = dag
        self.dag.cache_forward_dict = True
        self.flops_list = None

    @staticmethod
    def get_name(namescope_index: str, index: str):
        if namescope_index is None:
            return f"{index}"
        else:
            return f"{namescope_index}_{index}"

    def visualize(
            self,
            max_depth: int = 0,
            input_shape: Tuple[int, ...] = None,
            saving_path: Optional[str] = None,
            saving_format: str = "pdf",
    ):
        if input_shape is not None:
            self.dag.eval()
            _ = self.dag(torch.ones(size=input_shape))
            if max_depth == 0:
                self.flops_list = build_full_flops_list(
                    dag=self.dag, input_shape_without_batch=input_shape[1:], normalize=True)

        graph, input_node_names, output_node_names = self._visualize(
            dag=self.dag,
            max_depth=max_depth,
        )
        if saving_path is not None:
            graph.render(saving_path, format=saving_format)

        return graph

    # def get_weights_stats(self, node: nd.nodes.Node):
    #     if isinstance(node, (nd.ops.Conv2D, nd.ops.DepthwiseConv)):
    #         return self.compute_tensor_stats(node.filters)
    #     elif isinstance(node, nd.ops.Dense):
    #         return self.compute_tensor_stats(node.kernel)
    #     else:
    #         return None

    def get_module_meta(self, module: nn.Module) -> Dict:
        meta = {}
        if isinstance(module, nn.Conv2d):
            meta['kernel_size'] = module.kernel_size
            meta['in_channels'] = module.in_channels
            meta['out_channels'] = module.out_channels
            meta['groups'] = module.groups
        elif isinstance(module, smodules.ACTIVATION_MODULES):
            meta['activation_function'] = module.__class__.__name__
        # weights_stats = self.get_weights_stats(node)
        # if weights_stats is not None:
        #     mean, std, maximum, minimum = weights_stats
        #     meta['weights_mean_std_max_min'] = f'{mean:.3f}, {std:.3f}, {maximum:.3f}, {minimum:.3f}'

        return meta

    def add_nested_dag_as_subgraph(
            self,
            g: graphviz.Digraph,
            name: str,
            dag: DagModule,
            depth: int,
            max_depth: int,
    ) -> Tuple[graphviz.Digraph, List[str], List[str]]:
        with g.subgraph(name=f'cluster_{dag.name}') as s:
            fillcolor = self.get_depth_fill_color(depth)
            s.attr(
                label=f'{dag.name}',
                style='filled',
                fillcolor=fillcolor,
            )
            return self._visualize(
                dag=dag,
                graph=s,
                namescope_index=name,
                max_depth=max_depth,
                depth=depth,
            )

    def get_depth_fill_color(self, depth: int):
        if depth == 1:
            return DEPTH_1_FILL_COLOR
        elif depth == 2:
            return DEPTH_2_FILL_COLOR

    def compute_tensor_stats(self, tensor: torch.Tensor):
        mean = tensor.mean()
        std = torch.std(tensor)
        maximum = tensor.max()
        minimum = tensor.min()
        return mean, std, maximum, minimum

    def _visualize(
            self,
            dag: DagModule,
            graph: graphviz.Digraph = None,
            namescope_index: str = None,
            max_depth: int = 0,
            depth: int = 0,
    ) -> Tuple[graphviz.Digraph, List[str], List[str]]:
        if graph is None:
            g = graphviz.Digraph('model')
        else:
            g = graph
        g.node_attr.update(style='filled', shape='box')
        go_deeper = True if max_depth > 0 else False
        names = {}
        input_vertex_names = []
        output_vertex_names = []
        for k, vertex in enumerate(dag.vertices):
            name = self.get_name(namescope_index, str(k))
            names[k] = name
            if isinstance(vertex, InputVertex):
                label = f'input_{k}'
                g.node(
                    name,
                    label=label,
                    color=NODE_SHAPE_COLOR,
                    fillcolor=INPUT_COLOR,
                    shape=INPUT_SHAPE,
                )
                input_vertex_names.append(name)

            else:
                predecessors_indices = [dag.vertices.index(pd) for pd in vertex.predecessors]
                if isinstance(vertex.module, DagModule) and go_deeper:
                    sgraph, inputs, outputs = self.add_nested_dag_as_subgraph(
                        g=g,
                        name=name,
                        dag=vertex.module,
                        depth=depth + 1,
                        max_depth=max_depth - 1
                    )

                    for l, pd in enumerate(predecessors_indices):
                        edge = names[pd], inputs[l]
                        g.edge(edge[0], edge[1])
                    names[k] = outputs[0]
                    if vertex == dag.output_vertex:
                        output_vertex_names = [name]

                else:
                    module = vertex.module
                    fillcolor = get_vertex_color(module)
                    color = get_vertex_color(module)

                    label = f'{vertex.name}'
                    label += f'_idx_{dag.vertices.index(vertex)}'
                    if max_depth == 0 and depth == 0 and self.flops_list is not None:
                        flops = self.flops_list[self.dag.inner_vertices.index(vertex)]
                        label += f' \n kmapp: {flops}'
                    if vertex.orbit is not None:
                        label += f' \n orbit: {vertex.orbit}'

                    if len(self.get_module_meta(module).keys()) > 0:
                        label += f' \n ----------'

                    # add meta node info
                    for key, value in self.get_module_meta(module).items():
                        label += f' \n {key}: {value}'

                    # add output shape visualization
                    if dag.forward_dict is not None:
                        if isinstance(vertex.module, smodules.ArgModule):
                            pass
                        else:
                            label += f' \n ----------'
                            tensors = dag.forward_dict[vertex]
                            if not isinstance(dag.forward_dict[vertex], List):
                                tensors = [tensors]
                            shapes = []
                            for el in tensors:
                                if isinstance(el, torch.Tensor):
                                    shapes.append(tuple([int(e) for e in el.shape]))
                                else:
                                    shapes.append(tuple())

                            for tensor, shape in zip(tensors, shapes):
                                label += f' \n {shape}'
                                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float:
                                    mean, std, maximum, minimum = self.compute_tensor_stats(tensor)
                                    label += f' \n mean: {mean:.3f}, std: {std:.3f}, max: {maximum:.3f}, min: {minimum:.3f}'

                    if vertex == dag.output_vertex:
                        shape = OUTPUT_SHAPE
                        fillcolor = f'{OUTPUT_COLOR}:{fillcolor}'
                        output_vertex_names = [name]
                    else:
                        shape = NODE_SHAPE

                    g.node(
                        name,
                        label=label,
                        shape=shape,
                        color=NODE_SHAPE_COLOR,
                        fillcolor=fillcolor,
                    )
                    for pd in predecessors_indices:
                        edge = names[pd], names[k]
                        g.edge(edge[0], edge[1])

        return g, input_vertex_names, output_vertex_names


def parse_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_path', required=True)
    arg_parser.add_argument('--saving_path', required=True)
    arg_parser.add_argument('--depth', type=int, default=0)
    arg_parser.add_argument(
        '--input_shape',
        type=int,
        nargs='+',
        help='Input shape (including batch dimension) for visualizing tensor shapes in dag',
    )
    args = arg_parser.parse_args()
    return args


def visualize_dag(
    model_path,
    input_shape,
    saving_path,
    depth,
):
    visualizer = DagVisualizer(dag=DagModule.load(model_path))
    visualizer.visualize(
        max_depth=depth,
        input_shape=None if input_shape is None else tuple(input_shape),
        saving_path=saving_path,
    )


def main():
    args = parse_args()
    visualize_dag(
        args.model_path,
        args.input_shape,
        args.saving_path,
        args.depth,
    )


if __name__ == "__main__":
    main()
