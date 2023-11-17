#
# Copyright © TCL Research Europe. All rights reserved.
#
import warnings

import pytest
from collections import Counter

import node_api as nd
from node_api_conversion import from_nd_converter
import modelhub_client as mh

models_to_test = [
    # 'YOLOX_n',  # TODO: fix mismatch here
    'FBNetV3B',
    'resnet_50_torch_r224',
    'mbnet_v3_small_075',
    'enet_v2_M_m60',
    'mbnet_v2_segmentation',
    'eformer_l0',
]

models_and_input_shapes = {}
models_and_output_ranks = {}
for model_name in models_to_test:
    model = mh.api.Model.get(model_name)
    input_shape = tuple(model.card['inputs']['0']['test_shape'][0])
    models_and_input_shapes[str(model_name)] = input_shape

model_names = list(models_and_input_shapes.keys())


@pytest.fixture(scope='module')
def model_cells():
    result = {}
    for model_name in models_and_input_shapes.keys():
        mh_model = mh.api.Model.get(model_name)
        cell, _ = mh_model.load_cell()
        result[model_name] = cell
    yield result

def enforce_names_uniqueness_in_cell(cell: nd.cells.Cell):
    names = [v.node.name for v in cell.inner_cell_nodes]
    while len(names) != len(set(names)):
        names_counter = Counter()
        for v in cell.inner_cell_nodes:
            name = v.node.name
            names_counter[name] += 1
            if names_counter[name] > 1:
                new_name = f'{name}_{names_counter[name] - 1}'
                v.node.name = new_name
        names = [v.node.name for v in cell.inner_cell_nodes]


@pytest.mark.skip
@pytest.mark.parametrize("model_name", model_names)
def test_conversion_to_torch(
        model_name: str,
):
    model = mh.api.Model.get(model_name)
    cell, _ = model.load_cell()
    cell = cell.flatten()
    enforce_names_uniqueness_in_cell(cell)
    if isinstance(cell.inner_cell_nodes[0].node, nd.ops.NormalizationNode):
        cell.remove_node(cell.inner_cell_nodes[0])
    cell = nd.cells_utils.extract_subcell_for_a_given_output(
        original_cell=cell,
        inner_cell_node_index=-1,
    )
    torch_cell, nsr = from_nd_converter.convert_cell_to_torch_dag_module(
        cell,
        input_shape_without_batch=models_and_input_shapes[model_name][1:],
        batch_size_for_verification=4,
    )
    warnings.warn(f'Conversion NSR: {nsr}')
    assert nsr <= 0.0001
    # TODO: Try testing conversion to ONNX in the future
    # input_shape = models_and_input_shapes[model_name]
    # x = torch.ones(size=(1, input_shape[3], input_shape[1], input_shape[2]))
    # torch.onnx.export(
    #     torch_cell,
    #     x,
    #     f'./{model_name}.onnx',
    #     export_params=True,
    # )
    # subprocess.run(["onnx2tf", "-i", f'./{model_name}.onnx'])
