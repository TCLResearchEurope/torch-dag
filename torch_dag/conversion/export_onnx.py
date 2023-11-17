#
# Copyright © TCL Research Europe. All rights reserved.
#
import argparse
import os
import torch
import logging
from typing import Tuple

from torch_dag.core.dag_module import DagModule
from torch_dag.commons.flops_computation import log_dag_characteristics
import subprocess

logger = logging.getLogger(__name__)


def get_latest_opset():
    # Return max supported ONNX opset by this version of torch
    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k)


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
        help="Input shape to the orbitalized model (including batch dimension).",
    )
    args = arg_parser.parse_args()
    return args


def export_onnx(
    model_path: str,
    input_shape: Tuple[int, ...],
):
    path = model_path
    dag = DagModule.load(path)
    dag.eval()
    input_shape = tuple(input_shape)
    log_dag_characteristics(dag, input_shape_without_batch=input_shape[1:])

    onnx_path = f"{path}/model.onnx"
    x = torch.ones(size=input_shape)

    torch.onnx.export(
        dag,  # --dynamic only compatible with cpu
        x,
        onnx_path,
        verbose=False,
        opset_version=12,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        # input_names=['images'],
        # output_names=['output0', 'output1', 'output2'],
    )
    tflite_dir = os.path.join(path, "tflite")

    subprocess.run(["onnx2tf", "-i", onnx_path, "-o", tflite_dir])
    logger.info(f"ONNX path: {onnx_path}")


def main():
    args = parse_args()
    export_onnx(
        args.model_path,
        args.input_shape,
    )


if __name__ == "__main__":  # TODO main
    main()
