import os

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
    import tensorflow as tf
    tf.get_logger().setLevel("INFO")
except ModuleNotFoundError:
    pass


import click

from torch_dag.conversion.export_onnx import export_onnx as export_onnx_api

from torch_dag.commons.compute_stats import compute_stats as compute_stats_api
from torch_dag_algorithms.commons.transform_model import transform_model as transform_model_api
from torch_dag_algorithms.commons.transform_model import TRANSFORM_TYPES

from torch_dag_algorithms.pruning.prune_channels import (
    prune_channels as prune_channels_api,
)
from torch_dag_algorithms.pruning.orbitalize_model import (
    orbitalize_model as orbitalize_model_api,
)
from torch_dag_algorithms.pruning.orbitalize_model import PRUNING_MODES

from torch_dag.visualization.visualize_dag import visualize_dag as visualize_dag_api


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_path", type=str)
@click.option("-i", "--input_shape", type=int, nargs=4, required=True)
def export_onnx(model_path, input_shape):
    export_onnx_api(model_path=model_path, input_shape=input_shape)


@cli.command()
@click.argument("model_path", type=str)
@click.option("-i", "--input_shape", type=int, nargs=4, required=True)
@click.option("-o", "--saving_path", type=str, required=True)
@click.option("-d", "--depth", type=int, default=0)
def visualize_dag(model_path, input_shape, saving_path, depth=0):
    visualize_dag_api(model_path, input_shape, saving_path, depth)


@cli.command()
@click.argument("model_path", type=str)
@click.option("-i", "--input_shape", type=int, nargs=4, required=True)
@click.option("-o", "--saving_path", type=str, required=True)
@click.option("-t", "--type", type=click.Choice(TRANSFORM_TYPES), required=True)
def transform_model(
    model_path,
    input_shape,
    saving_path,
    type,
):
    transform_model_api(
        model_path,
        input_shape,
        saving_path,
        type,
    )


@cli.command()
@click.argument("model_path", type=str)
@click.option("-i", "--input_shape", type=int, nargs=4, required=True)
def compute_stats(model_path, input_shape):
    compute_stats_api(model_path, input_shape)


@cli.command()
@click.argument("model_path", type=str)
@click.option("-i", "--input_shape", type=int, nargs=4, required=True)
@click.option("-o", "--saving_path", type=str, required=True)
def prune_channels(model_path, input_shape, saving_path):
    prune_channels_api(model_path, input_shape, saving_path)


@cli.command()
@click.argument("model_path", type=str)
@click.option("-i", "--input_shape", type=int, nargs=4, required=True)
@click.option("-o", "--saving_path", type=str, required=True)
@click.option("-m", "--pruning_mode", type=click.Choice(PRUNING_MODES), required=True)
@click.option("-b", "--block_size", type=int, default=8)
def orbitalize_model(model_path, input_shape, saving_path, pruning_mode, block_size):
    orbitalize_model_api(model_path, input_shape, pruning_mode, block_size, saving_path)


if __name__ == "__main__":
    cli()
