import logging
from typing import Tuple

import numpy as np
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = False
session = tf.compat.v1.InteractiveSession(config=config)
import torch

import node_api as nd
from node_api.flops_computation import get_node_flops

logger = logging.getLogger(__name__)


def tf_per_channel_noise_to_signal_ratio(
        x: tf.Tensor,
        y: tf.Tensor,
        epsilon: float = 1e-6,
) -> tf.Tensor:
    """
    In computing NSR we always assume that the channels correspond to the last dimension.
    :return:
    """
    axis = list(range(x.shape.rank - 1))
    y_per_channel_variance = tf.square(tf.math.reduce_std(
        y, axis=axis,
    ))
    per_channel_squared_difference = tf.reduce_mean(
        tf.square(
            x - y,
        ),
        axis=axis,
    )
    return tf.reduce_mean(
        tf.divide(
            per_channel_squared_difference,
            y_per_channel_variance + epsilon,
        )
    )


def per_channel_noise_to_signal_ratio(
        x: torch.Tensor,
        y: torch.Tensor,
        non_channel_dim: Tuple[int, ...] = (0, 2, 3),
        epsilon: float = 1e-3,

) -> torch.Tensor:
    y_per_channel_variance = torch.square(torch.std(y, dim=non_channel_dim))
    per_channel_squared_difference = torch.square((x - y)).mean(dim=non_channel_dim)

    return torch.divide(per_channel_squared_difference, y_per_channel_variance + epsilon).mean()


def get_num_params(cell: nd.cells.Cell):
    result = 0
    for var in cell.vars:
        result += np.product(var.shape)
    return result


def compute_static_kmapp(
        cell: nd.cells.Cell,
        input_shape_without_batch: Tuple[int, ...],
        normalized: bool = True
) -> nd.backend.TENSOR_TYPE:
    input_instance = nd.nodes.InputTensorsNodeInstance(input_tensors=[tf.ones(shape=(1,) + input_shape_without_batch)])
    multiadds = get_node_flops(node=cell, input_instances=[input_instance])
    normalization = tf.cast(input_shape_without_batch[0] * input_shape_without_batch[1] * 1e3, tf.float32)
    return multiadds / normalization if normalized else multiadds


def log_cell_characteristics(
        cell: nd.cells.Cell,
        input_shape_without_batch: Tuple[int, ...],
):
    if len(input_shape_without_batch) < 2:
        logger.warning(f'One cannot compute `kmapp` for cell: {cell.name}, since the input_shape_without_batch '
                       f'has length less than 2.')
        return
    cell.predict()
    static_kmapp = compute_static_kmapp(cell, input_shape_without_batch)
    static_multiadds = compute_static_kmapp(cell, input_shape_without_batch, normalized=False)
    x = tf.ones(shape=(1,) + input_shape_without_batch)
    result = cell(x)
    logger.info(f'static_kmapp: {static_kmapp}')
    logger.info(f'static_multiadds (M): {static_multiadds / 1e6}')
    num_params = get_num_params(cell) / 1e6
    logger.info(f'number params (M): {num_params}')
    logger.info(f'number of output tensors: {len(result.output_tensors)}')
    for k, tensor in enumerate(result.output_tensors):
        logger.info(f'output shape of output tensor {k}: {tensor.shape}')
