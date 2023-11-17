from typing import Tuple, Optional, Type

import torch

from torch_dag.commons.flops_computation import compute_static_kmapp, build_full_flops_list
from torch_dag.core.dag_module import DagModule
from torch_dag_algorithms.commons.flops_computation import compute_kmapp
from torch_dag_algorithms.pruning.constants import PRUNING_DEFAULT_MODE_NAME
from torch_dag_algorithms.pruning.dag_orbitalizer import GeneralOrbitalizer
from torch_dag_algorithms.pruning.losses.bkd_losses import bkd_loss
from torch_dag_algorithms.pruning.losses.entropy_losses import entropy_loss
from torch_dag_algorithms.pruning.losses.latency_losses import latency_loss
from torch_dag_algorithms.pruning.remove_channels import remove_channels_in_dag

CPU_DEVICE = torch.device('cpu')


class ChannelPruning:

    def __init__(
            self,
            model: DagModule,
            input_shape_without_batch: Tuple[int, ...],
            pruning_proportion: float,
            num_training_steps: int,
            prune_stem: bool = True,
            block_size: Optional[int] = 8,
            pruning_mode: str = PRUNING_DEFAULT_MODE_NAME,
            entropy_loss_weight: float = 1.0,
            flops_loss_weight: float = 10.0,
            anneal_losses: bool = True,
            custom_unprunable_module_classes: Tuple[Type[torch.nn.Module]] = (),

    ):
        self.model = model
        self.model_orbitalized = None
        self.input_shape_without_batch = input_shape_without_batch
        self.pruning_proportion = pruning_proportion
        self.prune_stem = prune_stem
        self.block_size = block_size
        self.pruning_mode = pruning_mode
        self.full_flops_list = None
        self.entropy_loss_weight = entropy_loss_weight
        self.flops_loss_weight = flops_loss_weight
        self.num_training_steps = num_training_steps
        self.anneal_losses = anneal_losses
        self.target_normalized_flops = None
        self.custom_unprunable_module_classes = custom_unprunable_module_classes
        self.initial_normalized_flops = compute_static_kmapp(
            self.model, input_shape_without_batch=self.input_shape_without_batch)
        self.prunable_proportion = None

    def prepare_for_pruning(self) -> DagModule:
        orbitalizer = GeneralOrbitalizer(
            pruning_mode=self.pruning_mode,
            block_size=self.block_size,
            custom_unprunable_module_classes=self.custom_unprunable_module_classes,
        )
        dag_orbitalized, found_final_orbits, prunable_kmapps, total_kmapp = orbitalizer.orbitalize(
            dag=self.model,
            prune_stem=self.prune_stem,
            input_shape=(1,) + self.input_shape_without_batch,
            return_stats=True,
        )
        self.prunable_proportion = prunable_kmapps / total_kmapp
        dag_orbitalized.cache_forward_dict = True
        with torch.no_grad():
            self.full_flops_list = build_full_flops_list(
                dag=dag_orbitalized,
                input_shape_without_batch=self.input_shape_without_batch,
            )
        self.model_orbitalized = dag_orbitalized
        self.target_normalized_flops = self._compute_target_normalized_flops()
        self.model_orbitalized.cache_forward_dict = True
        return self.model_orbitalized

    def _compute_target_normalized_flops(self):
        return self.pruning_proportion * self.initial_normalized_flops

    def compute_current_proportion_and_pruning_losses(self, global_step: int):
        """
        This function should be used after the forward pass to build the full loss that includes
        additional losses related to pruning.
        :return: a tuple (current proportion, flops loss, entropy loss, bkd loss)
        """
        # comput current flops value
        normalized_flops = compute_kmapp(
            dag=self.model_orbitalized,
            input_shape_without_batch=self.input_shape_without_batch,
            full_flops_list=self.full_flops_list,
        )

        # compute additional distillation loss
        bkd_loss_value = bkd_loss(self.model_orbitalized)
        # compute additional entropy loss value
        entropy_loss_value = entropy_loss(
            dag=self.model_orbitalized,
            global_step=global_step,
            lmbda=self.entropy_loss_weight,
            decay_steps=self.num_training_steps,
            annealing=self.anneal_losses,
        )

        flops_loss_value = latency_loss(
            computed_kmapp=normalized_flops,
            target_kmapp=self.target_normalized_flops,
            global_step=global_step,
            lmbda=self.flops_loss_weight,
            decay_steps=self.num_training_steps,
            annealing=self.anneal_losses,
        )
        current_proportion = normalized_flops / self.initial_normalized_flops
        return current_proportion, flops_loss_value, entropy_loss_value, bkd_loss_value

    def remove_channels(self) -> DagModule:
        self.model_orbitalized.to(CPU_DEVICE)
        return remove_channels_in_dag(dag=self.model_orbitalized, input_shape=(1,) + self.input_shape_without_batch)


def compute_normalized_flops(
        model: DagModule,
        input_shape_without_batch: Tuple[int, ...],
):
    """
    Computes flops normalized by input resolution. In essense:
    normalized_flops = flops / (H * W * 1e3)
    where (H, W) is input resolution
    """
    return compute_static_kmapp(model, input_shape_without_batch=input_shape_without_batch)
