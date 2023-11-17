from typing import List
from typing import Optional

import numpy as np
import torch
from torch_dag.commons.mixins import ZeroFlopsMixin, ZeroParamsMixin

from torch_dag.core.dag_tracer import register_notrace_module
from torch_dag_algorithms.pruning import constants
from torch_dag_algorithms.pruning.utils import per_channel_noise_to_signal_ratio

PRUNING_MODES = (
    constants.PRUNING_BLOCK_SNPE_MODE_NAME,
    constants.PRUNING_DEFAULT_MODE_NAME,
    constants.PRUNING_WHOLE_BLOCK_MODE_NAME,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_split_list_of_logits(
        logits: torch.Tensor,
        block_size: int,
) -> List[int]:
    num_full_blocks = int(logits.shape[0]) // block_size
    residual = int(logits.shape[0]) % block_size
    split_list = num_full_blocks * [block_size]
    if residual != 0:
        split_list.append(residual)
    return split_list


def get_sorted_per_block_mean_logits(
        logits: torch.Tensor,
        block_size: int,
) -> List[torch.Tensor]:
    sorted_logits = torch.sort(logits, descending=True)[0]
    split_list = get_split_list_of_logits(logits, block_size=block_size)
    split_sorted_logits = torch.split(sorted_logits, split_size_or_sections=split_list)
    return [block.mean() for block in split_sorted_logits]


def get_sorted_per_block_max_logits(
        logits: torch.Tensor,
        block_size: int,
) -> List[torch.Tensor]:
    """
    Sorts the logits, splits them into blocks of size `block_size` (if num logits
    is not divisible by `block_size`, then there is a residual) and then computes maximum
    logits per each such block.
    :param logits:
    :return: list of scalar tensors
    """
    sorted_logits = torch.sort(logits, descending=True)[0]
    split_list = get_split_list_of_logits(logits, block_size=block_size)
    split_sorted_logits = torch.split(sorted_logits, split_size_or_sections=split_list)
    return [block.max() for block in split_sorted_logits]


def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_ = torch.cat([logits[:, None], torch.zeros_like(logits)[:, None]], dim=1)
    gs_sample = torch.nn.functional.gumbel_softmax(
        logits_,
        tau=constants.DEFAULT_TEMPERATURE,
        hard=False,
    )[:, 0]
    hard_mask_p = torch.sigmoid(logits).mean()
    hard_choice = torch.distributions.Bernoulli(probs=hard_mask_p).sample(sample_shape=logits.shape)
    return torch.where(
        hard_choice == 1,
        torch.ones_like(hard_choice).to(torch.float32),
        gs_sample
    )


@register_notrace_module
class OrbitModule(torch.nn.Module):

    def __init__(
            self,
            name: str,
            num_channels: int,
            distillation_mode: str = constants.PRUNING_DEFAULT_MODE_NAME,
            block_size: Optional[int] = None,
            indices_of_source_vertices=None,
    ):
        super().__init__()
        self.name = name
        self.num_channels = num_channels
        self.distillation_mode = distillation_mode
        self.block_size = block_size
        self._testing_logits = None
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel_size=3, groups=num_channels)
        self.conv2 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
        )
        self._optionally_set_block_size_for_whole_block_pruning(distillation_mode=distillation_mode)
        self._validate_distilation_mode_and_block_size(distillation_mode=distillation_mode, block_size=block_size)
        self.bkd_masking_losses = {}
        self.indices_of_source_vertices = indices_of_source_vertices
        self.debug_logits = None

    def _validate_distilation_mode_and_block_size(self, distillation_mode: str, block_size: int):
        if distillation_mode not in PRUNING_MODES:
            raise NotImplementedError(f'Distillation mode: {distillation_mode} not supported')
        if distillation_mode == constants.PRUNING_BLOCK_SNPE_MODE_NAME and block_size is None:
            raise AssertionError(f'In {constants.PRUNING_BLOCK_SNPE_MODE_NAME} pruning mode block size must not '
                                 f'be `None`.')

    def _optionally_set_block_size_for_whole_block_pruning(self, distillation_mode: str):
        if distillation_mode == constants.PRUNING_WHOLE_BLOCK_MODE_NAME:
            self.block_size = self.num_channels

    @staticmethod
    def clip_logits(
            logits: torch.Tensor,
            clip_val=constants.MAX_LOGITS_ABS_VALUE,
    ) -> torch.Tensor:
        return torch.clip(logits, min=-clip_val, max=clip_val)

    @property
    def logits(self) -> torch.Tensor:
        # TODO This is a hack for testing, remove/refactor it
        if self.debug_logits is not None:
            return self.debug_logits
        kernel_size = self.conv1.kernel_size
        device = self.conv1.weight.device
        x = torch.ones(size=(1, self.num_channels, *kernel_size), device=device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = (constants.INITIAL_LOGITS_VALUE_FOR_PRUNING + constants.SIMPLE_ORBIT_LOGITS_MULTIPLIER * x)
        return self.clip_logits(torch.mean(x, dim=(0, 2, 3), keepdim=False))

    def compute_average_number_of_output_channels(self):
        if self.distillation_mode == constants.PRUNING_DEFAULT_MODE_NAME:
            return torch.sigmoid(self.logits).sum()

        elif self.distillation_mode in (
                constants.PRUNING_BLOCK_SNPE_MODE_NAME, constants.PRUNING_WHOLE_BLOCK_MODE_NAME):
            split_list = get_split_list_of_logits(logits=self.logits, block_size=self.block_size)
            max_per_block_logits = get_sorted_per_block_max_logits(
                logits=self.logits,
                block_size=self.block_size,
            )
            num_channels = torch.stack(
                [float(block_size) * torch.sigmoid(max_logit) for \
                 block_size, max_logit in zip(split_list, max_per_block_logits)], dim=0).sum()
            return num_channels
        else:
            msg = f'Mode {self.distillation_mode} not implemented for average channels computation.'
            raise NotImplementedError(msg)

    def compute_output_channel_masks(
            self,
            predecessors_channel_masks: List[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        predecessors_channel_masks = [mask_list for mask_list in predecessors_channel_masks if mask_list is not None]
        logits = self.logits
        num_logits = int(logits.shape[0])
        if self.distillation_mode == constants.PRUNING_DEFAULT_MODE_NAME:
            scores_ = torch.where(
                logits > 0.0,
                1,
                0,
            )
        elif self.distillation_mode == constants.PRUNING_WHOLE_BLOCK_MODE_NAME:
            max_logits_per_block = get_sorted_per_block_max_logits(
                logits=logits,
                block_size=self.block_size,
            )
            max_logits_per_block_tensor = torch.stack(max_logits_per_block)
            indices_of_blocks_to_leave = np.where(max_logits_per_block_tensor > 0.)[0]
            if len(indices_of_blocks_to_leave) == 1:
                scores_ = np.ones(shape=(self.block_size,), dtype=np.int32)
            else:
                scores_ = np.zeros(shape=(self.block_size,), dtype=np.int32)

        elif self.distillation_mode == constants.PRUNING_BLOCK_SNPE_MODE_NAME:
            max_logits_per_block = get_sorted_per_block_max_logits(
                logits=logits,
                block_size=self.block_size,
            )
            max_logits_per_block_tensor = torch.stack(max_logits_per_block)
            indices_of_blocks_to_leave = np.where(max_logits_per_block_tensor > 0.)[0]
            if len(indices_of_blocks_to_leave) == 0:
                # removing whole orbit
                scores_ = np.zeros(shape=(self.num_channels,), dtype=np.int32)

            else:
                # compute block indices that are left
                sorted_logits = torch.sort(logits, descending=True)[0]
                split_list = get_split_list_of_logits(logits=logits, block_size=self.block_size)
                split_sorted_logits = list(torch.split(sorted_logits, split_list))
                residual = num_logits % self.block_size
                if residual != 0:
                    logits_fake_tail = split_sorted_logits[-1].mean() * torch.ones(
                        size=(self.block_size - residual,))
                    split_sorted_logits[-1] = torch.cat([split_sorted_logits[-1], logits_fake_tail], dim=0)
                split_sorted_logits = [e.detach().numpy() for e in split_sorted_logits]
                if len(split_sorted_logits) == 1:
                    res = split_sorted_logits
                else:
                    res = np.take(
                        split_sorted_logits,
                        axis=0,
                        indices=indices_of_blocks_to_leave,
                    )
                threshold_value = torch.tensor(res).min()
                scores_ = np.where(
                    logits >= threshold_value,
                    1,
                    0,
                )
        else:
            raise NotImplementedError

        if len(predecessors_channel_masks) == 0:
            return [torch.tensor(scores_)]
        else:
            return [torch.tensor(np.where(
                predecessors_channel_masks[0][0].sum() == 0,
                np.array([0] * self.num_channels, dtype=np.int32),
                scores_,
            ))]

    def sample(self):
        return sample_from_logits(logits=self.logits)


@register_notrace_module
class MaskModule(torch.nn.Module, ZeroFlopsMixin, ZeroParamsMixin):
    def __init__(self, orbit: OrbitModule):
        super().__init__()
        self.orbit = orbit

    def forward(self, x: torch.Tensor):
        mask = self.orbit.sample()
        # (B, C, H, W) case
        if len(x.shape) == 4:
            non_channel_dim = (0, 2, 3)
            mask = mask.view(1, -1, 1, 1)
        # (B, T, dim) case
        elif len(x.shape) == 3:
            non_channel_dim = (0, 1)
            mask = mask.view(1, 1, -1)
        elif len(x.shape) == 2:
            non_channel_dim = (0,)
            mask = mask.view(1, -1)
        else:
            raise NotImplementedError

        try:
            x_masked = mask * x
        except:
            # TODO: find more elegent solution to this channel dim switching issue
            mask = mask.view(1, 1, 1, -1)
            x_masked = mask * x
        if self.training:
            bkd_loss = per_channel_noise_to_signal_ratio(x_masked, x, non_channel_dim=non_channel_dim)
            self.orbit.bkd_masking_losses[self] = bkd_loss
        return x_masked
