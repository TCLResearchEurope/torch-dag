#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union, Type

import torch
from torch import nn

from torch_dag_algorithms.pruning import masking_insertion_strategy, filters, masking_inserter, orbits_extraction
from torch_dag_algorithms.pruning.commons import TRUNCATE_ON, Skipped
from torch_dag_algorithms.pruning.constants import PRUNING_DEFAULT_MODE_NAME
from torch_dag_algorithms.pruning.orbit import Orbit
from torch_dag_algorithms.pruning.orbits_search_stage import OrbitsDiscoveryStage
from torch_dag.commons.flops_computation import compute_torch_flops_for_dag_module
from torch_dag.core.dag_module import DagModule
from torch_dag.patterns.crude_patterns import ReverseGraphPattern
from torch_dag.patterns.specific_patterns import PRUNABLE_ATTENTION_PATTERN

logger = logging.getLogger(__name__)


def log_kmapps_stats(
        dag: DagModule,
        input_shape_without_batch: Tuple[int, ...],
        orbits: List[Orbit],
) -> Union[Tuple[float, float], None]:
    if len(input_shape_without_batch) != 3:
        return
    normalization = input_shape_without_batch[1] * input_shape_without_batch[2] * 1e3
    flops_dict = compute_torch_flops_for_dag_module(dag, input_shape_without_batch)
    normalized_flops_dict = {k: v * 2.0 / normalization for k, v in flops_dict.items()}

    for orbit in orbits:
        kmapps = 0.0

        for icn in list(orbit.sources) + list(orbit.non_border):
            kmapps += normalized_flops_dict[icn]

        for sink in orbit.sinks:
            if sink.orbit is None:
                kmapps += normalized_flops_dict[sink]

        orbit.kmapps = kmapps

    total_kmapp = sum(normalized_flops_dict.values())
    prunable_kmapps = sum([orbit.kmapps for orbit in orbits])
    unprunable_kmapp = (sum(normalized_flops_dict.values()) - prunable_kmapps)

    logger.info(f'[+] Total normalized flops: {total_kmapp}')
    logger.info(f'[+] Prunable normalized flops: {prunable_kmapps}')
    logger.info(f'[+] Unprunable normalized flops: {unprunable_kmapp}')
    logger.info(f'[+] Normalized flops per orbit:')
    for orbit in orbits:
        logger.info(f'Orbit[color={orbit.color}]: normalized flops={orbit.kmapps:.2f}')

    return prunable_kmapps, total_kmapp


class Orbitalizer(ABC):
    """
    Class that is used to orbitalize dag.
    Orbitalizing means adding channel ranker nodes and masking operations to cell so the pruning-aware training can be performed.
    """

    @abstractmethod
    def orbitalize(self, dag: DagModule, *args, **kwargs) -> Tuple[DagModule, Optional[List[Orbit]]]:
        """
        Function that performs orbitalization. It takes cell and return orbitalized version of recived dag.
        """
        raise NotImplementedError


class GeneralOrbitalizer(Orbitalizer):
    def __init__(
            self,
            pruning_mode: str = PRUNING_DEFAULT_MODE_NAME,
            truncate_on: nn.Module = TRUNCATE_ON,
            block_size: Optional[int] = None,
            custom_unprunable_module_classes: Tuple[Type[torch.nn.Module]] = (),
    ):
        self.pruning_mode = pruning_mode
        self.truncate_on = truncate_on
        self.block_size = block_size
        self.custom_unprunable_module_classes = custom_unprunable_module_classes

    def check_and_handle_known_custom_patterns(self, extended_orbit: Orbit) -> Union[Orbit, List[Orbit]]:
        attention_reverse_graph_pattern = ReverseGraphPattern(
            pattern=PRUNABLE_ATTENTION_PATTERN['pattern'],
            starting_key=PRUNABLE_ATTENTION_PATTERN['starting_key'],
            named_specs=PRUNABLE_ATTENTION_PATTERN['named_specs'],
        )
        extracted_custom_orbits = []
        for vertex in extended_orbit.vertices_in_scope:
            # [final_matmul, softmax, value, attention_matmul, query, key]

            result = attention_reverse_graph_pattern.match_overall_pattern(vertex)
            # make sure the result vertices are inside extended orbit
            if result:
                contained_in_extended_orbit = all([e in extended_orbit.vertices_in_scope for e in result])
                if contained_in_extended_orbit:
                    final_matmul, softmax, value, attention_matmul, query, key = result
                    key_query_orbit_color = next(orbits_extraction.COLOR_GENERATOR)
                    key_query_orbit = Orbit(color=key_query_orbit_color)

                    key_query_orbit.add_to_scope(key)
                    key_query_orbit.add_to_scope(query)
                    key_query_orbit.add_to_scope(attention_matmul)

                    key_query_orbit.mark_as_source(query)
                    key_query_orbit.mark_as_source(key)

                    key_query_orbit.mark_end_path_node_and_sink(key, attention_matmul)
                    key_query_orbit.mark_end_path_node_and_sink(query, attention_matmul)

                    value_orbit_color = next(orbits_extraction.COLOR_GENERATOR)
                    value_orbit = Orbit(color=value_orbit_color)
                    value_orbit.add_to_scope(value)
                    value_orbit.mark_as_source(value)
                    value_orbit.mark_end_path_node_and_sink(value, final_matmul)

                    extracted_custom_orbits.append(key_query_orbit)
                    extracted_custom_orbits.append(value_orbit)
                    logger.info(f'Found custom orbit: {key_query_orbit}')
                    logger.info(f'Found custom orbit: {value_orbit}')

        if extracted_custom_orbits:
            return extracted_custom_orbits

        return extended_orbit

    def orbitalize(
            self,
            dag: DagModule,
            prune_stem: bool = False,
            vis_final_orbits: bool = True,
            input_shape: Tuple[int, ...] = [(1, 3, 256, 256)],
            vis_saving_dir: str = None,
            skip_orbits_with_channels_less_than_block_size: bool = False,
            remove_tensor_mergers_and_extractors: bool = True,
            return_stats: bool = False,
            force_log_stats: bool = True,
    ) -> Union[Tuple[DagModule, List[Orbit]], Tuple[DagModule, List[Orbit], float, float]]:
        """Function that performs orbitalization. It takes cell and return orbitalized version of recived cell. It can optionally save visualization of final orbits for a given cell - by default it's saved to `./<cell.name>.pdf`.

        Args:
            :dag: (DagModule): DagModule to be orbitalized
            :prune_stem: (bool, optional): Flag indicating whether stem should be pruned. By stem we understand SOURCE_TYPES nodes that have nd.cells.InputCellNode as predecessor. Defaults to False.
            :vis_final_orbits: (bool, optional): Flag indicating whether visualization of orbitalized cell should be saved. Visualization is done based on final orbits. By default it saved under `./<cell.name>.pdf` path. Defaults to True.
            :input_shape: (Optional[List[Tuple[int]]], optional): Input shapes that will be used to calculate kmapps and during visualization to calculate shapes on a given layer. Defaults to [(1, 384, 384, 3)].
            :skip_orbits_with_channels_less_than_block_size: (bool, optional): Flag indicating whether orits with channels_num less than block_size should be skipped durining orbitalization.
            :remove_tensor_mergers_and_extractors: (bool, optional): Flag indicating whether tensors mergers and tensor extractors should be removed from flatten cell before orbitalization process.

        Returns:
            Tuple[nd.cells.Cell, List[Orbit]]: Orbitalized cell with removed nd.ops.TensorMergers and nd.ops.TensorExtractors.
        """
        if dag.device.type != 'cpu':
            raise AssertionError('Orbits can only be added when the `DagModule` is on CPU.')
        dag = dag.flatten(input_shape_for_verification=input_shape)

        # TODO: add this missing functionality
        # if remove_tensor_mergers_and_extractors:
        #     nd.cells.utils.remove_tensor_mergers_and_extractors_from_flattened_cell(new_cell)

        extended_orbits_filters = [
            filters.JoinOpAfterConcatPresentFilter(),
            filters.InputInScopeFilter(input_vertices=dag.input_vertices),
            filters.OutputInScopeFilter(output_vertex=dag.output_vertex),
            filters.ProperGroupedConvolutionPresent(),
            filters.NonPrunableCustomModulesFilter(custom_unprunable_modules=self.custom_unprunable_module_classes),
        ]

        found_extended_orbits = orbits_extraction.extract_orbits(
            vertices=dag.vertices,
            discovery_stage=OrbitsDiscoveryStage.EXTENDED_ORBIT_DISCOVERY
        )

        # TODO: do we really want to run filters on orbit that was created by custom handle too? Shouldn't filters run only on raw extended orbits?
        found_extended_orbits = [
            orbit for orbit in found_extended_orbits
            if orbit.is_valid(extended_orbits_filters)
        ]

        extended_orbits: List[Orbit] = []
        for orbit in found_extended_orbits:
            processed_orbit = self.check_and_handle_known_custom_patterns(orbit)
            extended_orbits += processed_orbit if isinstance(processed_orbit, list) else [processed_orbit]

        found_final_orbits: List[Orbit] = []
        for extended_orbit in extended_orbits:
            # if `discovery_stage != OrbitsDiscoveryStage.EXTENDED_ORBIT_DISCOVERY` it means that the orbit was created via custom known pattern handler in `check_and_handle_known_custom_patterns` method, hence we don't want to process it as extended orbit and re-run search of final orbits on it. We just take it as it is.
            if extended_orbit.discovery_stage == OrbitsDiscoveryStage.EXTENDED_ORBIT_DISCOVERY:
                found_final_orbits += orbits_extraction.extract_orbits(
                    truncate_on=self.truncate_on,
                    vertices=extended_orbit.vertices_in_scope,
                    sources=extended_orbit.sources,
                    discovery_stage=OrbitsDiscoveryStage.FINAL_ORBIT_DISCOVERY
                )
            else:
                found_final_orbits += [extended_orbit]

        final_orbits_filters = [
            filters.NonPrunableCustomModulesFilter(custom_unprunable_modules=self.custom_unprunable_module_classes),
        ]

        if not prune_stem:
            final_orbits_filters += [filters.StemPresentFilter(dag=dag)]

        found_final_orbits = [orbit for orbit in found_final_orbits if orbit.is_valid(final_orbits_filters)]

        masking_strategy = masking_insertion_strategy.AtTheEndOfPathStrategy()
        inserter = masking_inserter.MaskInserter(
            masking_strategy=masking_strategy,
            block_size=self.block_size
        )

        skipped = []
        for k, final_orbit in enumerate(found_final_orbits):
            res = inserter.create_orbit(
                name=f'orbit_{k}',
                dag=dag,
                orbit=final_orbit,
                pruning_mode=self.pruning_mode,
                skip_orbits_with_channels_less_than_block_size=skip_orbits_with_channels_less_than_block_size
            )
            if isinstance(res, Skipped):
                skipped += [final_orbit]

        found_final_orbits = [orbit for orbit in found_final_orbits if orbit not in skipped]

        if return_stats:
            prunable_kmapps, total_kmapp = log_kmapps_stats(dag, input_shape[1:], found_final_orbits)
            return dag, found_final_orbits, prunable_kmapps, total_kmapp
        else:
            if force_log_stats:
                prunable_kmapps, total_kmapp = log_kmapps_stats(dag, input_shape[1:], found_final_orbits)
            return dag, found_final_orbits
