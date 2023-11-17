import logging
from typing import Optional, Union

from torch_dag.core.dag_module import DagModule
from torch_dag_algorithms.pruning.modules import MaskModule, OrbitModule
from torch_dag_algorithms.pruning import commons
from torch_dag_algorithms.pruning.commons import get_source_out_channels
from torch_dag_algorithms.pruning.masking_insertion_strategy import MaskingInsertionStrategy
from torch_dag_algorithms.pruning.orbit import Orbit
from torch_dag.core import dag_module_utils

logger = logging.getLogger(__name__)


class MaskInserter:
    def __init__(
            self,
            masking_strategy: MaskingInsertionStrategy,
            block_size: Optional[int] = None
    ):
        self.masking_strategy = masking_strategy
        self.block_size = block_size

    def insert_masking(
            self,
            dag: DagModule,
            orbit: Orbit,
            orbit_module: OrbitModule,
    ):
        pairs = self.masking_strategy.find_reference_nodes(orbit)
        for i, (start_vertex, end_vertex) in enumerate(pairs):
            masker = MaskModule(
                orbit=orbit_module,
            )
            mask_vertex = dag_module_utils.insert_between(
                dag=dag,
                name=f'{orbit_module.name}_mask_{i}',
                after_vertex=start_vertex,
                new_module=masker,
                before_vertex=end_vertex,
            )
            mask_vertex.orbit = orbit_module.name

    def create_orbit(
            self,
            name: str,
            dag: DagModule,
            orbit: Orbit,
            pruning_mode: str,
            skip_orbits_with_channels_less_than_block_size: bool
    ) -> Union[OrbitModule, commons.Skipped]:
        num_channels = get_source_out_channels(orbit.sources[0].module)

        if skip_orbits_with_channels_less_than_block_size and self.block_size and num_channels <= self.block_size:
            logger.info(
                f' [!] Skipping orbit: {orbit} due to num channels[{num_channels}] <= block_size[{self.block_size}]')
            return commons.Skipped()

        indices_of_source_vertices = [dag.vertices.index(v) for v in orbit.sources]
        orbit_module = OrbitModule(
            name=name,
            distillation_mode=pruning_mode,
            num_channels=num_channels,
            indices_of_source_vertices=indices_of_source_vertices,
            block_size=self.block_size,
        )
        for vertex in orbit.vertices_in_scope:
            if vertex not in orbit.sinks or vertex in orbit.sources:
                vertex.orbit = name
        self.insert_masking(
            dag=dag,
            orbit=orbit,
            orbit_module=orbit_module,
        )

        return orbit_module
