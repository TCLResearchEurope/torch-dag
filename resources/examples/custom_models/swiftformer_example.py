import torch_dag_algorithms.commons.flops_computation
from SwiftFormer.models.swiftformer import SwiftFormer_S, EfficientAdditiveAttnetion
import torch
import torch_dag as td
import torch_dag_algorithms
from torch_dag_experiments.look_for_dagable_modules import look_for_dagable_modules

model = SwiftFormer_S()
look_for_dagable_modules(model)
input_shape=(1, 3, 224, 224)

custom_module_classes = (EfficientAdditiveAttnetion,)
dag = td.build_from_unstructured_module(
    model,
    custom_autowrap_torch_module_classes=custom_module_classes,
)

orbitalizer = torch_dag_algorithms.pruning.GeneralOrbitalizer(
    pruning_mode='default',
    block_size=None,
    custom_unprunable_module_classes=(EfficientAdditiveAttnetion, )
)
dag_orb, orbits = orbitalizer.orbitalize(
    dag=dag,
    vis_final_orbits=False,
    prune_stem=True,
    input_shape=input_shape,
)
dag_orb.cache_forward_dict = True

full_kmapp = td.commons.compute_static_kmapp(dag, input_shape_without_batch=input_shape[1:])

# randomize channel importance (for dummy pruning)
orbits_dict = torch_dag_algorithms.pruning.get_orbits_dict(dag_orb)
for k, v in orbits_dict.items():
    num_channels = v.num_channels
    v.debug_logits = torch.normal(mean=torch.zeros(size=(num_channels,)))

full_flops_list = td.commons.build_full_flops_list(dag_orb, input_shape_without_batch=input_shape[1:])
pre_kmapp = torch_dag_algorithms.commons.flops_computation.compute_kmapp(dag_orb, input_shape_without_batch=input_shape[1:], full_flops_list=full_flops_list)

dag_final = torch_dag_algorithms.pruning.remove_channels_in_dag(dag_orb, input_shape=(1, 3, 224, 224))
post_kmapp = td.commons.compute_static_kmapp(dag_final, input_shape_without_batch=input_shape[1:])

dag_final.save('/opt/devine/test_model')

print(full_kmapp)
print(pre_kmapp)
print(post_kmapp)
