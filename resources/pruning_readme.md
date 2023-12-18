# Channel pruning readme

## High level overview

In order to run channel pruning on your `torch` model, you have to do the following things:

### 1. Make sure you can convert your model to `torch-dag` `DagModule` format.

This can be accomplished by:

```python=
import torch_dag as td
model = MyModel(...)
dag_model = td.build_from_unstructured_module(model)

# run conversion sanity check
td.compare_module_outputs(first_module=model, second_module=dag_model, input_shape=(8, 3, 224, 224))
```

You can run this conversion either before or after the training of the original model.

> NOTE: You can run channel pruning on an untrained model, although it is usually better to train the baseline model
> first and only then prune it.

### 2. Compute initial model `FLOPs` (normalized):

```python=
import torch_dag as td
import torch_dag_algorithms as tda
initial_normalized_flops = tda.pruning.compute_normalized_flops(dag_model, input_shape_without_batch=INPUT_SHAPE[1:])
```

### 3. Specify:

* the `pruning proportion`, i.e., the target fraction of the baseline model size. This should be a number in `(0, 1)`.
* number of pruning training steps (:exclamation: for reasons related to optimization this should usually be > 10k
  steps)
* input shape (without the batch dimension), for example `(3, 224, 224)`

```python=
PRUNING_PROPORTION = 0.5  # target model size relative to the original model
NUM_PRUNING_STEPS = 10000
INPUT_SHAPE_WITHOUT_BATCH = (3, 224, 224)
```

### 4. Create the `pruning_config` as follows:

```python=
import torch_dag as td
import torch_dag_algorithms as tda
pruning_config = tda.pruning.ChannelPruning(
    model=dag_model,
    input_shape_without_batch=INPUT_SHAPE_WITHOUT_BATCH,
    pruning_proportion=PRUNING_PROPORTION,
    num_training_steps=NUM_PRUNING_STEPS,
)
```

### 5. Prepare model for pruning:

This will (a) trigger a graph-search algorithm that will automatically look for places where channels can be pruned
and (b) insert special pruning-related modules.

```python=
pruning_model = pruning_config.prepare_for_pruning()
```

> NOTE: This preparation process will happen on `cpu`, so if your model was on `gpu` you have to move it back to `gpu`
> after the preparation step.

### 6. In your training loop, after the forward pass you need to compute additional pruning losses and metrics:

```python=
proportion, flops_loss_value, entropy_loss_value, bkd_loss_value = pruning_config.compute_current_proportion_and_pruning_losses(global_step=global_step)
```

Here `proportion` is the current model size (in `FLOPs` terms) relative to the original model.
Add `flops_loss_value`, `entropy_loss_value` and `bkd_loss_value`
to your task loss.

### 7. Remove channels after training:

```python=
pruning_model.to(CPU_DEVICE)
pruned_model = pruning_config.remove_channels()
```

### 8. Compute final model size relative to the original model:

```python
import torch_dag as td
import torch_dag_algorithms as tda

final_normalized_flops = tda.pruning.compute_normalized_flops(pruned_model,
                                                                        input_shape_without_batch=INPUT_SHAPE[1:])
final_proportion = final_normalized_flops / initial_normalized_flops
```

## Best practices. What to watch out for?

* Make sure the pruning proportion you specify is achievable. After you run:
  ```python
  pruning_model = pruning_config.prepare_for_pruning()
  ```
  check `pruning_config.prunable_proportion` to see the fraction of `FLOPs` that can
  be pruned.
* Try logging `entropy_loss_value`: the one from:
  ```python=
  proportion, flops_loss_value, entropy_loss_value, bkd_loss_value = pruning_config.compute_current_proportion_and_pruning_losses(global_step=global_step)
  ```
  High values suggest that the training with pruning should perhaps run for a larger number
  of steps, otherwise the target proportion of the model size may not be reached.
* The optimization setup should use `Adam` or `AdamW` optimizer with learning rate `0.0001`.
  Higher learning rates can also work, but not lower than `0.0001`.
* Longer training usually leads to better convergence and ensures that the final proportion
  of the original model size will be closer to the one we want to achieve. We suggest around `50k` - `100k`
  trainig steps when pruning.

## Supported models

### `timm` (==0.9.5)

For a full list of supported models, with the proprtion of prunable `FLOPs` for each model,
see [link](../resources/supported_models_table.md). 

#### Supported model families

In order to see all the models in a given family just run:

```python
import timm
timm.list_models(filter)
```
where filter can be for example `'deit3*'`.

* `beit*` (MLP pruning only)
* `caformer*` (excluding b36) - experimental support
* `convnext*` (including V2)
* `deit3*`
* `densenet*`
* `dla*`
* `efficientnet*`
* `efficientformerv2*` (MLP pruning only)
* `fbnet*`
* `flexivit*`
* `hardcorenas*`
* `hrnet*`
* `mobilenetv2*`
* `mobilenetv3*`
* `poolformer*`
* `regnet*`
* `repvit*`
* `resnet*`
* `resnext*`
* `seresnet*`
* `seresnext*`
* `tf_efficientnet*`
* `tf_mobilenetv3*`
* `vit*`

### Unsupported model families

> NOTE: The list below is not exhaustive. It just tries to capture the 
> model families that are popular and well known.

* `davit*`
* `cait*`
* `coat*`
* `crossvit*`
* `cs3darknet*`
* `cspdarknet*`
* `dm_nfnet*`
* `eca*`
* `edgenext*`
* `eva*`
* `levit*`
* `maxvit*`
* `mvitv2*`
* `repvgg*`
* `resmlp*`
* `swin*`
* `xcit*`
