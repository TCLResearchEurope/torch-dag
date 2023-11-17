# Quick intro to `torch-dag` pruning

## 0. Requirements

In addition to 
```
registry.gitlab.com/tcl-research/auto-ml/devine/tensorflow-experimental:stable
```

one needs to run:
```bash
pip install fvcore
pip install timm 
```

Using `torch-dag` on `k8s` can be accomplished by:
```bash
git clone https://gitlab+deploy-token-1843951:9AUWbQMy9cM4f3TvhARw@gitlab.com/tcl-research/auto-ml/torch-dag.git
```

Remember to add it to your `PYTHONPATH`. If there is an issue with `timm` (I had some)
just run:
```bash
git clone https://github.com/rwightman/pytorch-image-models.git
```
and remember to add it to your `PYTHONPATH`.

## 1. Model orbitalization
To add orbits to your `DagModule` just run
```bash
cd torch-dag/torch_dag/orbits
python orbitalize_model.py

  --model_path MODEL_PATH
  --saving_path SAVING_PATH
  --block_size BLOCK_SIZE
  --pruning_mode {default,block_snpe,whole_block}
  --input_shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Input shape to the orbitalized model (including batch dimension).
```
## 2. Runing training with orbits. 
This can be best explained in pure `torch` code. The example is taken from:
```
torch-dag/torch_dag/orbits/training/playground.py
```

```python
import logging

import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
from node_api_datasets.dataset_ops import dali_texture_augmentations
from node_api_datasets.dataset_ops.multiclass_augmentations import distortion_augmentations
from node_api_datasets.dataset_readers.dali_utils import normalization_params
from node_api_datasets.datasets import coco_pipeline
from torch_dag.dag_module import DagModule
from torch_dag.flops_computation import build_full_flops_list, compute_kmapp, log_dag_characteristics
from torch_dag.orbits.losses.bkd_losses import bkd_loss
from torch_dag.orbits.losses.entropy_losses import entropy_loss
from torch_dag.orbits.losses.latency_losses import latency_loss
from node_api_datasets.dataset_readers.dali_utils import dali_reader_constants
from torch_dag.dag_processors import clear_bkd_losses
from torch_dag.orbits.utils import per_channel_noise_to_signal_ratio
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch_dag.dag_processors import set_bn_to_eval
import os
from torch_dag.orbits.commons import get_orbits_dict

logger = logging.getLogger(__name__)

TEACHER_MODEL_PATH = '/nas/projects/auto-ml/torch_dag/models/yolov8n/'
MODEL_PATH = '/nas/projects/auto-ml/torch_dag/models/yolov8n/orbitalized'
SAVING_PATH = '/nas/projects/auto-ml/torch_dag/models/yolov8n/orbitalized/checkpoint'
NUM_STEPS = 100000
INPUT_SHAPE = (4, 3, 640, 640)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cpu':
    dali_reader_constants.CPU_DEBUG_MODE = True

augmentations = [
    dali_texture_augmentations.Brightness(),
    dali_texture_augmentations.Contrast(),
    dali_texture_augmentations.Saturation(),
    dali_texture_augmentations.Hue(),
    distortion_augmentations.FlipLeftRight(),
    distortion_augmentations.Rotate(),
    distortion_augmentations.TranslateX(),
    distortion_augmentations.TranslateY(),
]

pipeline = coco_pipeline.CocoPipeline(
    batch_size=16,
    image_size=(640, 640),
    mask_size=(640, 640),
    augmentations_sequence=augmentations,
    image_normalization_params=normalization_params.PRESET_ZERO_TO_ONE,
    mode='train',
)

dali_iter = DALIGenericIterator(pipelines=[pipeline], output_map=['data', 'label'])
global_step = 0

dag_orb = DagModule.load(MODEL_PATH)
dag_teacher = DagModule.load(TEACHER_MODEL_PATH)
dag_orb.eval()
dag_teacher.eval()

with torch.no_grad():
    log_dag_characteristics(dag_teacher, input_shape_without_batch=INPUT_SHAPE[1:])
    x = torch.ones(size=INPUT_SHAPE)
    _ = dag_orb(x)

    full_flops_list = build_full_flops_list(
        dag=dag_orb,
        input_shape_without_batch=INPUT_SHAPE[1:],
    )

dag_orb.clear_tensor_dicts()
dag_orb.cache_forward_dict = True
dag_teacher.cache_forward_dict = False

dag_teacher.to(DEVICE)
dag_orb.to(DEVICE)

dag_orb.train()
dag_orb.traverse(clear_bkd_losses)
dag_orb.traverse(set_bn_to_eval)

orbits_dict = get_orbits_dict(dag_orb)
orbits_params = []
for orb in orbits_dict.values():
    orbits_params.extend(orb.parameters())

optimizer = torch.optim.Adam(dag_orb.parameters(), lr=0.0001)
writer = SummaryWriter(log_dir=os.path.join(MODEL_PATH, 'logs'))

for step in trange(NUM_STEPS):
    optimizer.zero_grad()
    data = next(dali_iter)[0]
    x = data['data'].to(DEVICE)
    x = x.permute(0, 3, 1, 2)
    y = dag_orb(x)
    y_teacher = dag_teacher(x)

    kmapp = compute_kmapp(
        dag=dag_orb,
        input_shape_without_batch=INPUT_SHAPE[1:],
        full_flops_list=full_flops_list,
    )
    bkd_loss_value = bkd_loss(dag_orb)
    entropy_loss_value = entropy_loss(
        dag=dag_orb,
        global_step=step,
        lmbda=1.0,
        decay_steps=NUM_STEPS,
        annealing=True,
    )
    full_entropy_loss_value = entropy_loss(
        dag=dag_orb,
        global_step=step,
        lmbda=1.0,
        decay_steps=NUM_STEPS,
        annealing=False,
    )
    latency_loss_value = latency_loss(
        computed_kmapp=kmapp,
        target_kmapp=10.0,
        global_step=step,
        lmbda=10.0,
        decay_steps=NUM_STEPS,
        annealing=True,
    )
    nsr_loss = 0.0
    for t0, t1 in zip(y, y_teacher):
        nsr_loss += per_channel_noise_to_signal_ratio(
            x=t0,
            y=t1,
        )

    loss = nsr_loss + latency_loss_value + bkd_loss_value + entropy_loss_value
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dag_orb.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        logger.info(f'kmapp: {kmapp}')
        logger.info(f'bkd loss: {bkd_loss_value}')
        logger.info(f'entropy loss: {entropy_loss_value}')
        logger.info(f'full entropy loss: {full_entropy_loss_value}')
        logger.info(f'nsr loss: {nsr_loss}')

    if step % 10 == 0:
        writer.add_scalar('kmapp', kmapp, step)
        writer.add_scalar('bkd_loss', bkd_loss_value, step)
        writer.add_scalar('entropy-loss', entropy_loss_value, step)
        writer.add_scalar('full_entropy_loss', full_entropy_loss_value, step)
        writer.add_scalar('nsr_loss', nsr_loss, step)

    if step % 1000 == 0:
        dag_orb.save(SAVING_PATH)

```