import os

import pytest
import torch

import torch_dag as td

ROOT_DIR = '/nas/projects/auto-ml/torch_dag/optimized_models/pruned'


def walk_through(root_dir: str):
    result = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            path = os.path.abspath(os.path.join(dirpath, dirname))
            if 'config.dict.json' in os.listdir(path):
                result.append(path)
    return result


paths = walk_through(ROOT_DIR)


@pytest.mark.ioheavy
@pytest.mark.parametrize(
    'path',
    paths,
)
def test_load_optimized_models(path):
    dag = td.io.load_dag_from_path(path)
    x = torch.ones(size=(2, 3, 224, 224))
    _ = dag(x)
