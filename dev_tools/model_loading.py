import os
from typing import Tuple

import timm
import torch

import torch_dag as td

MODEL_SIZE_THRESHOLD = 1000


class ModelLoader:
    """
    This class ensures that model loading errors appear in individual tests
    instead of crashing/skipping tests for these models.
    """

    pass

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def load(self, pretrained=False):
        raise NotImplementedError


class TimmModelLoader(ModelLoader):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self, pretrained=False):
        model = timm.create_model(self._model_name, pretrained=False)
        size = sum(p.numel() for p in model.parameters()) / 1000000
        input_shape = (1,) + tuple(model.default_cfg["input_size"])
        if size > MODEL_SIZE_THRESHOLD:
            raise Exception(f"Model too large: {size:.2f}M params")
        if pretrained:
            model = timm.create_model(self._model_name, pretrained=pretrained)
        return model, input_shape


class NasModelLoader(ModelLoader):
    def __init__(self, model_path: str, input_shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.model_path = model_path
        self.input_shape = input_shape

    def load(self, pretrained=False):
        if os.path.isdir(self.model_path) and "config.dict.json" in os.listdir(
                self.model_path
        ):
            return td.io.load_dag_from_path(self.model_path), self.input_shape
        else:
            return torch.load(self.model_path), self.input_shape

    @property
    def model_name(self) -> str:
        return self.model_path
