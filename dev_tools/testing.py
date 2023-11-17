import logging
from dataclasses import field, dataclass
from typing import List, Tuple

import torch
from tqdm import trange

import torch_dag as td
from dev_tools import algorithms_testing_tools as tt
from dev_tools import constants as c
from dev_tools import model_loading
from dev_tools import testing_result_registry as rr
from dev_tools import timeout
from torch_dag.core.unstructured_to_structured import build_from_unstructured_module

logger = logging.getLogger(__name__)

TIMEOUT = 6 * 60
MODEL_SIZE_THRESHOLD = 1000


@dataclass
class TesterResult:
    success: bool
    message: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "success": self.success,
            "message": self.message,
            **self.extras,
        }


class Converter:
    @classmethod
    def run(cls, model, input_shape):
        if isinstance(model, td.DagModule):
            return model
        elif isinstance(model, torch.nn.Module):
            return build_from_unstructured_module(model)
        else:
            raise Exception(f"Unsupported conversion: {type(model)} -> td.DagModule")


class Tester:
    NAME = None
    converter = Converter()

    def run(self, model_loader) -> TesterResult:
        try:
            model, input_shape = model_loader.load()
            td_model = self.converter.run(model, input_shape=input_shape)
            assert td_model is not None
            return self.test(td_model, input_shape)
        except Exception as e:
            return TesterResult(False, str(e))


class WrappingTester(Tester):
    NAME = c.CONVERSION_TO_DAG

    def test(self, model: td.DagModule, input_shape: Tuple[int, ...]) -> TesterResult:
        return TesterResult(True, "")




class ChannelPruningTester(Tester):
    NAME = c.CHANNEL_PRUNING_NAME

    def __init__(self, prob_removal: float) -> None:
        super().__init__()
        self.prob_removal = prob_removal

    def test(self, model: td.DagModule, input_shape: Tuple[int, ...]) -> TesterResult:
        r = tt.test_orbitalization_and_channel_removal(
            dag=model,
            input_shape=input_shape,
            prob_removal=self.prob_removal,
        )
        succ = r["status"] == c.SUCCESS_RESULT
        message = str(r["reason"])
        score = r["prunable_fraction"]
        return TesterResult(succ, message, {"score": score})


def get_tests(channel_prob_removal=0.1):
    return [
        WrappingTester(),
        ChannelPruningTester(channel_prob_removal),
    ]


@timeout.timeout(TIMEOUT)
def test_model(model_loader: model_loading.ModelLoader, testers: List[Tester]):
    results = {}
    for t in testers:
        try:
            r = timeout.timeout(TIMEOUT)(t.run)(model_loader)
            results[t.NAME] = r.to_dict()
        except timeout.TimeoutError as e:
            logger.warning(f"Timeout on {model_loader.model_name}: {str(e)}")
    return results


def test_models(
        model_loaders: List[model_loading.ModelLoader],
        tests: List[Tester],
        result_registry=None,
):
    results = {}
    for id in trange(len(model_loaders)):
        model_loader = model_loaders[id]
        logger.info(f"{model_loader.model_name}")
        try:
            r = test_model(model_loader, tests)
            results[model_loader.model_name] = r
            if result_registry is not None:
                if rr.is_jsonable({model_loader.model_name: r}):
                    result_registry.update(model_loader.model_name, r)
        except timeout.TimeoutError as e:
            logger.warning(f"Timeout on {model_loader.model_name}: {str(e)}")
    return results
