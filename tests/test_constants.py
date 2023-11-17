from dataclasses import dataclass
from typing import List


@dataclass
class TimmModelTestCase:
    timm_name: str
    channel_pruning_threshold: float
    test_regular_tracing: bool
    test_compile: bool
    prob_removal: float = 0.1


base_test_cases: List[TimmModelTestCase] = [
    TimmModelTestCase('resnetv2_50d', 0.95, True, True),
    TimmModelTestCase('tf_mobilenetv3_large_100', 0.95, False, False),
    TimmModelTestCase('tf_efficientnetv2_b0', 0.95, False, False, True),
    TimmModelTestCase('mobilenetv3_small_100', 0.95, True, False, True),
]

extra_test_cases: List[TimmModelTestCase] = [
    TimmModelTestCase('deit3_small_patch16_224', 0.5, False, True),
    TimmModelTestCase('vit_small_patch16_224', 0.5, False, True),
    TimmModelTestCase('efficientformerv2_s0', 0.55, False, True),
    TimmModelTestCase('convnextv2_nano', 0.95, True, True),
    TimmModelTestCase('hardcorenas_c', 0.97, True, False),
    TimmModelTestCase('rexnetr_200.sw_in12k_ft_in1k', 0.83, True, True, 0.0),
    TimmModelTestCase('beit_base_patch16_224.in22k_ft_in22k_in1k', 0.52, False, True),
    TimmModelTestCase('poolformerv2_s12.sail_in1k', 0.8, False, True),
    TimmModelTestCase('caformer_s18', 0.9, False, True, 0.0),
    TimmModelTestCase('xcit_nano_12_p8_224', 0.63, False, False, 0.0),
    TimmModelTestCase('edgenext_small', 0.84, False, False, 0.0),
    TimmModelTestCase('crossvit_tiny_240', 0.85, False, True, 0.0),
    TimmModelTestCase('mobilevitv2_050', 0.6, False, True, 0.0),
]

test_cases = base_test_cases + extra_test_cases
