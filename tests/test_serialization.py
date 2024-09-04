#
# Copyright © TCL Research Europe. All rights reserved.
#

import torch

import torch_dag as td
from .test_utils import TestBase


@td.register_notrace_module
class DummyDagModuleSubclass(td.DagModule):
    pass


td.ALLOWED_CUSTOM_MODULES += [DummyDagModuleSubclass]


class ModuleOne(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.silu = torch.nn.SiLU()
        self.channel_affine = td.ChannelAffineModule(num_channels=channels, use_bias=False)

    def forward(self, x):
        x = self.relu(x)
        x = self.channel_affine(x)
        x = self.silu(x)
        return x


class ModuleTwo(torch.nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 5)
        self.silu = torch.nn.SiLU()
        self.module_one_a = td.build_from_unstructured_module(ModuleOne(channels=channels), name='module_one_a')
        self.module_one_b = td.build_from_unstructured_module(ModuleOne(channels=channels), name='module_one_b')
        self.module_one_a = DummyDagModuleSubclass(
            name='module_one_a',
            vertices=self.module_one_a.vertices,
            output_vertex=self.module_one_a.output_vertex,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.silu(x)
        a = self.module_one_a(x)
        b = self.module_one_b(x)
        return a + b


class ModuleThree(torch.nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, channels, 3)
        self.silu = torch.nn.SiLU()
        self.module_two = td.build_from_unstructured_module(ModuleTwo(channels=channels))

    def forward(self, x):
        x = self.conv(x)
        x = self.silu(x)
        y = self.module_two(x)
        return y


class TestSerialzationOfDagModule(TestBase):

    def test_nested_serialization(self, tmpdir):
        module = ModuleThree()
        module.eval()
        dag = td.build_from_unstructured_module(module)
        dag.save(tmpdir)
        loaded_dag = td.DagModule.load(tmpdir)
        first_conv_weight = dag.inner_vertices[0].module.weight.detach()
        first_conv_weight_loaded = loaded_dag.inner_vertices[0].module.weight.detach()
        self.assert_array_equal(first_conv_weight, first_conv_weight_loaded)
        dummy_subclass_module = dag.inner_vertices[2].module.inner_vertices[2].module
        dummy_subclass_module_loaded = loaded_dag.inner_vertices[2].module.inner_vertices[2].module
        assert type(dummy_subclass_module_loaded) == type(dummy_subclass_module)
