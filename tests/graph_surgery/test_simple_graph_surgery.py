#
# Copyright © TCL Research Europe. All rights reserved.
#
import logging

import torch

# from torch_dag.core import dag_module_utils
# from torch_dag.dag_module import DagModule, InnerVertex
from tests.test_utils import TestBase
# from torch_dag.unstructured_to_structured import build_from_unstructured_module
import torch_dag as td

logger = logging.getLogger(__name__)


class TestModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, 3)
        self.bn = torch.nn.BatchNorm2d(64, 3)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class NoShapeChangeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(64, 3)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.bn(x)
        return self.act(x)


class NestedTestModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, 3)
        self.bn = torch.nn.BatchNorm2d(64, 3)
        self.inner_dag = td.build_from_unstructured_module(NoShapeChangeModule())
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.inner_dag(x)
        return self.act(x)


class TestGraphSurgery(TestBase):

    def test_module_switch(self, tmpdir):
        model = TestModule()
        dag = td.build_from_unstructured_module(model)
        new_act = torch.nn.ReLU()
        dag.inner_vertices[-1].module = new_act
        dag.save(tmpdir)
        loaded_dag = td.DagModule.load(tmpdir)
        x = self.get_random_tensor(shape=(1, 32, 6, 6))
        dag.eval()
        loaded_dag.eval()
        y = dag(x)
        y_loaded = loaded_dag(x)
        self.assert_all_close(y, y_loaded)
        assert dag.inner_modules[-1] == new_act

    def test_vertex_removal(self):
        model = TestModule()
        dag = td.build_from_unstructured_module(model)
        vertex_to_remove = dag.inner_vertices[1]
        td.remove_vertex(dag, vertex=vertex_to_remove)
        x = self.get_random_tensor((8, 32, 6, 6))
        _ = dag(x)
        assert len(dag.inner_vertices) == 2

    def test_output_vertex_removal(self):
        model = TestModule()
        dag = td.build_from_unstructured_module(model)
        vertex_to_remove = dag.output_vertex
        td.remove_vertex(dag, vertex=vertex_to_remove)
        x = self.get_random_tensor((8, 32, 6, 6))
        _ = dag(x)
        assert len(dag.inner_vertices) == 2

    def test_insert_before(self):
        model = TestModule()
        dag = td.build_from_unstructured_module(model)
        new_module = torch.nn.ReLU6()
        td.insert_before(
            dag, reference_vertex=dag.inner_vertices[0], new_module=new_module, name='relu6')
        x = self.get_random_tensor((8, 32, 6, 6))
        _ = dag(x)
        assert dag.inner_vertices[0].module == new_module

    def test_insert_after(self):
        model = TestModule()
        dag = td.build_from_unstructured_module(model)
        new_module = torch.nn.ReLU6()
        td.insert_after(
            dag, reference_vertex=dag.inner_vertices[0], new_module=new_module, name='relu6')
        x = self.get_random_tensor((8, 32, 6, 6))
        _ = dag(x)
        assert dag.inner_vertices[1].module == new_module

    def test_traverse_act_change(self, tmpdir):
        model = NestedTestModule()
        dag = td.build_from_unstructured_module(model)

        def f(vertex: td.InnerVertex):
            if isinstance(vertex.module, torch.nn.SiLU):
                vertex.module = torch.nn.ReLU()

        first_silu_vertex = dag.inner_vertices[-1]
        second_silu_vertex = dag.inner_vertices[2].module.inner_vertices[1]
        dag.traverse(f)

        assert isinstance(first_silu_vertex.module, torch.nn.ReLU)
        assert isinstance(second_silu_vertex.module, torch.nn.ReLU)

        dag.save(tmpdir)
        loaded_dag = td.DagModule.load(tmpdir)

        first_silu_vertex = loaded_dag.inner_vertices[-1]
        second_silu_vertex = loaded_dag.inner_vertices[2].module.inner_vertices[1]

        assert isinstance(first_silu_vertex.module, torch.nn.ReLU)
        assert isinstance(second_silu_vertex.module, torch.nn.ReLU)
