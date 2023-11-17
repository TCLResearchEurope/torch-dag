import torch

import torch_dag as td
import torch_dag_algorithms


class MyStrangeActivation(torch.nn.Module):
    # definitely non-traceable (dynamic control flow)

    def __init__(self):
        super().__init__()
        self.act0 = torch.nn.GELU()
        self.act1 = torch.nn.ReLU6()

    def forward(self, x):
        B, C, H, W = x.shape
        if H == W:
            return self.act0(x)
        else:
            return self.act1(x)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.strange_act = MyStrangeActivation()
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dense0 = torch.nn.Linear(16, 8)
        self.dense1 = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.strange_act(self.conv1(x))
        x = self.conv2(x)
        x = self.pool(x).mean(dim=(2, 3))
        x = self.dense1(self.relu(self.dense0(x)))
        return x


def test_custom_untraceable_but_prunable_module():
    expected_number_of_orbits = 3
    custom_module_classes = (MyStrangeActivation,)

    model = Model()
    dag = td.build_from_unstructured_module(model, custom_autowrap_torch_module_classes=custom_module_classes)

    orbitalizer = torch_dag_algorithms.pruning.GeneralOrbitalizer(
        pruning_mode='default',
        block_size=None,
    )
    dag_orb, orbits = orbitalizer.orbitalize(
        dag=dag,
        vis_final_orbits=False,
        prune_stem=True,
        input_shape=(1, 3, 224, 224),
    )

    assert expected_number_of_orbits == len(orbits)

    # randomize channel importance (for dummy pruning)
    orbits_dict = torch_dag_algorithms.pruning.get_orbits_dict(dag_orb)
    for k, v in orbits_dict.items():
        num_channels = v.num_channels
        v.debug_logits = torch.normal(mean=torch.zeros(size=(num_channels,)))

    dag_final = torch_dag_algorithms.pruning.remove_channels_in_dag(dag_orb, input_shape=(1, 3, 224, 224))


def test_custom_untraceable_and_unprunable_module():
    expected_number_of_orbits = 2

    model = Model()
    dag = td.build_from_unstructured_module(model, custom_autowrap_torch_module_classes=(MyStrangeActivation,))

    orbitalizer = torch_dag_algorithms.pruning.GeneralOrbitalizer(
        pruning_mode='default',
        block_size=None,
        custom_unprunable_module_classes=(MyStrangeActivation,)
    )
    dag_orb, orbits = orbitalizer.orbitalize(
        dag=dag,
        vis_final_orbits=False,
        prune_stem=True,
        input_shape=(1, 3, 224, 224),
    )

    assert expected_number_of_orbits == len(orbits)

    # randomize channel importance (for dummy pruning)
    orbits_dict = torch_dag_algorithms.pruning.get_orbits_dict(dag_orb)
    for k, v in orbits_dict.items():
        num_channels = v.num_channels
        v.debug_logits = torch.normal(mean=torch.zeros(size=(num_channels,)))

    dag_final = torch_dag_algorithms.pruning.remove_channels_in_dag(dag_orb, input_shape=(1, 3, 224, 224))
