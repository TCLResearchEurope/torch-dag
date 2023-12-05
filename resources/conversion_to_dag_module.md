# How to convert `torch.nn.Module` instances to `DagModule`?

> NOTE: in these notes `model` will refer to a `torch.nn.Module` instance that
> you want to convert to `DagModule` and run model compression algorithms.

## 1. Basics

Given a `torch` `model` one converts it to a `DagModule` by running:

```python
import torch_dag as td
dag = td.build_from_unstructured_module(model)
```

We made sure it works for plenty of `timm` models (see [coverage table]).

## 2. Why conversion can fail. 

There are a number of reasons why your `model` cannot be converted to a `DagModule` 
staright away:

A. the `model` cannot be traced by `torch.FX`

B. tracing works but there is no support for converting some `torch.FX` substructures to
`DagModule`

How can you avoid `A`? See official [documentation](https://pytorch.org/docs/stable/fx.html#dynamic-control-flow). In essence, 
there are cases when one can easile change the code in the `forward` method of the `model` to make it more
tracing-friendly. Handling `B` is trickier. We suggest you create an issue in the `torch-dag` repository and let
us handle that!

## 3. How to get feedback about why conversion fails?

There is a useful tool to see which modules in a given `model` can be converted to a `DagModule`. Running

```python
import logging
logging.basicConfig(level=logging.INFO)
import torch_dag as td
dagable_classes, undagable_classes = td.commons.look_for_dagable_modules(model)
```

will produce some useful information and return two sets:
`dagable_classes` and  `undagable_classes`. You can start debugging here. The classes of modules that cannot be converted to
`DagModule` will be in `undagable_classes`.

## 4. What to do when a `model` contains a module that cannot be traced?

There is an option to manually set some module classes to be left alone when converting to `DagModule`. Here is an example of how it can be used. Assume that in your `model` you have a module like that:

```python
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
```

`MyStrangeActivation` cannot be traced by `torch.FX` because of dynamic control flow. Assume that your model is defined by the following code snippet:

```python
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
```

You can still convert `Model` to `DagModule`, provided you pass on additional info to the converter:

```python
import torch_dag as td

model = Model()
custom_module_classes = (MyStrangeActivation, )
dag = td.build_from_unstructured_module(model, custom_autowrap_torch_module_classes=custom_module_classes)
```

> **NOTE:** If you build and save a `model` with `custom_module_classes`, you need to pass the custom module classes, when loading the model after saving:
 ```python
import torch_dag as td
loaded_model  = td.io.load_dag_from_path(
    path=model_path,
    custom_module_classes=(MyStrangeActivation, )
)
```




