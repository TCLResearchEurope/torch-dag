# Contribute to Torch-DAG

- [Contribute to Torch-DAG](#contribute-to-torch-dag)
  - [How to contribute](#how-to-contribute)
    - [Report an issue](#report-an-issue)
    - [Plan a contribution](#plan-a-contribution)
    - [Implement](#implement)
      - [Bugfix](#bugfix)
      - [Feature](#feature)
  - [Codebase structure](#codebase-structure)
    - [torch\_dag](#torch_dag)
    - [torch\_dag\_algorithms](#torch_dag_algorithms)
    - [torch\_dag\_timm\_plugin](#torch_dag_timm_plugin)
  - [Dependencies](#dependencies)
  - [Code conventions](#code-conventions)

## How to contribute

### Report an issue

If you're running into a problem with this repository, report it by opening an issue.  
Please include:

- the error message
- steps to reproduce
- description of your environment/platform

### Plan a contribution

Find or open an issue describing a problem/new feature. Outline your solution idea, preferably in the form of a design doc. For larger scope features, we may require a more detailed description. Gather our comments and best wait for some form of acceptance (e.g. a TODO label) before implementing.

### Implement

#### Bugfix

1. Make sure an issue for this bug exists
2. Fork the main branch of this repository
3. Write a test for the bug. It must fail initially
4. Fix the issue. Your test must now pass
5. Publish the changes and submit a Pull Request to the main repository. In the PR's description, link the issue using "Closes: #\<number\>"

It may not always be possible to write the aforementioned test. If you have another automated way of testing for this bug, you may also include it in the PR. Otherwise, consider describing the issue, symptoms, and solution.

#### Feature

1. Make sure an issue for this feature exists and contains an approved solution outline
2. Fork the `main` branch of this repository
3. Implement your feature. You should include some tests for it. They are especially important for us to ensure your feature doesn't stop working with other future changes
4. If your change requires additional libraries, make sure to add them as described in [Dependencies](#dependencies)
5. Create a PR to the main repository. In the PR's description, link the issue using "Closes: #\<number\>". A reviewer will be assigned
6. Iterate through your solution with the reviewer until your PR is merged.

## Codebase structure

### torch_dag

Core of the library, contains the implementation of the DAG network representation format, conversion methods, and basic functionality.

### torch_dag_algorithms

Contains implementations of algorithms running on the DAG structure.

### torch_dag_timm_plugin

Contains methods needed to convert various models from Timm into TorchDAG.

## Dependencies

1. Packages required for the library to work properly must be placed in `requirements.txt`
2. Packages required for developing the project, such as `black` or `flake8`, must be placed in `requirements-dev.txt`
3. Packages required for tutorials and demos only should be placed in `requirements-dev.txt`
4. Avoid requiring a specific package version.

## Code conventions

Read and follow the [Google Python styleguide](https://google.github.io/styleguide/pyguide.html).  

- Follow [`pep8`](https://www.python.org/dev/peps/pep-0008)
- Use [`black`](https://github.com/ambv/black) for code
formatting  
- Use [`flake8`](http://flake8.pycqa.org/en/latest/) for linting

All Python code must be written **compatible with Python 3.8+**.

Document your code using [Google style Python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)  
In short:

```python3
def fun(param1: T1, param2: T2) -> T3:
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value.

    """
    ...
```
