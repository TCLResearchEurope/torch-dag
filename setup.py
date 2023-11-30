"""
Setup for installability.
"""
import os

from pkg_resources import parse_version
from setuptools import find_packages
from setuptools import setup


def read_version(version_file="VERSION"):
    with open(version_file, "r") as f:
        return f.read()


def read_requirements(requirements_file="requirements.txt"):
    with open(requirements_file) as f:
        return f.read().splitlines()


setup(
    name="torch_dag",
    version=read_version(),
    author="TCL Research Europe",
    description="torch_dag",
    install_requires=read_requirements(),
    long_description=open("README.md").read(),
    packages=find_packages(exclude=("tests.*")),
    entry_points={"console_scripts": ["td_utils = torch_dag_algorithms.cli:cli"]},
    zip_safe=True,
)
