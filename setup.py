#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dynamics-learning",
    version="1.0.0",
    author="Bernardo Fichera",
    author_email="bernardo.fichera@gmail.com",
    description="Investigation of different strategies for Dynamical Systems learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nash169/dynamics-learning.git",
    packages=find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",                # math
        "matplotlib",           # plotting
        "torch",                # net framework
        "torchdiffeq",          # ode integration
    ],
    extras_require={
        "pytorch": [
            "torchvision",      # net framework GPU
        ],
        "dev": [
            "pylint",           # python linter
        ]
    },
)
