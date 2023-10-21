#!/usr/bin/env python

from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="emg-regression",
    version="1.0.0",
    author="Carolina Correia",
    author_email="cgprcorreia@gmail.com",
    description="EMG Regression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nash169/emg-regression.git",
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
    ],
    extras_require={
        "pytorch": [
            "torchvision",      # net framework GPU
            "tensorboard"       # visualizations
        ],
        "dev": [
            "pylint",           # python linter
        ]
    },
    # package_data={
    #     "learn_embedding.data.lasahandwriting": ["*.mat", "*.csv", "*.pkl"],
    #     "learn_embedding.data.roboticdemos": ["*.mat", "*.csv", "*.pkl"],
    # }
)
