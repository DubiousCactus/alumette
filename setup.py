#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from setuptools import find_packages, setup

setup(
    name="alumette",
    packages=find_packages(include=["alumette", "alumette.nn"]),
    version="0.1.0",
    description="Mini torch-like reverse-mode automatic differentiation and its tiny neural networks library",
    author="Théo Morales",
    license="MIT",
    install_requires=["tqdm", "numpy"],
)
