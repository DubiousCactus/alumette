#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Neural Nets yaaay.
"""

from functools import reduce
from typing import Any, List, Tuple, Union

from .engine import Tensor
import alumette

import numpy as np
import random
import abc


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.array([0])

    def parameters(self) -> List[Tensor]:
        return []

class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int, activation="relu", use_bias=True) -> None:
        self.weights = Tensor(np.random.random((input_dim, output_dim)), requires_grad=True)
        self.bias = Tensor(np.random.random((input_dim)), requires_grad=True) if use_bias else None
        self._activation = activation

    def __call__(self, x: Tensor) -> Any:
        output = self.weights.T @ x
        if self.bias is not None:
            output = output + self.bias
        if self._activation == "relu":
            # TODO: Gradient for this? (done?)
            output = alumette.relu(output)
        elif self._activation == "tanh":
            output = alumette.tanh(output)
        elif self._activation == "identity":
            pass
        return output

    def parameters(self) -> List[Tensor]:
        return [self.weights] + ([self.bias] if self.bias is not None else [])


class NeuralNet(abc.ABC, Module):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        def find_params(obj: object) -> List[Tensor]:
            if isinstance(obj, Linear):
                return obj.parameters()
            params = []
            for _, attr in obj.__dict__.items():
                if type(attr) is list:
                    for e in attr:
                        params.extend(find_params(e))
                else:
                    params.extend(find_params(attr))
            return params

        return find_params(self)

class MLP(NeuralNet):
    def __init__(
        self,
        input_dim: int,
        layer_width: int,
        output_dim: int,
        n_layers: int,
        activation="relu",
        output_activation="identity",
    ) -> None:
        self.layers = [Linear(input_dim, layer_width, activation=activation)]
        for _ in range(n_layers - 2):
            self.layers += [Linear(layer_width, layer_width, activation=activation)]
        self.layers += [
            Linear(layer_width, output_dim, activation=output_activation),
        ]

    def forward(self, x):
        # TODO: return reduce(lambda layer_n, layer_n1: layer_n1(layer_n(x)), self.layers)
        y = x
        for l in self.layers:
            y = l(y)
        return y


def MSE(x: Union[List[Tensor], Tensor], y: Union[List[Tensor], Tensor]):
    if isinstance(x, list) or isinstance(y, list):
        assert type(x) is type(
            y
        ), "If one argument is a list, both arguments must be a list!"
        assert isinstance(x, Tensor) and isinstance(
            y, Tensor
        ), "Arguments of MSE must be Tensor objects!"
        res = sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])
    else:
        assert isinstance(x, Tensor) and isinstance(
            y, Tensor
        ), "Arguments of MSE must be Tensor objects!"
        res = (x - y) ** 2
    return res


class SGD:
    def __init__(self, parameters: List, lr=1e-3) -> None:
        self._params = parameters
        self._lr = lr

    def step(self):
        for p in self._params:
            assert p.grad is not None
            p.data -= self._lr * p.grad

    def zero_grad(self):
        for p in self._params:
            p.grad = 0
