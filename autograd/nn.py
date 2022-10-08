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

from typing import Any, List, Tuple, Union

from .engine import Value, ReLUOp as ReLU, TanhOp as Tanh

import random
import abc


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p._grad = 0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):
    def __init__(self, input_dim: int, bias=True, activation="relu") -> None:
        self._synapses = [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
        self._bias = Value(random.uniform(-1, 1)) if bias else None
        self._activation = "relu"

    def __call__(self, inputs: Union[Value, List]) -> Any:
        output = Value(0)
        if isinstance(inputs, Value):
            inputs = [inputs]
        assert isinstance(inputs, list) or isinstance(
            inputs, Value
        ), "Neuron inputs should be a Value or a list!"
        assert len(inputs) == len(
            self._synapses
        ), "Dim of inputs doesn't match dim of synapses!"
        for i, x in enumerate(inputs):
            output += self._synapses[i] * x
        if self._bias is not None:
            output += self._bias
        if self._activation == "relu":
            # TODO: Gradient for this? (done?)
            output = ReLU.act(output)
        elif self._activation == "tanh":
            output = Tanh.act(output)
        elif self._activation == "identity":
            pass
        else:
            raise NotImplementedError(
                f"Activation {self._activation} is not implemented"
            )
        return output

    def parameters(self) -> List[Value]:
        return self._synapses + ([self._bias] if self._bias is not None else [])


class Layer(Module):
    def __init__(self, input_dim: int, output_dim: int, activation="relu") -> None:
        self.neurons = [
            Neuron(input_dim, activation=activation) for _ in range(output_dim)
        ]

    def __call__(self, inputs: List[Neuron]) -> Any:
        outputs = []
        for n in self.neurons:
            outputs.append(n(inputs))
        return outputs

    def parameters(self) -> List[Value]:
        return [param for n in self.neurons for param in n.parameters()]


class NeuralNet(abc.ABC, Module):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # TODO: Implicitely convert inputs to Values?
        assert isinstance(args[0], Value), "Input must be a Value object!"
        return self.forward(*args)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> List[Value]:
        def find_params(obj: object) -> List[Value]:
            if isinstance(obj, Layer):
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

def MSE(x: Union[List[Value], Value], y: Union[List[Value], Value]):
    if isinstance(x, list) or isinstance(y, list):
        assert type(x) is type(
            y
        ), "If one argument is a list, both arguments must be a list!"
        assert isinstance(x, Value) and isinstance(
            y, Value
        ), "Arguments of MSE must be Value objects!"
        res = sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])
    else:
        assert isinstance(x, Value) and isinstance(
            y, Value
        ), "Arguments of MSE must be Value objects!"
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
            p._grad = 0
