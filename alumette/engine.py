#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Autograd engine
"""

import numpy as np
import abc

from typing import Any, List, Tuple


"""
Learning notes:

- We're not using the numerical gradient estimation methods because it requires to recompute all
  operations up to the final output, which is a lot of ops. It is greatly reduced by using the
  chain rule and derivation rules!
"""

DEFAULT_DTYPE=np.float32

class Op(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def backward(node: Any) -> None:
        pass


class NoOp(Op):
    @staticmethod
    def backward(node: Any) -> None:
        pass


def make_tensor_data(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(DEFAULT_DTYPE)
    elif isinstance(x, list):
        return np.array(x, dtype=DEFAULT_DTYPE)
    elif isinstance(x, float) or isinstance(x, int):
        return np.array([x], dtype=DEFAULT_DTYPE)
    else:
        raise TypeError(f"Tensor class only accepts np.ndarray and compatible data, not {type(x)}")

class Tensor:
    def __init__(
        self,
        data: np.ndarray | float | int | List,
        _parents=(),
        _grad_fn=NoOp.backward,
        requires_grad=True,
    ):
        self.data = make_tensor_data(data)
        self._parents = _parents
        self._grad = 0
        self._grad_fn = _grad_fn
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self._grad}, _grad_fn={self._grad_fn})"

    def __add__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return Tensor(
            self.data + other.data, _parents=(self, other), _grad_fn=AddOp.backward
        )

    def __mul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return Tensor(
            self.data * other.data, _parents=(self, other), _grad_fn=MulOp.backward
        )

    def __pow__(self, other):
        assert type(other) in [
            int,
        ], "__pow__ only handles int exponents"
        return Tensor(
            self.data**other,
            _parents=(self, Tensor(other, requires_grad=False)),
            _grad_fn=PowOp.backward,
        )

    def __neg__(self):
        # Could to return self * -1 but that would result in one more function call...
        return self * -1

    def __sub__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return self + (-other)
        # return Tensor(-self.data, _grad_fn=NegOp.backward)

    def __truediv__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        # assert other.data != 0, "Zero division encountered!"
        return self * other**-1
        # return Tensor(self.data / other.data, _parents=(self, other), _grad_fn=DivOp.backward)

    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other): #other/self
        return self ** -1 * other

#     def __eq__(self, __o: object) -> bool:
        # if type(__o) == float:
            # return self.data == __o
        # elif isinstance(__o, Value):
            # return self.data == __o.data
        # else:
            # raise NotImplementedError(
                # f"Comparing Value with type {type(__o)} is not implemented"
            # )

    # def __hash__(self) -> int:
    #         return hash(self.data) + sum([hash(o) for o in self.parents])

    def backward(self):
        topology = []
        visited = set()

        def build_topology(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for p in node.parents:
                    if p.requires_grad:
                        build_topology(p)
                topology.append(node)

        self._grad = 1.0
        build_topology(self)
        for node in reversed(topology):
            node._grad_fn(node)

    @property
    def parents(self) -> Tuple:
        return self._parents

    @property
    def grad(self):
        return self._grad

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def dtype(self) -> Tuple:
        return self.data.dtype


class AddOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        parents[0]._grad += node._grad
        parents[1]._grad += node._grad


class MulOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        # assert (
        # node._grad != 0
        # ), "Output node has a 0 gradient while trying to backpropagate to parents!"
        parents[0]._grad += node._grad * parents[1].data
        parents[1]._grad += node._grad * parents[0].data


class NegOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert len(parents) == 1, "NegOp has more than one parent!"
        parents[0]._grad += -node._grad


class PowOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert (
            node._grad != 0
        ), "Output node has a 0 gradient while trying to backpropagate to parents!"
        # TODO: Handle pow(value, value)!
        parents[0]._grad += (
            parents[1].data * (parents[0].data ** (parents[1].data- 1)) * node._grad
        )
