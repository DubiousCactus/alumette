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
import sys
import abc

from typing import Any, List, Tuple


"""
Learning notes:

- We're not using the numerical gradient estimation methods because it requires to recompute all
  operations up to the final output, which is a lot of ops. It is greatly reduced by using the
  chain rule and derivation rules!
"""

DEFAULT_DTYPE = np.float32
sys.setrecursionlimit(1500)# TODO: Memoization


class Op(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def backward(node: Any) -> None:
        pass


class NoOp(Op):
    @staticmethod
    def backward(node: Any) -> None:
        for p in node.parents:
            p.grad = p.grad + node.grad


def make_tensor_data(x: Any, dtype=None) -> np.ndarray:
    type_ = DEFAULT_DTYPE if dtype is None else dtype
    if isinstance(x, np.ndarray):
        type_ = x.dtype if dtype is None else dtype
        return x.astype(type_)
    elif isinstance(x, list):
        return np.array(x, dtype=type_)
    elif (
        isinstance(x, float)
        or isinstance(x, int)
    ):
        return np.array(x, dtype=type_)
    elif type(x) in [np.float32, np.float16, np.float64, np.float128, np.int32, np.int64]:
        return np.array(x, dtype=type(x))
    elif isinstance(x, Tensor):
        raise NotImplementedError("Tensor creation from other Tensor is not yet supported")
    else:
        raise TypeError(
            f"Tensor class only accepts np.ndarray and compatible data, not {type(x)}"
        )


class Tensor:
    def __init__(
        self,
        data: np.ndarray | float | int | List,
        _parents=(),
        _grad_fn=NoOp.backward,
        requires_grad=True,
        dtype=None,
    ):
        self.data = make_tensor_data(data, dtype=dtype)
        self._parents = _parents
        self.grad = np.array([0.0])
        self._grad_fn = _grad_fn
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad}, _grad_fn={self._grad_fn})"

    def __getitem__(self, idx: int | Tuple[int]) -> DEFAULT_DTYPE:
        return self.data[idx]

    def __add__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return Tensor(
            self.data + other.data, _parents=(self, other), _grad_fn=AddOp.backward
        )

    def __mul__(self, other):
        """
        Hadamard product: A*B
        """
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return Tensor(
            self.data * other.data, _parents=(self, other), _grad_fn=MulOp.backward
        )

    def __matmul__(self, other):  # self @ other
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return Tensor(
            self.data @ other.data, _parents=(self, other), _grad_fn=MatMulOp.backward
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
        return self * -1

    def __sub__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return self + (-other)

    def __truediv__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        return self * other**-1

    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):  # other/self
        return self**-1 * other

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

    def backward(self) -> None:
        topology = []
        visited = set()

        def build_topology(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for p in node.parents:
                    if p.requires_grad:
                        build_topology(p)
                topology.append(node)

        # assert (
            # self.data.squeeze().shape == ()
        # ), "grad can be implicitly created only for scalar outputs"
        self.grad = np.ones_like(self.data)
        build_topology(self)
        for node in reversed(topology):
            node._grad_fn(node)

    @property
    def parents(self) -> Tuple:
        return self._parents

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def dtype(self) -> Tuple:
        return self.data.dtype

    def item(self) -> float:
        assert len(self.data.shape) == 1
        return self.data[0]

    @property
    def T(self):
        return Tensor(self.data.T, _parents=(self,))

    def numpy(self) -> np.ndarray:
        return self.data

    def __getattr__(self, name):
        try:
            return getattr(self.data, name)
        except AttributeError:
            raise AttributeError(f"'alumette.Tensor' object has no attribute '{name}'")


class AddOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        parents[0].grad = parents[0].grad + node.grad
        parents[1].grad = parents[1].grad + node.grad

class MulOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        parents[0].grad = parents[0].grad + node.grad * parents[1].data
        parents[1].grad = parents[1].grad + node.grad * parents[0].data


class MatMulOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        # TODO: Write more unit test?
        parents = node.parents
        if len(parents[0].shape) > 1 and len(parents[1].shape) == 1:
            # Matrix-vector product: Ab
            parents[0].grad = (
                parents[0].grad
                + node.grad * (np.ones_like(parents[0].data) * parents[1].data).T
            )
            parents[1].grad = parents[1].grad + node.grad @ parents[0].data
        elif len(parents[0].shape) == 1 and len(parents[1].shape) > 1:
            # Matrix-vector product: Ba
            parents[1].grad = (
                parents[1].grad
                + node.grad * (np.ones_like(parents[1].data) * parents[0].data).T
            )
            parents[0].grad = parents[0].grad + node.grad @ parents[1].data
        elif parents[0].shape == parents[1].shape:
            # Matrix-matrix product or vector-vector product: the easy way out
            parents[0].grad = parents[0].grad + node.grad * parents[1].data.T
            parents[1].grad = parents[1].grad + node.grad * parents[0].data.T
        else:
            # Work out the dimensions.
            # TODO: Find a more efficient way to do this!
            if node.grad.shape[-1] == parents[1].data.shape[0]:
                parents[0].grad = parents[0].grad + node.grad @ parents[1].data
            else:
                parents[0].grad = parents[0].grad + node.grad @ parents[1].data.T
            if node.grad.shape[-1] == parents[0].data.shape[0]:
                parents[1].grad = parents[1].grad + node.grad @ parents[0].data
            else:
                parents[1].grad = parents[1].grad + node.grad @ parents[0].data.T


class NegOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert len(parents) == 1, "NegOp has more than one parent!"
        parents[0].grad += -node.grad


class PowOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        # TODO: Handle pow(value, value)!
        parents[0].grad += (
            parents[1].data * (parents[0].data ** (parents[1].data - 1)) * node.grad
        )
