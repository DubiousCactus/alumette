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

import abc

from typing import Any, Tuple


"""
Learning notes:

- We're not using the numerical gradient estimation methods because it requires to recompute all
  operations up to the final output, which is a lot of ops. It is greatly reduced by using the
  chain rule and derivation rules!
"""

class Op(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def backward(node: Any) -> None:
        pass


class NoOp(Op):
    @staticmethod
    def backward(node: Any) -> None:
        pass


class Value:
    def __init__(
        self, data: float, _parents=(), _grad_fn=NoOp.backward, requires_grad=True
    ):
        self.data = data
        self._parents = _parents
        self._grad = 0.0
        self._grad_fn = _grad_fn
        self.requires_grad = requires_grad
        self._visited = False

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self._grad}, _grad_fn={self._grad_fn})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        self._visited = False
        return Value(
            self.data + other.data, _parents=(self, other), _grad_fn=AddOp.backward
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        self._visited = False
        return Value(
            self.data * other.data, _parents=(self, other), _grad_fn=MulOp.backward
        )

    def __pow__(self, other):
        assert type(other) in [
            int,
            float,
        ], "__pow__ only handles float or int exponents"
        self._visited = False
        return Value(
            self.data**other,
            _parents=(self, Value(other, requires_grad=False)),
            _grad_fn=PowOp.backward,
        )

    def __neg__(self):
        # Could to return self * -1 but that would result in one more function call...
        self._visited = False
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        self._visited = False
        return self + (-other)
        # return Value(-self.data, _grad_fn=NegOp.backward)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        assert other.data != 0, "Zero division encountered!"
        self._visited = False
        return self * other**-1
        # return Value(self.data / other.data, _parents=(self, other), _grad_fn=DivOp.backward)

    def __radd__(self, other):  # other + self
        self._visited = False
        return self + other

    def __rsub__(self, other):  # other - self
        self._visited = False
        return (-self) + other

    def __rmul__(self, other):
        self._visited = False
        return self * other

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
        # TODO: Write a test with a cyclic graph (or almost cyclic but you know, one that needs
        # topology sorting). This is suspiciously more efficient than backward_with_topo
        def _backward(node: Value):
            for p in node.parents:
                if p.requires_grad and not p._visited:
                    p._grad_fn(p)
                    p._visited = True
                    _backward(p)

        # import time
        # start = time.perf_counter_ns()
        self._grad = 1
        self._grad_fn(self) # Backprop to parents
        _backward(self)
        # end = time.perf_counter_ns()
        # print(f"Clock time of backward1: {(end-start)*(1e-6)}ms")

    def backward_with_topo(self):
        topology = []
        visited = set()
        def build_topology(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for p in node.parents:
                    if p.requires_grad:
                        build_topology(p)
                topology.append(node)

        self._grad = 1
        # import time
        # start = time.perf_counter_ns()
        build_topology(self)
        for node in reversed(topology):
            node._grad_fn(node)
        # end = time.perf_counter_ns()
        # print(f"Clock time of backward1: {(end-start)*(1e-6)}ms")


    @property
    def parents(self) -> Tuple:
        return self._parents

    @property
    def grad(self) -> float:
        return self._grad


class AddOp(Op):
    @staticmethod
    def backward(node: Value) -> None:
        parents = node.parents
        parents[0]._grad += node._grad
        parents[1]._grad += node._grad


class MulOp(Op):
    @staticmethod
    def backward(node: Value) -> None:
        parents = node.parents
        # assert (
        # node._grad != 0
        # ), "Output node has a 0 gradient while trying to backpropagate to parents!"
        parents[0]._grad += node._grad * parents[1].data
        parents[1]._grad += node._grad * parents[0].data


class NegOp(Op):
    @staticmethod
    def backward(node: Value) -> None:
        parents = node.parents
        assert len(parents) == 1, "NegOp has more than one parent!"
        parents[0]._grad += -node._grad


class PowOp(Op):
    @staticmethod
    def backward(node: Value) -> None:
        parents = node.parents
        assert (
            node._grad != 0
        ), "Output node has a 0 gradient while trying to backpropagate to parents!"
        # TODO: Handle pow(value, value)!
        parents[0]._grad += (
            parents[1].data * (parents[0].data ** (parents[1].data - 1)) * node._grad
        )
