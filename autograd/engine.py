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

from typing import Any, Optional, Set


def NoOp(a: Any, b: Any) -> float:
    return 1

def MaxOp(a: Any, b: Any) -> float:
    # TODO: But what am I doing?
    return max(a, b)

class Value:
    def __init__(self, data: float, _parents=(), _op=NoOp):
        self.data = data
        self._parents = set(_parents)
        self._op = _op
        self._grad = None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self._grad})"

    def __add__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        return Value(self.data + b.data, _parents=(self, b), _op=float.__add__)

    def __sub__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        return Value(self.data - b.data, _parents=(self, b), _op=float.__sub__)

    def __pow__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        return Value(self.data ** b.data, _parents=(self, b), _op=float.__pow__)

    def __neg__(self):
        v = Value(-self.data, _parents=self._parents, _op=self._op)
        v._grad = self._grad
        return v

    def __mul__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        return Value(self.data * b.data, _parents=(self, b), _op=float.__mul__)

    def __truediv__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        assert b.data != 0, "Zero division encountered!"
        return Value(self.data / b.data, _parents=(self, b), _op=float.__truediv__)

    def __eq__(self, __o: object) -> bool:
        if type(__o) == float:
            return self.data == __o
        elif type(__o) == Value:
            return self.data == __o.data
        else:
            raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.data) + sum([hash(o) for o in self.parents])

    # def __float__(self) -> float:
        # return self.data

    def backward(self):
        grad(self)

    @property
    def parents(self) -> Set:
        return self._parents

    @property
    def op(self):
        return self._op

    @property
    def grad(self) -> Optional[float]:
        return self._grad

    @property
    def safe_grad(self) -> float:
        return self._grad if self._grad is not None else 1.0


def grad(output: Value):
    delta = 1e-5
    for p in output.parents:
        # TODO: How do I handle other ops now? Like ReLU?...
        # Like this maybe: (see the rest in the ReLU implementation)
        if p._grad is None:
            # TODO: Handle all the ops!
            output_delta = 1.0 if output.op in [float.__pow__, float.__mul__, float.__truediv__] else .0
            for p2 in output.parents:
                _dat = p2.data
                if p == p2:
                    _dat += delta
                output_delta = output.op(output_delta, _dat)
            p._grad = (output_delta-output.data)/delta * output.safe_grad
        grad(p)

