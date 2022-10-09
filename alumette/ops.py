#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Diverse ops
"""

from functools import reduce
from typing import List, Any

from numpy import require

from .engine import Tensor

import math
import abc


class Op(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def backward(node: Any) -> None:
        pass


class LogOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert len(parents) == 1, "LogOp has more than one parent!"
        # TODO: Write unit test
        parents[0]._grad += 1/node.data * node.grad

    @staticmethod
    def act(node: Tensor) -> Tensor:
        return Tensor(math.log(node.data), _parents=(node,), _grad_fn=LogOp.backward)


class ExpOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        # TODO: Write unit test
        parents = node.parents
        assert len(parents) == 1, "ExpOp has more than one parent!"
        parents[0]._grad += math.exp(node.data) * node.grad


    @staticmethod
    def act(node: Tensor) -> Tensor:
        return Tensor(math.exp(node.data), _parents=(node,), _grad_fn=ExpOp.backward)


class SoftPlusOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        # TODO: Write unit test
        parents = node.parents
        assert len(parents) == 1, "SoftPLusOp has more than one parent!"
        parents[0]._grad += 1/(1+math.exp(-node.data)) * node.grad

    @staticmethod
    def act(node: Tensor) -> Tensor:
        """
        SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the
        output of a machine to always be positive.
        """
        exp = ExpOp.act(Tensor(node.data, requires_grad=False))
        exp.requires_grad = False
        log = LogOp.act(1 + exp)
        log.requires_grad = False
        return Tensor(
            log.data,
            _parents=(node,),
            _grad_fn=SoftPlusOp.backward,
        )


def mean(values: List[Tensor]) -> Tensor:
    # TODO: Write an op to reduce number of nodes
    # TODO: Write a test for this
    values = values if isinstance(values, list) else [values]
    return reduce(lambda a, b: a + b, values) / len(values)


class ReLUOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert len(parents) == 1, "ReLUOp has more than one parent!"
        assert node.data >= 0, "ReLU's output node has negative value"
        parents[0]._grad += (1 if node.data > 0 else 0) * node._grad

    @staticmethod
    def act(node: Tensor) -> Tensor:
        return Tensor(max(0, node.data), _parents=(node,), _grad_fn=ReLUOp.backward)


class TanhOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert len(parents) == 1, "TanhOp has more than one parent!"
        parents[0]._grad += (1 - node.data**2) * node._grad

    @staticmethod
    def act(node: Tensor) -> Tensor:
        n = node.data
        t = (math.exp(max(min(2 * n, 709), -708)) - 1) / (
            math.exp(max(min(2 * n, 709), -708)) + 1
        )
        return Tensor(t, _parents=(node,), _grad_fn=TanhOp.backward)
