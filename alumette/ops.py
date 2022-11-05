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

from .tensor import Tensor

import numpy as np
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
        parents[0].grad += 1 / node.data * node.grad

    @staticmethod
    def act(node: Tensor) -> Tensor:
        return Tensor(np.log(node.data), _parents=(node,), _grad_fn=LogOp.backward)


class ExpOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        # TODO: Write unit test
        parents = node.parents
        assert len(parents) == 1, "ExpOp has more than one parent!"
        parents[0].grad += np.exp(node.data) * node.grad

    @staticmethod
    def act(node: Tensor) -> Tensor:
        return Tensor(np.exp(node.data), _parents=(node,), _grad_fn=ExpOp.backward)


class SoftPlusOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        # TODO: Write unit test
        parents = node.parents
        assert len(parents) == 1, "SoftPLusOp has more than one parent!"
        parents[0].grad += 1 / (1 + np.exp(-node.data)) * node.grad

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


def mean(values: List[Tensor] | Tensor) -> Tensor:
    # TODO: Write an op to reduce number of nodes
    # TODO: Write a test for this
    if isinstance(values, list):
        values = values if isinstance(values, list) else [values]
        return reduce(lambda a, b: a + b, values) / len(values)
    # elif isinstance(values, Tensor):
    # assert len(values.shape) == 1 # TODO: Handle matrices
    # comps = [values[i] for i in range(values.shape[0])]
    #     return reduce(lambda a, b: a + b, comps) / len(comps)


class ReLUOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert len(parents) == 1, "ReLUOp has more than one parent!"
        assert np.all(node.data >= 0), "ReLU's output node has negative value"
        parents[0].grad = (
            parents[0].grad
            + np.where(
                node.data > 0,
                np.ones_like(parents[0].grad),
                np.zeros_like(parents[0].grad),
            )
            * node.grad
        )

    @staticmethod
    def act(node: Tensor) -> Tensor:
        return Tensor(
            np.maximum(0, node.data), _parents=(node,), _grad_fn=ReLUOp.backward
        )


class TanhOp(Op):
    @staticmethod
    def backward(node: Tensor) -> None:
        parents = node.parents
        assert len(parents) == 1, "TanhOp has more than one parent!"
        parents[0].grad = parents[0].grad + (np.ones_like(node.data) - node.data**2) * node.grad

    @staticmethod
    def act(node: Tensor) -> Tensor:
        n = node.data
        t = (np.exp(np.clip(2 * n, -708, 709)) - 1) / (
            np.exp(np.clip(2 * n, -708, 709)) + 1
        )
        return Tensor(t, _parents=(node,), _grad_fn=TanhOp.backward)


softplus = SoftPlusOp.act
relu = ReLUOp.act
tanh = TanhOp.act
exp = ExpOp.act
log = LogOp.act

