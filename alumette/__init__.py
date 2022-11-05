#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Karpathy's autograd replica.
"""

from .tensor import Tensor
from .ops import (
    mean,
    relu,
    tanh,
    softplus,
    exp,
    log,
)

from typing import Callable, List

import numpy as np


def grad_check(inputs: List[Tensor], exp: Callable, wrt: int, grad: np.ndarray | float):
    """
    Use numerical gradient computation to compute the gradients of an entire arbitrary expression
    with respect to one of its inputs.
    """
    const = np.array(1e-9, dtype=np.float128)
    raw_inputs = [x.numpy().copy().astype(np.float128) for x in inputs]
    raw_output = exp(*raw_inputs)
    if isinstance(raw_output, Tensor):
        raw_output = raw_output.numpy()
    num_grad = np.zeros_like(raw_inputs[wrt], dtype=np.float128)
    for i in np.ndindex(raw_inputs[wrt].shape):
        og = raw_inputs[wrt][i]
        raw_inputs[wrt][i] += const
        delta_output = exp(*raw_inputs)
        if isinstance(delta_output, Tensor):
            delta_output = delta_output.numpy()
        num_grad[i] = (delta_output - raw_output) / const
        raw_inputs[wrt][i] = og
    assert np.allclose(
        num_grad, grad, rtol=1e-4
    ), f"Numerical grad: {num_grad} -- Given grad: {grad}"
    return True
