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

from typing import Callable, List

import numpy as np

from .ops import exp, log, mean, relu, softplus, tanh
from .tensor import Tensor
from .utils import allclose


def grad_check(inputs: List[Tensor], exp: Callable, wrt: int, grad: np.ndarray):
    """
    Use numerical gradient computation to compute the gradients of an entire arbitrary expression
    with respect to one of its inputs.
    """
    const = np.array(1e-8, dtype=np.float128)
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
        num_grad, grad.astype(np.float128), rtol=1e-3
    ), f"Numerical grad: {num_grad} -- Given grad: {grad}"
    return True
