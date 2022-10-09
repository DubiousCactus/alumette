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

from .engine import Tensor
from .ops import (
    mean,
    ReLUOp as relu,
    TanhOp as tanh,
    SoftPlusOp as softplus,
    ExpOp as exp,
    LogOp as log,
)
