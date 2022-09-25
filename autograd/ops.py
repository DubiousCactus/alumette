#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Operations.
"""

import abc
from .engine import MaxOp, NoOp, Value


class F:
    @staticmethod 
    @abc.abstractmethod
    def ReLU(val) -> Value:
        # TODO: Analytical gradient? And somehow putting it into the graph.
        # This might reduce computation by removing unnecessary graph trasversal when we could just
        # populate the .grad of this node!
        v = Value(max(0, val.data), _parents=(Value(0), val), _op=NoOp)
        v._grad = val.data
        return v
