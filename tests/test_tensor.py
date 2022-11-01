#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Tensor creation and manipulation.
"""

import random
import unittest

import numpy as np

from alumette import Tensor
from alumette.tensor import DEFAULT_DTYPE, set_default_dtype


class TensorOpsTests(unittest.TestCase):
    def test_create_from_float(self):
        set_default_dtype(np.float32)
        val = random.uniform(-100, 100)
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertTrue(a.data.dtype == np.float32)
        self.assertTrue(a.data == val)
        set_default_dtype(np.intc)
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertTrue(a.data.dtype == np.intc)
        self.assertTrue(a.data == int(val))

    def test_create_from_int(self):
        set_default_dtype(np.float32)
        val = random.randint(-100, 100)
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertTrue(a.data.dtype == np.float32)
        self.assertTrue(a.data == float(val))
        set_default_dtype(np.intc)
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertTrue(a.data.dtype == np.intc)
        self.assertTrue(a.data == val)

    def test_create_from_list(self):
        set_default_dtype(np.float32)
        val = [random.uniform(-100, 100) for _ in range(random.randint(1, 10))]
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertTrue(a.data.dtype == np.float32)
        self.assertTrue(np.allclose(a.data, np.array(val)))
        set_default_dtype(np.intc)
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertTrue(np.allclose(a.data, np.array(val).astype(np.intc)))
        self.assertTrue(a.data.dtype == np.intc)

    def test_create_from_numpy(self):
        set_default_dtype(np.float32)
        val = np.random.random((random.randint(1,10), random.randint(1,10)))
        a = Tensor(val)
        self.assertTrue(type(a.data), np.ndarray)
        self.assertTrue(np.allclose(a.data, val))
        self.assertTrue(a.data.dtype == np.float32)
        set_default_dtype(np.intc)
        a = Tensor(val)
        self.assertTrue(type(a.data), np.ndarray)
        self.assertTrue(np.allclose(a.data, val.astype(np.intc)))
        self.assertTrue(a.data.dtype == np.intc)

if __name__ == "__main__":
    unittest.main()
