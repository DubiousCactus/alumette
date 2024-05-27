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
from alumette.tensor import DEFAULT_DTYPE


class TensorOpsTests(unittest.TestCase):
    def test_create_from_float(self):
        val = random.uniform(-100, 100)
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertEqual(a.data.dtype, DEFAULT_DTYPE)
        self.assertAlmostEqual(a.data, val, places=5)
        a = Tensor(val, dtype=np.intc)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertEqual(a.data.dtype, np.intc)
        self.assertTrue(a.data == int(val))

    def test_create_from_int(self):
        val = random.randint(-100, 100)
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertEqual(a.data.dtype, DEFAULT_DTYPE)
        self.assertTrue(a.data == float(val))
        a = Tensor(val, dtype=np.intc)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertEqual(a.data.dtype, np.intc)
        self.assertTrue(a.data == val)

    def test_create_from_list(self):
        val = [random.uniform(-100, 100) for _ in range(random.randint(1, 10))]
        a = Tensor(val)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertEqual(a.data.dtype, DEFAULT_DTYPE)
        self.assertTrue(np.allclose(a.data, np.array(val)))
        a = Tensor(val, dtype=np.intc)
        self.assertTrue(type(a.data) is np.ndarray)
        self.assertTrue(np.allclose(a.data, np.array(val).astype(np.intc)))
        self.assertEqual(a.data.dtype, np.intc)

    def test_create_from_numpy(self):
        val = np.random.random((random.randint(1, 10), random.randint(1, 10)))
        a = Tensor(val)
        self.assertTrue(type(a.data), np.ndarray)
        self.assertTrue(np.allclose(a.data, val))
        self.assertEqual(a.data.dtype, val.dtype)
        a = Tensor(val, dtype=np.intc)
        self.assertTrue(type(a.data), np.ndarray)
        self.assertTrue(np.allclose(a.data, val.astype(np.intc)))
        self.assertEqual(a.data.dtype, np.intc)

    def test_itemize(self):
        array = np.random.random((random.randint(2, 10), random.randint(2, 10)))
        t = Tensor(array)
        throws = False
        try:
            _ = t.item()
        except Exception:
            throws = True
        finally:
            self.assertTrue(throws)
        array = np.random.random((random.randint(2, 10), 1))
        t = Tensor(array)
        self.assertEqual(t.squeeze().shape, (array.shape[0],))
        val = random.random()
        t = Tensor([[[val]]])
        self.assertAlmostEqual(t.item(), val)
        val = random.random()
        t = Tensor([val])
        self.assertAlmostEqual(t.item(), val)
        val = random.random()
        t = Tensor(val)
        self.assertAlmostEqual(t.item(), val)


if __name__ == "__main__":
    unittest.main()
