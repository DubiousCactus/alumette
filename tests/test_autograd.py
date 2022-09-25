#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Tests !
"""

import unittest

from autograd import Value, grad

class TestAutoGrad(unittest.TestCase):
    def test_1(self):
        a, b, c = Value(2), Value(-3), Value(10)
        d = a*b + c
        self.assertEqual(a.grad, None)
        self.assertEqual(b.grad, None)
        self.assertEqual(c.grad, None)
        print()
        grad(d)
        self.assertAlmostEqual(c.grad, 1)
        self.assertAlmostEqual(a.grad, b.data)
        self.assertAlmostEqual(b.grad, a.data)

    def test_2(self):
        a, b, c = Value(-8), Value(-2), Value(4)
        d = (a+b) * c
        self.assertEqual(a.grad, None)
        self.assertEqual(b.grad, None)
        self.assertEqual(c.grad, None)
        print()
        grad(d)
        self.assertAlmostEqual(c.grad, a.data+b.data)
        self.assertAlmostEqual(a.grad, c.data)
        self.assertAlmostEqual(b.grad, c.data)

    def test_3(self):
        a, b, c, d = Value(2), Value(-3), Value(10), Value(-8)
        L = d*(c+(a*b))
        self.assertEqual(a.grad, None)
        self.assertEqual(b.grad, None)
        self.assertEqual(c.grad, None)
        self.assertEqual(d.grad, None)
        grad(L)
        self.assertAlmostEqual(d.grad,(a.data*b.data)+c.data)
        self.assertAlmostEqual(c.grad, d.data)
        self.assertAlmostEqual(b.grad, a.data*d.data)
        self.assertAlmostEqual(a.grad, b.data*d.data)

if __name__ == '__main__':
    unittest.main()
