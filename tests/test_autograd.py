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
import random

from autograd import Value
import autograd


class TestAutoGrad(unittest.TestCase):
    def test_int_linear(self):
        a, b, c = Value(random.randint(-100, 100)), Value(random.randint(-100, 100)), Value(random.randint(-100, 100))
        d = a * b + c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, b.data)
        self.assertEqual(b.grad, a.data)

    def test_float_linear(self):
        a, b, c = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))
        d = a * b + c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, b.data)
        self.assertEqual(b.grad, a.data)

    def test_int_linear_2(self):
        a, b, c = Value(random.randint(-100, 100)), Value(random.randint(-100, 100)), Value(random.randint(-100, 100))
        d = (a + b) * c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, a.data + b.data)
        self.assertEqual(a.grad, c.data)
        self.assertEqual(b.grad, c.data)

    def test_float_linear_2(self):
        a, b, c = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))
        d = (a + b) * c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, a.data + b.data)
        self.assertEqual(a.grad, c.data)
        self.assertEqual(b.grad, c.data)

    def test_int_chain(self):
        a, b, c, d = Value(random.randint(-100, 100)), Value(random.randint(-100, 100)), Value(random.randint(-100, 100)), Value(random.randint(-100, 100))
        L = d * (c + (a * b))
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        self.assertEqual(d.grad, 0)
        L.backward()
        self.assertEqual(d.grad, (a.data * b.data) + c.data)
        self.assertEqual(c.grad, d.data)
        self.assertEqual(b.grad, a.data * d.data)
        self.assertEqual(a.grad, b.data * d.data)

    def test_float_chain(self):
        a, b, c, d = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))
        L = d * (c + (a * b))
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        self.assertEqual(d.grad, 0)
        L.backward()
        self.assertEqual(d.grad, (a.data * b.data) + c.data)
        self.assertEqual(c.grad, d.data)
        self.assertEqual(b.grad, a.data * d.data)
        self.assertEqual(a.grad, b.data * d.data)

    def test_int_squared_chain(self):
        a, b, c, d = Value(random.randint(-100, 100)), Value(random.randint(-100, 100)), Value(random.randint(-100, 100)), Value(random.randint(-100, 100))
        exp, denom = 3, 8
        L = a*((b-c)**3) + (d/8)
        L.backward()
        self.assertEqual(a.grad, (b.data-c.data)**exp)
        self.assertEqual(b.grad, exp*a.data*(b.data-c.data)**(exp-1))
        self.assertEqual(c.grad, -exp*a.data*(b.data-c.data)**(exp-1))
        self.assertEqual(d.grad, 1/denom)

    def test_float_squared_chain(self):
        a, b, c, d = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))
        exp, denom = 3, 8
        L = a*((b-c)**3) + (d/8)
        L.backward()
        self.assertAlmostEqual(a.grad, (b.data-c.data)**exp)
        self.assertAlmostEqual(b.grad, exp*a.data*(b.data-c.data)**(exp-1))
        self.assertAlmostEqual(c.grad, -exp*a.data*(b.data-c.data)**(exp-1))
        self.assertAlmostEqual(d.grad, 1/denom)

    def test_r_add(self):
        a = Value(random.uniform(-100, 100))
        b= random.uniform(-100, 100)
        L = b + a
        L.backward()
        self.assertEqual(a.grad, 1)

    def test_r_sub(self):
        a = Value(random.uniform(-100, 100))
        b= random.uniform(-100, 100)
        L = b - a
        L.backward()
        self.assertEqual(a.grad, -1)

    def test_r_mul(self):
        a = Value(random.uniform(-100, 100))
        b= random.uniform(-100, 100)
        L = b * a
        L.backward()
        self.assertEqual(a.grad, b)

    def test_neg_r_mul(self):
        a = Value(random.uniform(-100, 100))
        b= random.uniform(-100, 100)
        L = (-b) * a
        L.backward()
        self.assertEqual(a.grad, -b)

    def test_positive_pow(self):
        a = Value(random.uniform(-100, 100))
        b = random.uniform(0, 100)
        (a**b).backward()
        self.assertEqual(a.grad, b*(a.data**(b-1)))

    def test_negative_pow(self):
        a = Value(random.uniform(-100, 100))
        b = random.uniform(-1000, -1)
        (a**b).backward()
        self.assertEqual(a.grad, b*(a.data**(b-1)))

    def test_zero_pow(self):
        a = Value(random.uniform(-100, 100))
        b = 0
        (a**b).backward()
        self.assertEqual(a.grad, 0)

    def test_div(self):
        a, b = Value(random.uniform(-10, 10)), Value(random.uniform(-10, 10))
        L = a / b
        L.backward()
        self.assertAlmostEqual(a.grad, 1/b.data, places=4)
        self.assertAlmostEqual(b.grad, -a.data/(b.data**2), places=4)

    def test_large_div(self):
        a, b = Value(random.uniform(-1000, 1000)), Value(random.uniform(-1000, 1000))
        L = a / b
        L.backward()
        self.assertAlmostEqual(a.grad, 1/b.data, places=4)
        self.assertAlmostEqual(b.grad, -a.data/(b.data**2), places=4)

    def test_r_ops(self):
        a, b, c = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100)), Value(random.uniform(-10, 10))
        d, e, f = random.uniform(-100, 100),random.uniform(-100, 100),random.uniform(-10, 10)
        L = (-d * (e - a)) + ((f + b) / (c**f))
        L.backward()
        self.assertEqual(a.grad, d)
        self.assertEqual(b.grad, 1/(c.data**f))
        self.assertAlmostEqual(c.grad, -f*(c.data**(-f-1))*(f+b.data), places=4)

    def test_relu_op(self):
        a = Value(random.uniform(0, 1000))
        L = autograd.engine.ReLUOp.act(a)
        L.backward()
        self.assertEqual(a.grad, 1)
        a = Value(random.uniform(-10000, 0))
        L = autograd.engine.ReLUOp.act(a)
        L.backward()
        self.assertEqual(a.grad, 0)
        a = Value(random.uniform(-100, 100))
        L = autograd.engine.ReLUOp.act(a)
        L.backward()
        self.assertEqual(a.grad, 1 if a.data > 0 else 0)

    def test_MSE(self):
        a, b = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))
        L = (a-b)**2
        L.backward()
        self.assertEqual(a.grad, 2*(a.data-b.data))
        self.assertEqual(b.grad, -2*(a.data-b.data))

    def test_pow_diff(self):
        a, b = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))
        c = random.uniform(-100, 100)
        L = (a-b)**c
        L.backward()
        self.assertEqual(a.grad, c*((a.data-b.data)**(c-1)))
        self.assertEqual(b.grad, -c*((a.data-b.data)**(c-1)))

    def test_MSE_relu(self):
        a, b = Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))
        L = autograd.engine.ReLUOp.act((a-b))**2
        L.backward()
        self.assertEqual(a.grad, 2*autograd.engine.ReLUOp.act(a-b).data*(1 if a.data > 0 else 0))
        self.assertEqual(b.grad, 2*autograd.engine.ReLUOp.act(a-b).data*(1 if b.data > 0 else 0))




if __name__ == "__main__":
    unittest.main()
