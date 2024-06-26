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

import random
import unittest

import numpy as np

import alumette
from alumette import Tensor, grad_check


# TODO: Reorganize tests into categories?
class ScalarManualTests(unittest.TestCase):
    def test_int_linear(self):
        a, b, c = (
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
        )
        d = a * b + c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, b.data)
        self.assertEqual(b.grad, a.data)

    def test_float_linear(self):
        a, b, c = (
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
        )
        d = a * b + c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, b.data)
        self.assertEqual(b.grad, a.data)

    def test_int_linear_2(self):
        a, b, c = (
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
        )
        d = (a + b) * c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, a.data + b.data)
        self.assertEqual(a.grad, c.data)
        self.assertEqual(b.grad, c.data)

    def test_float_linear_2(self):
        a, b, c = (
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
        )
        d = (a + b) * c
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        d.backward()
        self.assertEqual(c.grad, a.data + b.data)
        self.assertEqual(a.grad, c.data)
        self.assertEqual(b.grad, c.data)

    def test_int_chain(self):
        a, b, c, d = (
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
        )
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
        a, b, c, d = (
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
        )
        L = d * (c + (a * b))
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        self.assertEqual(d.grad, 0)
        L.backward()
        self.assertEqual(d.grad, (a.data * b.data) + c.data)
        self.assertEqual(c.grad, d.data)
        self.assertTrue(np.allclose(b.grad, a.data * d.data))
        self.assertTrue(np.allclose(a.grad, b.data * d.data))

    def test_int_squared_chain(self):
        a, b, c, d = (
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
            Tensor(random.randint(-100, 100)),
        )
        exp, denom = 3, 8
        L = a * ((b - c) ** 3) + (d / 8)
        L.backward()
        self.assertEqual(a.grad, (b.data - c.data) ** exp)
        self.assertEqual(b.grad, exp * a.data * (b.data - c.data) ** (exp - 1))
        self.assertEqual(c.grad, -exp * a.data * (b.data - c.data) ** (exp - 1))
        self.assertEqual(d.grad, 1 / denom)

    def test_float_squared_chain(self):
        a, b, c, d = (
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
        )
        exp, denom = 3, 8
        L = a * ((b - c) ** 3) + (d / 8)
        L.backward()
        self.assertTrue(np.allclose(a.grad, (b.data - c.data) ** exp))
        self.assertTrue(
            np.allclose(b.grad, exp * a.data * (b.data - c.data) ** (exp - 1))
        )
        self.assertTrue(
            np.allclose(c.grad, -exp * a.data * (b.data - c.data) ** (exp - 1))
        )
        self.assertTrue(np.allclose(d.grad, 1 / denom))

    def test_add_self(self):
        a = Tensor(random.uniform(-100, 100))
        (a + a + a).backward()
        self.assertEqual(a.grad, 3)

    def test_r_add(self):
        a = Tensor(random.uniform(-100, 100))
        b = random.uniform(-100, 100)
        L = b + a
        L.backward()
        self.assertEqual(a.grad, 1)

    def test_r_sub(self):
        a = Tensor(random.uniform(-100, 100))
        b = random.uniform(-100, 100)
        L = b - a
        L.backward()
        self.assertEqual(a.grad, -1)

    def test_r_mul(self):
        a = Tensor(random.uniform(-100, 100))
        b = random.uniform(-100, 100)
        L = b * a
        L.backward()
        self.assertTrue(np.allclose(a.grad, b))

    def test_neg_r_mul(self):
        a = Tensor(random.uniform(-100, 100))
        b = random.uniform(-100, 100)
        L = (-b) * a
        L.backward()
        self.assertTrue(np.allclose(a.grad, -b))

    def test_positive_pow(self):
        a = Tensor(random.uniform(-100, 100))
        b = int(random.uniform(0, 10))
        (a**b).backward()
        self.assertTrue(np.allclose(a.grad, b * (a.data ** (b - 1))))

    def test_negative_pow(self):
        a = Tensor(random.uniform(-100, 100))
        b = int(random.uniform(-10, -1))
        (a**b).backward()
        self.assertTrue(np.allclose(a.grad, b * (a.data ** (b - 1))))

    def test_zero_pow(self):
        a = Tensor(random.uniform(-100, 100))
        b = 0
        (a**b).backward()
        self.assertEqual(a.grad, 0)

    def test_div(self):
        a, b = Tensor(random.uniform(-10, 10)), Tensor(random.uniform(-10, 10))
        L = a / b
        L.backward()
        self.assertTrue(np.allclose(a.grad, 1 / b.data))
        self.assertTrue(np.allclose(b.grad, -a.data / (b.data**2)))

    def test_large_div(self):
        a, b = Tensor(random.uniform(-1000, 1000)), Tensor(random.uniform(-1000, 1000))
        L = a / b
        L.backward()
        self.assertTrue(np.allclose(a.grad, 1 / b.data))
        self.assertTrue(np.allclose(b.grad, -a.data / (b.data**2)))

    def test_r_ops(self):
        a, b, c = (
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-100, 100)),
            Tensor(random.uniform(-10, 10)),
        )
        d, e, f = (
            random.uniform(-100, 100),
            random.uniform(-100, 100),
            int(random.uniform(-10, 10)),
        )
        L = (-d * (e - a)) + ((f + b) / (c**f))
        L.backward()
        self.assertTrue(np.allclose(a.grad, d))
        self.assertTrue(np.allclose(b.grad, 1 / (c.data**f)))
        self.assertTrue(np.allclose(c.grad, -f * (c.data ** (-f - 1)) * (f + b.data)))

    def test_relu_op_backward(self):
        a = Tensor(random.uniform(0, 1000))
        L = alumette.relu(a)
        L.backward()
        self.assertEqual(a.grad, 1)
        a = Tensor(random.uniform(-10000, 0))
        L = alumette.relu(a)
        L.backward()
        self.assertEqual(a.grad, 0)
        a = Tensor(random.uniform(-100, 100))
        L = alumette.relu(a)
        L.backward()
        self.assertEqual(a.grad, 1 if a.data > 0 else 0)

    def test_MSE(self):
        a, b = Tensor(random.uniform(-100, 100)), Tensor(random.uniform(-100, 100))
        L = (a - b) ** 2
        L.backward()
        self.assertEqual(a.grad, 2 * (a.data - b.data))
        self.assertEqual(b.grad, -2 * (a.data - b.data))

    def test_pow_diff(self):
        a, b = Tensor(random.uniform(-100, 100)), Tensor(random.uniform(-100, 100))
        c = int(random.uniform(-10, 10))
        L = (a - b) ** c
        L.backward()
        self.assertTrue(np.allclose(a.grad, c * ((a.data - b.data) ** (c - 1))))
        self.assertTrue(np.allclose(b.grad, -c * ((a.data - b.data) ** (c - 1))))

    def test_MSE_relu(self):
        a, b = Tensor(random.uniform(-100, 100)), Tensor(random.uniform(-100, 100))
        (alumette.relu((a - b)) ** 2).backward()
        self.assertEqual(
            a.grad,
            2 * alumette.relu(a - b).data * (1 if (a.data - b.data) > 0 else 0),
        )
        self.assertEqual(
            b.grad,
            -2 * alumette.relu(a - b).data * (1 if (a.data - b.data) > 0 else 0),
        )

    def test_tanh_op_backward(self):
        a = Tensor(random.uniform(-100, 100))
        (alumette.tanh(a)).backward()
        self.assertTrue(np.allclose(a.grad, 1 - (alumette.tanh(a).data ** 2)))

    def test_MSE_tanh(self):
        a, b = Tensor(random.uniform(-100, 100)), Tensor(random.uniform(-100, 100))
        (alumette.tanh((a - b)) ** 2).backward()
        self.assertTrue(
            np.allclose(
                a.grad,
                2 * alumette.tanh(a - b).data * (1 - (alumette.tanh(a - b).data ** 2)),
            )
        )
        self.assertTrue(
            np.allclose(
                b.grad,
                -2 * alumette.tanh(a - b).data * (1 - (alumette.tanh(a - b).data ** 2)),
            )
        )


class MatrixGradcheckTests(unittest.TestCase):
    """
    More complex tests suite using grad_check.
    """

    def test_add_self(self):
        a = Tensor(np.random.random((random.randint(1, 50), random.randint(1, 50))))
        (a + a + a).backward()
        self.assertTrue(np.allclose(a.grad, np.ones_like(a.data) * 3))

    def test_r_add(self):
        dims = (random.randint(1, 50), random.randint(1, 50))
        a = Tensor(np.random.random(dims))
        b = Tensor(np.random.random(dims))
        (b + a).backward()
        self.assertTrue(np.allclose(a.grad, np.ones_like(a.data)))

    def test_r_sub(self):
        dims = (random.randint(1, 50), random.randint(1, 50))
        a = Tensor(np.random.random(dims))
        b = Tensor(np.random.random(dims))
        (b - a).backward()
        self.assertTrue(np.allclose(a.grad, -np.ones_like(a.data)))

    def test_r_mul(self):
        a = Tensor(random.uniform(-100, 100))
        b = random.uniform(-100, 100)
        (b * a).backward()
        self.assertTrue(np.allclose(a.grad, b))

    def test_neg_r_mul(self):
        a = Tensor(random.uniform(-100, 100))
        b = random.uniform(-100, 100)
        ((-b) * a).backward()
        self.assertTrue(np.allclose(a.grad, -b))

    def test_positive_pow(self):
        a = Tensor(random.uniform(-100, 100))
        b = int(random.uniform(0, 10))
        (a**b).backward()
        self.assertTrue(np.allclose(a.grad, b * (a.data ** (b - 1))))

    def test_negative_pow(self):
        a = Tensor(random.uniform(-100, 100))
        b = int(random.uniform(-10, -1))
        (a**b).backward()
        self.assertTrue(np.allclose(a.grad, b * (a.data ** (b - 1))))

    def test_zero_pow(self):
        a = Tensor(random.uniform(-100, 100))
        b = 0
        (a**b).backward()
        self.assertEqual(a.grad, 0)

    def test_div(self):
        a, b = Tensor(random.uniform(-10, 10)), Tensor(random.uniform(-10, 10))
        L = a / b
        L.backward()
        self.assertTrue(np.allclose(a.grad, 1 / b.data))
        self.assertTrue(np.allclose(b.grad, -a.data / (b.data**2)))

    def test_vector_matmul(self):
        dim = random.randint(1, 20)
        a = Tensor(np.random.random((dim)))
        b = Tensor(np.random.random((dim)))
        (a @ b).backward()
        exp = lambda a, b: a @ b
        self.assertTrue(grad_check([a, b], exp, 0, a.grad))
        self.assertTrue(grad_check([a, b], exp, 1, b.grad))

    def test_matrix_vector_matmul(self):
        vec_dim = random.randint(1, 20)
        mat_dim = vec_dim, random.randint(1, 20)
        a = Tensor(np.random.random(mat_dim))
        b = Tensor(np.random.random(mat_dim))
        c = Tensor(np.random.random((vec_dim)))
        (((a @ b.T) @ c) @ c).backward()
        exp = lambda a, b, c: ((a @ b.T) @ c) @ c
        self.assertTrue(grad_check([a, b, c], exp, 0, a.grad))
        self.assertTrue(grad_check([a, b, c], exp, 1, b.grad))
        self.assertTrue(grad_check([a, b, c], exp, 2, c.grad))

    def test_matrix_vector_mul(self):
        vec_dim = random.randint(1, 20)
        mat_dim = vec_dim, random.randint(1, 20)
        a = Tensor(np.random.random(mat_dim))
        b = Tensor(np.random.random(mat_dim))
        c = Tensor(np.random.random((vec_dim)))
        d = Tensor(np.random.random((mat_dim[1])))
        exp = lambda a, b, c, d: ((a * b).T @ c) @ d
        exp(a, b, c, d).backward()
        self.assertTrue(grad_check([a, b, c, d], exp, 0, np.array(a.grad)))
        self.assertTrue(grad_check([a, b, c, d], exp, 1, np.array(b.grad)))
        self.assertTrue(grad_check([a, b, c, d], exp, 2, np.array(c.grad)))
        self.assertTrue(grad_check([a, b, c, d], exp, 3, np.array(d.grad)))

    def test_vector_mul(self):
        dim = random.randint(1, 20)
        a = Tensor(np.random.random((dim)))
        b = Tensor(np.random.random((dim)))
        c = Tensor(np.random.random((dim)))
        exp = lambda a, b, c: (a * b) @ c
        exp(a, b, c).backward()
        self.assertTrue(grad_check([a, b, c], exp, 0, np.array(a.grad)))
        self.assertTrue(grad_check([a, b, c], exp, 1, np.array(b.grad)))
        self.assertTrue(grad_check([a, b, c], exp, 2, np.array(c.grad)))

    def test_linear_layer(self):
        vec_dim = random.randint(2, 20)
        mat_dim = vec_dim, random.randint(2, 20)
        w = Tensor(np.random.random(mat_dim))
        x = Tensor(np.random.random((vec_dim)))
        b = Tensor(np.random.random((mat_dim[1])))
        c = Tensor(np.random.random((mat_dim[1])))
        exp = lambda w, x, b, c: (w.T @ x + b) @ c
        exp(w, x, b, c).backward()
        self.assertTrue(grad_check([w, x, b, c], exp, 0, np.array(w.grad)))
        self.assertTrue(grad_check([w, x, b, c], exp, 2, np.array(b.grad)))
        self.assertTrue(grad_check([w, x, b, c], exp, 3, np.array(c.grad)))

    def test_linear_layer_relu(self):
        vec_dim = random.randint(2, 20)
        mat_dim = vec_dim, random.randint(2, 20)
        w = Tensor(np.random.random(mat_dim))
        x = Tensor(np.random.random((vec_dim)))
        b = Tensor(np.random.random((mat_dim[1])))
        c = Tensor(np.random.random((mat_dim[1])))
        exp = lambda w, x, b, c: (w.T @ x + b) @ c

        def relu_layer(w, x, b, c):
            res = exp(w, x, b, c)
            return alumette.relu(Tensor(res) if not isinstance(res, Tensor) else res)

        relu_layer(w, x, b, c).backward()
        self.assertTrue(grad_check([w, x, b, c], relu_layer, 0, np.array(w.grad)))
        self.assertTrue(grad_check([w, x, b, c], relu_layer, 2, np.array(b.grad)))
        self.assertTrue(grad_check([w, x, b, c], relu_layer, 3, np.array(c.grad)))

    def test_linear_layer_tanh(self):
        vec_dim = random.randint(2, 20)
        mat_dim = vec_dim, random.randint(2, 20)
        w = Tensor(np.random.random(mat_dim))
        x = Tensor(np.random.random((vec_dim)))
        b = Tensor(np.random.random((mat_dim[1])))
        c = Tensor(np.random.random((mat_dim[1])))
        exp = lambda w, x, b, c: (w.T @ x + b) @ c

        def tanh_layer(w, x, b, c):
            res = exp(w, x, b, c)
            return alumette.tanh(Tensor(res) if not isinstance(res, Tensor) else res)

        tanh_layer(w, x, b, c).backward()
        self.assertTrue(grad_check([w, x, b, c], tanh_layer, 0, np.array(w.grad)))
        # self.assertTrue(grad_check([w, x, b, c], tanh_layer, 2, np.array(b.grad)))
        # self.assertTrue(grad_check([w, x, b, c], tanh_layer, 3, np.array(c.grad)))


if __name__ == "__main__":
    unittest.main()
