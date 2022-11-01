#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Test the grad_check function
"""

import numpy as np
import unittest
import random
import torch

from alumette import Tensor, grad_check
import alumette

class TestGradcheck(unittest.TestCase):
    def test_vector_mul(self):
        dim = random.randint(1, 20)
        a = Tensor(np.random.random((dim)))
        b = Tensor(np.random.random((dim)))
        c = Tensor(np.random.random((dim)))
        exp = lambda a, b, c: (a * b) @ c
        ta, tb, tc = torch.tensor(a.numpy().copy(), requires_grad=True), torch.tensor(
            b.numpy().copy(), requires_grad=True
        ), torch.tensor(
            c.numpy().copy(), requires_grad=True
        )
        exp(ta, tb, tc).backward()
        self.assertTrue(grad_check([a, b, c], exp, 0, np.array(ta.grad)))
        self.assertTrue(grad_check([a, b, c], exp, 1, np.array(tb.grad)))
        self.assertTrue(grad_check([a, b, c], exp, 2, np.array(tc.grad)))

    def test_vector_matmul(self):
        dim = random.randint(1, 20)
        a = Tensor(np.random.random((dim)))
        b = Tensor(np.random.random((dim)))
        exp = lambda a, b: a @ b
        ta, tb = torch.tensor(a.numpy().copy(), requires_grad=True), torch.tensor(
            b.numpy().copy(), requires_grad=True
        )
        exp(ta, tb).backward()
        self.assertTrue(grad_check([a, b], exp, 0, np.array(ta.grad)))
        self.assertTrue(grad_check([a, b], exp, 1, np.array(tb.grad)))

    def test_matrix_vector_mul(self):
        vec_dim = random.randint(1, 20)
        mat_dim = vec_dim, random.randint(1, 20)
        a = Tensor(np.random.random(mat_dim))
        b = Tensor(np.random.random(mat_dim))
        c = Tensor(np.random.random((vec_dim)))
        d = Tensor(np.random.random((mat_dim[1])))
        exp = lambda a, b, c, d: ((a * b).T @ c) @ d
        ta, tb, tc, td = (
            torch.tensor(a.numpy().copy(), requires_grad=True),
            torch.tensor(b.numpy().copy(), requires_grad=True),
            torch.tensor(c.numpy().copy(), requires_grad=True),
            torch.tensor(d.numpy().copy(), requires_grad=True),
        )
        exp(ta, tb, tc, td).backward()
        self.assertTrue(grad_check([a, b, c, d], exp, 0, np.array(ta.grad)))
        self.assertTrue(grad_check([a, b, c, d], exp, 1, np.array(tb.grad)))
        self.assertTrue(grad_check([a, b, c, d], exp, 2, np.array(tc.grad)))
        self.assertTrue(grad_check([a, b, c, d], exp, 3, np.array(td.grad)))


    def test_matrix_vector_matmul(self):
        vec_dim = random.randint(1, 20)
        mat_dim = vec_dim, random.randint(1, 20)
        a = Tensor(np.random.random(mat_dim))
        b = Tensor(np.random.random(mat_dim))
        c = Tensor(np.random.random((vec_dim)))
        exp = lambda a, b, c: ((a @ b.T) @ c) @ c
        ta, tb, tc = (
            torch.tensor(a.numpy().copy(), requires_grad=True),
            torch.tensor(b.numpy().copy(), requires_grad=True),
            torch.tensor(c.numpy().copy(), requires_grad=True),
        )
        exp(ta, tb, tc).backward()
        self.assertTrue(grad_check([a, b, c], exp, 0, np.array(ta.grad)))
        self.assertTrue(grad_check([a, b, c], exp, 1, np.array(tb.grad)))
        self.assertTrue(grad_check([a, b, c], exp, 2, np.array(tc.grad)))

    def test_linear_layer(self):
        vec_dim = random.randint(2, 20)
        mat_dim = vec_dim, random.randint(2, 20)
        w = Tensor(np.random.random(mat_dim))
        x = Tensor(np.random.random((vec_dim)))
        b = Tensor(np.random.random((mat_dim[1])))
        c = Tensor(np.random.random((mat_dim[1])))
        exp = lambda w, x, b, c: (w.T @ x + b) @ c
        tw, tx, tb, tc = (
            torch.tensor(w.numpy().copy(), requires_grad=True),
            torch.tensor(x.numpy().copy(), requires_grad=True),
            torch.tensor(b.numpy().copy(), requires_grad=True),
            torch.tensor(c.numpy().copy(), requires_grad=True),
        )
        exp(tw, tx, tb, tc).backward()
        self.assertTrue(grad_check([w, x, b, c], exp, 0, np.array(tw.grad)))
        self.assertTrue(grad_check([w, x, b, c], exp, 2, np.array(tb.grad)))
        self.assertTrue(grad_check([w, x, b, c], exp, 3, np.array(tc.grad)))

    def test_linear_layer_relu(self):
        vec_dim = random.randint(2, 20)
        mat_dim = vec_dim, random.randint(2, 20)
        w = Tensor(np.random.random(mat_dim))
        x = Tensor(np.random.random((vec_dim)))
        b = Tensor(np.random.random((mat_dim[1])))
        c = Tensor(np.random.random((mat_dim[1])))
        exp = lambda w, x, b, c: (w.T @ x + b) @ c
        def relu_layer(w, x, b, c):
            res = Tensor(exp(w,x,b,c))
            return alumette.relu(res)

        tw, tx, tb, tc = (
            torch.tensor(w.numpy().astype(np.float64).copy(), requires_grad=True),
            torch.tensor(x.numpy().astype(np.float64).copy(), requires_grad=True),
            torch.tensor(b.numpy().astype(np.float64).copy(), requires_grad=True),
            torch.tensor(c.numpy().astype(np.float64).copy(), requires_grad=True),
        )
        torch.relu(exp(tw, tx, tb, tc)).backward()
        self.assertTrue(grad_check([w, x, b, c], relu_layer, 0, np.array(tw.grad)))
        self.assertTrue(grad_check([w, x, b, c], relu_layer, 2, np.array(tb.grad)))
        self.assertTrue(grad_check([w, x, b, c], relu_layer, 3, np.array(tc.grad)))

    # TODO: All edge cases where we have matrices of (Nx1), (1xN) and the sort

if __name__ == "__main__":
    alumette.set_default_dtype(np.float128)
    unittest.main()
