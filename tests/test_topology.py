#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

import unittest

import numpy as np
import torch

from alumette import Tensor, relu


class TestTopology(unittest.TestCase):
    def test_sanity_check(self):
        x = Tensor(-4.0)
        z = 2 * x + 2 + x
        q = relu(z) + z * x
        h = relu(z * z)
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xpt, ypt = x, y

        # forward pass went well
        self.assertEqual(ymg.data, ypt.data.item())
        self.assertEqual(xmg.grad, xpt.grad.item())

    def test_more_ops(self):
        a = Tensor(-4.0)
        b = Tensor(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + relu(b + a)
        d += 3 * d + relu(b - a)
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6
        self.assertTrue(np.allclose(gmg.data, gpt.data.item(), rtol=tol))
        self.assertTrue(np.allclose(amg.data, apt.data.item(), rtol=tol))
        self.assertTrue(np.allclose(bmg.grad, bpt.grad.item(), rtol=tol))


if __name__ == "__main__":
    unittest.main()
