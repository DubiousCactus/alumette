#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Test the neural network library on a toy regression problem.
"""

import unittest
import random

from alumette import Tensor
from alumette.nn import NeuralNet, Layer, SGD, MSE

from tqdm import trange

class MyNet(NeuralNet):
    def __init__(self, activation) -> None:
        super().__init__()
        self.layer1 = Layer(1, 5, activation=activation)
        # self.layer2 = Layer(10, 10, activation="tanh")
        self.layer3 = Layer(5, 1, activation="identity")

    def forward(self, x):
        y = self.layer1(x)
        # y = self.layer2(y)
        y = self.layer3(y)
        return y

def test_func_1(x):
    return 9*x**3 + (3 * (x**2)) - (8 * x) + 3/4


class TestNeuralNet(unittest.TestCase):
    def test_MLP_tanh(self):
        nn = MyNet("tanh")
        opt = SGD(nn.parameters(), lr=1e-5)
        xs = []
        for _ in range(1000):
            xs.append(random.uniform(-2, 2))

        N_EPOCHS = 1000
        t = trange(N_EPOCHS, desc='Training', leave=True)
        for i in t:
            tot_loss = 0.0
            opt.zero_grad()
            random.shuffle(xs)
            ys = [test_func_1(x) for x in xs]
            for x, y in zip(xs, ys):
                y_hat = nn(Tensor(x))[0]
                loss = MSE(y_hat, Tensor(y))
                tot_loss += loss
            tot_loss.backward()
            opt.step()
            tot_loss = tot_loss.data/len(xs)
            t.set_description(f"[*] Training -- Epochs {i+1}/{N_EPOCHS}: loss={tot_loss:.4f}")
            t.refresh() # to show immediately the update


        print("[*] Testing...")
        for _ in range(500):
            xs.append(random.uniform(-2, 2))
        ys = [test_func_1(x) for x in xs]

        test_loss = .0
        for x, y in zip(xs, ys):
            y_hat = nn(Tensor(x))[0]
            loss = MSE(y_hat, Tensor(y))
            test_loss += loss
        print(f"--> Final test loss: {test_loss.data/len(xs):.4f}")


if __name__ == "__main__":
    unittest.main()
