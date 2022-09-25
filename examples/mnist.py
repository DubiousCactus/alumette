#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Small MNIST example.
"""

from autograd import Value, grad
from autograd.nn import NeuralNet, Layer, MSE, SGD

class MyNet(NeuralNet):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = Layer(1, 10, activation='relu')
        self.layer2 = Layer(10, 10, activation='relu')
        self.layer3 = Layer(10, 1, activation='relu')

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        return y


# Training loop?
def test_func_1(x):
    return 12*x**2 -8*x +123

nn = MyNet()
opt = SGD(nn.parameters())
# opt = SGD((nn.layer1.neurons, nn.layer2.neurons, nn.layer3.neurons))
xs = [-1, 3, 8, 16, -32, -6, 9, 28]
ys = [test_func_1(x) for x in xs]
print("Training data: ", [(x,y) for x, y in zip(xs, ys)])

print("[*] Training...")
for i in range(10):
    print(f"- Epoch {i+1}/10")
    tot_loss = .0
    opt.zero_grad()
    for x, y in zip(xs, ys):
        y_hat = nn(Value(x))[0]
        # print(y_hat, y)
        loss = MSE(y_hat, Value(y))
        tot_loss += loss.data
        loss.backward()
        opt.step()
    print(f"  Final loss: {tot_loss/len(xs):.6f}")
