<p align="center">
  <img src="alumette_logo.png" />
</p>

## What is Alumette?

Alumette (match in French, as in *a tiny torch*) is a tiny neural networks library with a
reverse-mode automatic differentiation engine. It is roughly based on Karpathy's micrograd, but it
aims to be a little bit more usable by wrapping Numpy arrays around Tensors and implementing other
Tensor optimization gadgets.


### Why?
Because it feels so good to understand how Pytorch really works! And it's super fun to code :)


## Installation

Run `python setup.py build && python setup.py install` in your environement and you're ready to go!
I recommend Python 3.11 for the speed boost!

## Usage

### Simple differentiable scalar Tensor operations
```Python
from alumette import  Tensor
a = Tensor(3.2, requires_grad=True)
b = Tensor(-4, requires_grad=True)
((a*b - (b/a))**2).backward() # Compute gradients of all nodes that require grad
print(a.grad, b.grad) # Access node gradient
```

### Differentiable of Numpy nd-array Tensor operations
```Python
from alumette import  Tensor
import numpy as np
a = Tensor(np.random.random((5, 2)), requires_grad=True) # From Numpy nd-array
b = Tensor([[0.1], [-1.5]], requires_grad=True) # Automatic nd-array creation from list
c = Tensor(np.random.random((5, 1)), requires_grad=True)
((a@b).T @ c).backward() # Compute gradients of all nodes that require grad
print(a.grad, b.grad, c.grad) # Access node gradient
```

### Neural networks training
```Python
from alumette.nn import Linear, NeuralNet, MSE, SGD
import random

class MyNet(NeuralNet):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = Linear(1, 15, activation="relu")
        self.layer2 = Linear(15, 1, activation="identity")

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return y


def test_func_1(x):
    return 9 * x**3 + (3 * (x**2)) - (8 * x) + 3 / 4

nn = MyNet()
opt = SGD(nn.parameters(), lr=1e-5)
xs = [random.uniform(-1, 1) for _ in range(1000)]

for _ in range(100):
	tot_loss = 0.0
	opt.zero_grad()
	random.shuffle(xs)
	ys = [test_func_1(x) for x in xs]
	for x, y in zip(xs, ys):
		y_hat = nn(Tensor(x).unsqueeze(0))
		loss = MSE(y_hat, Tensor(y))
		tot_loss += loss
	tot_loss.backward()
	opt.step()

```

## Resources

- [Karpathy's video course](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Karpathy's micrograd project](https://github.com/karpathy/micrograd)
- [Geohotz's Tinygrad](https://github.com/geohot/tinygrad/)
- [Ari Seff's video introduction on automatic differentiation](https://www.youtube.com/watch?v=wG_nF1awSSY)
- [This very good PDF on Tensor derivatives](https://cs231n.stanford.edu/vecDerivs.pdf)

## TODO:

- [x] Build autograd on scalars
- [x] Build small neural network library
- [x] Write neural net example
- [x] Test gradients numerically
- [x] Implement a Tensor class to wrap Numpy ndarrays
- [x] Implement a neural net training example for 1D curve fitting
- [ ] Make grad a Tensor to allow for higher-order differentiation
- [ ] Implement batching
- [ ] Implement convolutions
- [ ] Implement a neural net training example for image classification (MNIST)
- [ ] GPU acceleration (PyCuda?)
