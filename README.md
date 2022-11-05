## Introduction

Alumette (match in French, as in *a tiny torch*) is a tiny neural networks library with a
reverse-mode automatic differentiation engine. It is roughly based on Karpathy's micrograd, but it
aims to be a little bit more usable by wrapping Numpy arrays around Tensors and implementing other
Tensor optimization gadgets.


### Why?
Because it feels so good to understand how Pytorch really works! And it's super fun to code ;)


## Installation

**TODO**

I recommend Python 3.11 for the speed boost!

## Usage

**TODO**

## Resources

- [Karpathy's video course]()
- [Karpathy's autograd code]()
- [Geohotz's Tinygrad]()
- [This very good PDF on Tensor derivatives]()

## TODO:

- [x] Build autograd on scalars
- [x] Build small neural network library
- [x] Write neural net example
- [x] Test gradients numerically
- [x] Implement a Tensor class to wrap Numpy ndarrays
- [x] Implement a neural net training example for 1D curve fitting
- [ ] Implement batching
- [ ] Implement convolutions
- [ ] Implement a neural net training example for image classification (MNIST)
- [ ] GPU acceleration (PyCuda?)
