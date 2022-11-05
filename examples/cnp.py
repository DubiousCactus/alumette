#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training a Conditional Neural Process
"""

import matplotlib.pyplot as plt
import numpy as np
import minigauss

from alumette.nn import NeuralNet, Layer, MSE, SGD, MLP
from alumette.engine import Value
import alumette

from numpy.random import default_rng
from collections import namedtuple
from typing import List
from tqdm import tqdm

from minigauss import GaussianProcess
from minigauss.priors import ExponentialKernel
from minigauss.priors.mean import ConstantFunc

sample = namedtuple("sample", ["context", "targets"])


class CNP(NeuralNet):
    def __init__(
        self,
        input_dim=2,
        output_dim=2,
        encoder_dim=128,
        decoder_dim=128,
        encoder_layers=4,
        decoder_layers=3,
    ) -> None:
        super().__init__()
        # The encoder (MLP) takes in pairs of (x, y) context points and returns r_i
        #   -> It concatenates x and y
        #   -> Num of layers is a hyperparameter (4?)
        #   -> Width of layers is a hyperparameter (4?)
        #   -> ReLU activations except for the last layer
        self.encoder = MLP(
            input_dim, encoder_dim, encoder_dim, encoder_layers, activation="relu"
        )
        # The aggregator (mean operator) aggregates context point encodings r_i to produce r
        #   -> Of dim r_i
        # The decoder (MLP) takes in the aggregated r and a target x_i to produce a mean estimate mu_i + sigma_i
        #   -> Num of layers is a hyperparameter (4?)
        #   -> Width of layers is a hyperparameter (4?)
        self.decoder = MLP(
            encoder_dim + input_dim // 2,
            decoder_dim,
            output_dim,
            decoder_layers,
            activation="relu",
        )

    def forward(
        self,
        context_xs: List[Value],
        context_ys: List[Value],
        target_xs: List[Value],
    ):
        reps = [
            self.encoder((context_x, context_y))
            for context_x, context_y in zip(context_xs, context_ys)
        ]
        # Should have a list of N_CONTEXT_PTS lists of 128 values
        print(len(reps), len(reps[0]))
        aggregated_r = alumette.mean(reps)
        print(aggregated_r)
        # target_ys is a list of tuples (mu, log_sigma)
        target_ys = [self.decoder((aggregated_r, target_x)) for target_x in target_xs]
        mus, log_sigmas = map(list, zip(*target_ys))
        sigmas = [0.1 + 0.9 * alumette.softplus(log_sigma) for log_sigma in log_sigmas]
        return mus, sigmas


class OracleGP:
    def __init__(self, max_ctx_pts, mean=0, sigma_y=1.0, l=0.4, batch_size=64) -> None:
        self.gp = GaussianProcess(
            ConstantFunc(value=mean), ExponentialKernel(sigma_y=sigma_y, l=l)
        )
        self.rng = default_rng()
        self.max_context_pts = max_ctx_pts
        self.max_target_pts = max_ctx_pts
        self._batch_size = batch_size
        # TODO: Pregenerate N GP samples to resample from in generate()
        # TODO: Rename generate() to sample()

    def generate(self, eval=False):
        xrange = (-2, 2)
        # Sort the inputs
        x = np.sort((xrange[0] - xrange[1]) * np.random.rand(100, 1) + xrange[1])
        # TODO: Add batched input in minigauss!
        batch_f = np.zeros((self._batch_size, 100))
        for i in range(self._batch_size):
            f, _, _ = self.gp.sample(x)
            # TODO: Return same dim as input in minigauss!
            batch_f[i, :] = f
        batch_x = np.resize(np.expand_dims(x, 0), (self._batch_size, 100))
        # TODO: Randomize n_context/targets in the batch itself or just across batches is fine?
        n_context = int(self.rng.uniform(low=3, high=self.max_context_pts, size=1)[0])
        n_targets = (
            100 - n_context
            if eval
            else int(self.rng.uniform(low=2, high=self.max_target_pts, size=1)[0])
        )
        idx = np.zeros((self._batch_size, 100), dtype=int)
        for i in range(self._batch_size):
            idx[i, :] = np.random.permutation(100).astype(int)
        sorted_ctx_x = [0]*self._batch_size#np.zeros((self._batch_size, n_context))
        sorted_ctx_y = [0]*self._batch_size#np.zeros((self._batch_size, n_context))
        for i in range(self._batch_size):
            sorted_ctx_x[i] = [Value(x) for x in batch_x[i, idx[i, :n_context]]]
            sorted_ctx_y[i] = [Value(y) for y in batch_f[i, idx[i, :n_context]]]

        if eval:
            yield sample(
                (sorted_ctx_x, sorted_ctx_y),
                (batch_x, batch_f),
            )
        else:
            # sorted_tgt_x = np.zeros((self._batch_size, n_context+n_targets))
            # sorted_tgt_y = np.zeros((self._batch_size, n_context+n_targets))
            sorted_tgt_x = [0]*self._batch_size#np.zeros((self._batch_size, n_context))
            sorted_tgt_y = [0]*self._batch_size#np.zeros((self._batch_size, n_context))
            for i in range(self._batch_size):
                sorted_tgt_x[i] = [Value(x) for x in batch_x[i, idx[i, : n_context + n_targets]]]
                sorted_tgt_y[i] = [Value(y) for y in batch_f[i, idx[i, : n_context + n_targets]]]
            yield sample(
                (sorted_ctx_x, sorted_ctx_y),
                (sorted_tgt_x, sorted_tgt_y),
            )


def plot_functions(target_x, target_y, context_x, context_y, pred_y, var):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape batchsize x number_targets x 1 that contains the
          x values of the target points.
      target_y: An array of shape batchsize x number_targets x 1 that contains the
          y values of the target points.
      context_x: An array of shape batchsize x number_context x 1 that contains
          the x values of the context points.
      context_y: An array of shape batchsize x number_context x 1 that contains
          the y values of the context points.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted means of the y values at the target points in target_x.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted variance of the y values at the target points in target_x.
    """
    # Plot everything
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2)
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2)
    plt.plot(context_x[0], context_y[0], "ko", markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.2,
        facecolor="#65c9f7",
        interpolate=True,
    )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid("off")
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.show()


# -> A dataset of functions coming from multiple random GPs with the same kernel (but different
# hyperparameters?)
# -> Use K context points + K target points
# -> The targets (during training) are (context + target)

# Define the training loop
# o At each iteration:
#   + Sample a meta-batch (functions from the GP with) and sample random context/target points
#   + Run inference on batch, compute loss w.r.t. GT targets
#
#   -> Random amount of context points (between 3 and MAX)
#   -> Loss function: log likelihood of ground truth y_i under a GP(mu_i, sigma_i)
#   -> Clip the variance sigma_i at 0.1 to avoid collapsing?
#   -> Batch size: 64 | Max context points: 10

epochs = 100
val_every = 1
iterations = 10
batch_size = 16
cnp = CNP()
oracle = OracleGP(max_ctx_pts=10, batch_size=batch_size)
opt = SGD(cnp.parameters(), lr=1e-4)
# nllloss = alumette.nn.GaussianNLLLoss()
nllloss = MSE
for i in range(epochs):
    epoch_loss = 0.0
    for _ in tqdm(range(iterations)):
        opt.zero_grad()
        f = next(oracle.generate())
        # dist, means, vars = cnp(f.context, f.targets[0])
        # means, vars = cnp(f.context[0], f.context[1], f.targets[0])
        means, vars = cnp(f.context[0][0], f.context[1][0], f.targets[0][0])
        # loss = -torch.mean(dist.log_prob(f.targets[1]))
        loss = nllloss(means, f.targets[1], vars)
        epoch_loss += loss.detach().item()
        loss.backward()
        opt.step()
    epoch_loss /= iterations
    print(f"Epoch {i+1}/{epochs} -- Loss={epoch_loss}")

    if i % val_every == 0:
        rng = default_rng()
        f = next(oracle.generate(eval=True))
        means, vars = cnp(f.context[0][0], f.context[1][0], f.targets[0][0])
        # _, mu, var = cnp(f.context, f.targets[0], eval=True)
        plot_functions(
            f.targets[0],
            f.targets[1],
            f.context[0],
            f.context[1],
            means,
            vars,
        )

rng = default_rng()
f = next(oracle.generate(eval=True))
mu, var = cnp(f.context, f.targets[0], eval=True)
# _, mu, var = cnp(f.context, f.targets[0], eval=True)
plot_functions(
    f.targets[0].cpu(),
    f.targets[1].cpu(),
    f.context[0].cpu(),
    f.context[1].cpu(),
    mu.cpu(),
    var.cpu(),
)
