"""
Utilities for MLP object.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import abc
from collections import OrderedDict
import autograd.numpy as np


def relu(x):
    """Rectified Linear activation."""
    return np.maximum(x, 0)


def init_weights_matrix(dim_in, dim_out, scale=1.):
    return np.random.normal(0, scale * 0.01, size=(dim_in, dim_out))


def init_bias(dim, shift=0.):
    return np.zeros(dim, dtype=float) + shift


def gauss_nnet_forward(params, x):
    logvar_b = params[-1]
    logvar_w = params[-2]
    mean_b = params[-3]
    mean_w = params[-4]


class HiddenLayer(object):
    """Hidden Layer of a Neural Net structure."""

    def __init__(self, dim_in, dim_out):
        """Initialize a hidden layer of a MLP.

        Parameters
        ----------
        dim_in : int
            Dimension of the input.
        dim_out : int
            Dimension of the output.

        """
        # Initialize the weight matrix and the bias vector.
        weights = _init_weights_matrix(dim_in, dim_out, activation)
        bias = _init_bias(dim_out)

        # Parameters to update.
        self.params = [weights, bias]

    def forward(self, x):
        """Forward an input through the hidden layer."""


class MLP(metaclass=abc.ABCMeta):
    """Abstract base class for MLP object."""

    def __init__(self, dim_in, n_layers, n_units):
        """Create and initialize the basic structure of the MLP.

        Parameters
        ----------
        input : theano variable
            Input to the MLP.
        dim_in : int
            Dimension of the input.
        n_layers : int
            Number of hidden layers.
        n_units : int
            Number of units per layer.
        activation : str
            Name of the activation for the hiddent units. Can be one of
            the following:
              * sigmoid
              * tanh
              * relu
              * linear

        """
        # Parameters to update during the training.
        self.params = []

        # Symbolic variable of the input.
        self.input = input

        # Create the first hidden layer.
        self.layers = [HiddenLayer(
            input=self.input,
            dim_in=dim_in,
            dim_out=n_units,
            activation=activation
        )]
        self.params += self.layers[0].params

        # Add other layer if there are any.
        for i in range(1, n_layers):
            self.layers.append(HiddenLayer(
                input=self.layers[-1].output,
                dim_in=n_units,
                dim_out=n_units,
                activation=activation))
            self.params += self.layers[-1].params

        # Define the output as the one of the last layer. This may
        # be overriden by subclasses.
        self.output = self.layers[-1].output


class MLPGaussian(metaclass=abc.ABCMeta):
    """Abstract base class for MLP with Gaussian output."""

    def __init__(self, input, dim_in, dim_out, n_layers, n_units, activation):
        """Create and initialize the basic structure of the MLP.

        Parameters
        ----------
        input : theano variable
            Input to the MLP.
        dim_in : int
            Dimension of the input.
        dim_out : int
            Dimension of the output.
        n_layers : int
            Number of hidden layers.
        n_units : int
            Number of units per layer.
        activation : str
            Name of the activation for the hiddent units. Can be one of
            the following:
              * sigmoid
              * tanh
              * relu
              * linear

        """
        # Initialize the MLP structure.
        MLP.__init__(self, input, dim_in, n_layers, n_units, activation)

        # Create the output final layer (mean and log of the variance).
        self.mean_layer = HiddenLayer(
            input=self.layers[-1].output,
            dim_in=n_units,
            dim_out=dim_out,
            activation='linear'
        )
        self.params += self.mean_layer.params
        self.mean = self.mean_layer.output

        self.log_var_layer = HiddenLayer(
            input=self.layers[-1].output,
            dim_in=n_units,
            dim_out=dim_out,
            activation='linear'
        )
        self.params += self.log_var_layer.params
        self.log_var = self.log_var_layer.output

        # Defines the output as the mean the log variance concatenated.
        self.output = T.concatenate([self.mean, self.log_var])

