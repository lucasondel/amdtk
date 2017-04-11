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
import numpy as np
import theano
import theano.tensor as T


def _linear(x):
    """Linear activation. Do nothing on the input."""
    return x


# Possible activation for the hidden units.
ACTIVATIONS = {
    'sigmoid': T.nnet.sigmoid,
    'tanh': T.tanh,
    'relu': T.nnet.relu,
    'linear': _linear
}


class MLPError(Exception):
    """Base class for exceptions in this module."""

    pass


class UnkownActivationError(MLPError):
    """Raised when the given activation is not known."""

    def __init__(self, activation):
        """Initialize the exception.

        Parameters
        ----------
        activation : str
            Name of the activation.

        """
        self.activation = str(activation)

    def __str__(self):
        """Description of the exception.

        Returns
        -------
        msg : str
            Error message.

        """
        return '"' + self.activation + '" is not one of the pre-defined " \
            "activations: "' + '", "'.join(ACTIVATIONS.keys()) + '"'


def _init_weights_matrix(dim_in, dim_out, activation, borrow=True):
    if activation == 'sigmoid':
        retval = 4 * np.random.uniform(
            low=-np.sqrt(6. / (dim_in + dim_out)),
            high=np.sqrt(6. / (dim_in + dim_out)),
            size=(dim_in, dim_out)
        )
    elif activation == 'tanh':
        retval = np.random.uniform(
            low=-np.sqrt(6. / (dim_in + dim_out)),
            high=np.sqrt(6. / (dim_in + dim_out)),
            size=(dim_in, dim_out)
        )
    elif activation == 'relu':
        retval = np.random.normal(
            0.,
            .01,
            size=(dim_in, dim_out)
        )
    elif activation == 'linear':
        retval = np.random.normal(
            0.,
            .01,
            size=(dim_in, dim_out)
        )
    else:
        raise UnkownActivationError(activation)

    return theano.shared(np.asarray(retval, dtype=theano.config.floatX),
                         borrow=borrow)


def _init_bias(dim, borrow=True):
    return theano.shared(np.zeros(dim, dtype=theano.config.floatX),
                         borrow=borrow)


class HiddenLayer(object):
    """Hidden Layer of a Neural Net structure."""

    def __init__(self, input, dim_in, dim_out, activation):
        """Initialize a hidden layer of a MLP.

        Parameters
        ----------
        input : theano variable
            Symbolic variatble of the input to the hidden layer.
        dim_in : int
            Dimension of the input.
        dim_out : int
            Dimension of the output.
        activation : function
            Non-linear actication of the hidden layer.

        """
        # Initialize the weight matrix and the bias vector.
        weights = _init_weights_matrix(dim_in, dim_out, activation)
        bias = _init_bias(dim_out)

        # Symbolic computation of the hidden layer.
        self.output = ACTIVATIONS[activation](T.dot(input, weights) + bias)

        # Parameters to update.
        self.params = [weights, bias]


class MLP(metaclass=abc.ABCMeta):
    """Abstract base class for MLP object."""

    def __init__(self, input, dim_in, n_layers, n_units, activation):
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
        # be override by subclasses.
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
            activation='linear')
        values = np.random.normal(0., 1., (n_units, dim_out))
        values = np.asarray(values, dtype=theano.config.floatX)
        self.mean_layer.params[0].set_value(values)
        self.params += self.mean_layer.params
        self.mean = self.mean_layer.output

        self.log_var_layer = HiddenLayer(
            input=self.layers[-1].output,
            dim_in=n_units,
            dim_out=dim_out,
            activation='linear')
        values = np.asarray(np.zeros(dim_out) - 3., dtype=theano.config.floatX)
        self.log_var_layer.params[1].set_value(values)
        self.params += self.log_var_layer.params
        self.log_var = self.log_var_layer.output

        # Defines the output as the mean the log variance concatenated.
        self.output = T.concatenate([self.mean, self.log_var])
