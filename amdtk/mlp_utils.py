
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
import numpy as np
import theano
import theano.tensor as T


def _linear(x):
    """Linear activation. Do nothing on the input."""
    return x


# Possible activation for the hidden units.
ACTIVATIONS = {
    'softmax': T.nnet.softmax,
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
        self.activation = str(activation)

    def __str__(self):
        return '"' + self.activation + '" is not one of the pre-defined " \
               "activations: "' + '", "'.join(ACTIVATIONS.keys()) + '"'


def _init_weights_matrix(dim_in, dim_out, activation, borrow=True):
    val = np.sqrt(6. / (dim_in + dim_out))
    if activation == 'sigmoid':
        retval = 4 * np.random.uniform(low=-val, high=val,
                                       size=(dim_in, dim_out))
    elif activation == 'tanh':
        retval = np.random.uniform(low=-val, high=val,
                                   size=(dim_in, dim_out))
    elif (activation == 'relu' or activation == 'linear' or
         activation == 'softmax'):
        retval = np.random.normal(0., 0.01, size=(dim_in, dim_out))
    else:
        raise UnkownActivationError(activation)

    return theano.shared(np.asarray(retval, dtype=theano.config.floatX),
                         borrow=borrow)


def init_residual_weights_matrix(dim_in, dim_out, borrow=True):
    """Partial isometry initialization."""
    if dim_out == dim_in:
        weights = np.identity(dim_in)
    else:
        d = max(dim_in, dim_out)
        weights = np.linalg.qr(np.random.randn(d,d))[0][:dim_in,:dim_out]
    return theano.shared(np.asarray(weights, dtype=theano.config.floatX),
                         borrow=borrow)


def _init_bias(dim, borrow=True):
    return theano.shared(np.zeros(dim, dtype=theano.config.floatX) + .01,
                                  borrow=borrow)


class LogisticRegressionLayer(object):

    def __init__(self, inputs, dim_in, dim_out, activation):
        self.inputs = inputs
        self.dim_in = dim_in
        self.dim_out = dim_out
        weights = _init_weights_matrix(dim_in, dim_out, activation)
        bias = _init_bias(dim_out)
        self.outputs = ACTIVATIONS[activation](T.dot(inputs, weights) + bias)
        self.params = [weights, bias]

class StdLayer(object):

    def __init__(self, inputs, dim_in, dim_out, activation):
        self.inputs = inputs
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        weights = _init_weights_matrix(dim_in, dim_out, activation)
        bias = _init_bias(dim_out)
        self.outputs = ACTIVATIONS[activation](T.dot(inputs, weights) + bias)
        self.params = [weights, bias]


class GaussianLayer(object):

    def __init__(self, inputs, dim_in, dim_out, activation):
        self.inputs = inputs
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        shared_layer = StdLayer(inputs, dim_in, 2 * dim_out, activation)

        self.mean, raw_logvar = \
            theano.tensor.split(shared_layer.outputs, [dim_out, dim_out], 2,
                                axis=-1)
        #self.var = T.log(1 + T.exp(raw_logvar))
        self.var = T.exp(raw_logvar)

        self.params = shared_layer.params
        self.outputs = self.mean


# Possible layer types.
LAYER_TYPES = {
   'standard': StdLayer,
   'gaussian': GaussianLayer
}


class NeuralNetwork(object):

    def __init__(self, structure, residuals, inputs=None):
        if inputs is None:
            self.inputs = T.matrix(dtype=theano.config.floatX)
        else:
            self.inputs = inputs

        # Build the neural network.
        self.layers = []
        self.params = []
        current_inputs = self.inputs
        for layer_type, dim_in, dim_out, activation in structure:
            self.layers.append(LAYER_TYPES[layer_type](current_inputs, dim_in,
                                                       dim_out, activation))
            self.params += self.layers[-1].params
            current_inputs = self.layers[-1].outputs

        # Add the residual connections.
        for residual_in, residual_out in residuals:
            dim_in = self.layers[residual_in].dim_in
            dim_out = self.layers[residual_out].dim_out
            weights = init_residual_weights_matrix(dim_in, dim_out)
            self.params += [weights]
            self.layers[residual_out].outputs += \
                T.dot(self.layers[residual_in].inputs, weights)

        self.outputs = self.layers[-1].outputs


class MLP(NeuralNetwork):

    def __init__(self, structure, residuals, inputs):
        NeuralNetwork.__init__(self, structure, residuals, inputs)
        self.log_pred = T.log(self.layers[-1].outputs)

        # Build the functions.
        self.forward = theano.function(
            inputs=[self.inputs],
            outputs=[self.log_pred]
        )


class GaussianNeuralNetwork(NeuralNetwork):

    def __init__(self, structure, residuals, inputs=None):
        NeuralNetwork.__init__(self, structure, residuals, inputs)
        self.mean = self.layers[-1].outputs
        self.var = self.layers[-1].var

        # Noise variable for the reparameterization trick.
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams()
        else:
            srng = T.shared_randomstreams.RandomStreams()
            self.eps = srng.normal(self.mean.shape)

        # Latent variable.
        self.sample = self.mean + T.sqrt(self.var) * self.eps

        # Build the functions.
        self.forward = theano.function(
            inputs=[self.inputs],
            outputs=[self.mean, self.var]
        )


class MLPold(metaclass=abc.ABCMeta):
    """Base class for MLP objects."""

    @staticmethod
    def init_layer_params(dim_in, dim_out, scale):
        """Initialize a weights matrix and bias vector."""
        #weights = np.random.randn(dim_in, dim_out)
        weights = np.random.uniform(
            low=-np.sqrt(6. / (dim_in + dim_out)),
            high=np.sqrt(6. / (dim_in + dim_out)),
            size=(dim_in, dim_out)
        )
        bias = np.zeros(dim_out)
        return [scale * weights, bias]

    @staticmethod
    def forward(params, activation, data):
        """Forward an input matrix through the Gaussian residual MLP."""
        inputs = data
        for idx in range(0, len(params), 2):
            weights = params[idx]
            bias = params[idx + 1]
            outputs = np.dot(inputs, weights) + bias
            inputs = activation(outputs)
        return inputs

class GaussianMLP(MLP):
    """Static implementation of a Gaussian residual MLP."""

    @staticmethod
    def create(dim_in, dim_out, dim_h, n_layers, scale, precision):
        """Create a Gaussian residual MLP."""
        params = MLP.init_layer_params(dim_in, dim_h, scale)
        for idx in range(n_layers - 1):
            params += MLP.init_layer_params(dim_h, dim_h, scale)
        params += MLP.init_layer_params(dim_h, 2 * dim_out, scale)

        # Initialize the precision.
        params[-1][dim_out:] -= np.log(precision)
        return params

    @staticmethod
    def extract_params(params):
        """Extract the different part of the Gaussian MLP."""
        return [params[-2:], params[:-2]]

    @staticmethod
    def forward(params, activation, data):
        """Forward an input matrix through the Gaussian residual MLP."""
        linear_params, h_params = GaussianMLP.extract_params(params)
        inputs = MLP.forward(h_params, activation, data)
        outputs = np.dot(inputs, linear_params[0]) + linear_params[1]
        mean, logvar = np.split(outputs, 2, axis=-1)
        #var = np.log(1 + np.exp(logvar))
        var = np.exp(logvar)
        return mean, var

    @staticmethod
    def natural_params(mean, var):
        np1 = - 1 / (2 * var)
        np2 = mean / var
        return np1, np2

    @staticmethod
    def std_params(np1, np2):
        var = -1 / (2 * np1)
        mean = np2 * var
        return mean, var


class GaussianResidualMLP(GaussianMLP):
    """Static implementation of a Gaussian residual MLP."""

    @staticmethod
    def init_residual_params(dim_in, dim_out):
        """Partial isometry initialization."""
        if dim_out == dim_in:
            return [np.identity(dim_in)]
        d = max(dim_in, dim_out)
        weights = np.linalg.qr(np.random.randn(d,d))[0][:dim_in,:dim_out]
        return [weights]

    @staticmethod
    def create(dim_in, dim_out, dim_h, n_layers, scale, precision):
        """Create a Gaussian residual MLP."""
        params = GaussianMLP.create(dim_in, dim_out, dim_h, n_layers, scale,
                                    precision)
        #params += GaussianResidualMLP.init_residual_params(dim_in, dim_out)
        return params

    @staticmethod
    def forward(params, activation, data):
        """Forward an input matrix through the Gaussian residual MLP."""
        #gauss_params, res_params = params[:-1], params[-1]
        gauss_params = params
        mean, var = GaussianMLP.forward(gauss_params, activation, data)
        #mean = mean + np.dot(data, res_params)
        mean = mean + data
        return mean, var

    @staticmethod
    def _kl_div(mean_post, var_post, mean_prior, var_prior):
        """KL divergence between the posterior and the prior."""
        kl_div = (.5 * (mean_prior - mean_post)**2) / var_prior
        ratio = var_post / var_prior
        kl_div = kl_div + .5 * (ratio - 1 - np.log(ratio))
        return np.sum(kl_div, axis=1)

    @staticmethod
    def sample(params, activation, inputs, prior_params=None):
        """Sample from the Gaussian residual MLP."""
        mean, var = GaussianResidualMLP.forward(params, activation, inputs)
        eps = np.random.randn(*inputs.shape)
        samples = mean + np.sqrt(var) * eps
        if prior_params is not None:
            kl_div = GaussianResidualMLP._kl_div(mean, var, prior_params[0],
                                                 prior_params[1])
            return samples, kl_div

        return samples

    @staticmethod
    def sample_np(params, activation, inputs, exp_np1, exp_np2):
        """Sample from the Gaussian residual MLP with nat. params."""
        mean, var = GaussianResidualMLP.forward(params, activation, inputs)
        np1, np2 = GaussianMLP.natural_params(mean, var)
        prior_mean, prior_var = GaussianMLP.std_params(exp_np1, exp_np2)
        post_mean, post_var = GaussianMLP.std_params(np1 + exp_np1,
                                                     np2 + exp_np2)
        eps = np.random.randn(*inputs.shape)
        samples = post_mean + np.sqrt(post_var) * eps
        kl_div = GaussianResidualMLP._kl_div(post_mean, post_var, prior_mean,
                                             prior_var)
        return samples, kl_div


    @staticmethod
    def llh(params, activation, inputs, targets):
        """Log-likelihood of the Gaussian residual MLP."""
        mean, var = GaussianResidualMLP.forward(params, activation, inputs)
        N, D = targets.shape
        retval = -.5 * np.sum(np.log(var), axis=1)
        retval = retval - .5 * np.sum(((targets - mean) ** 2) / var, axis=1)
        return retval

