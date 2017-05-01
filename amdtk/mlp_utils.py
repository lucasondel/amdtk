
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
import autograd.numpy as np


class MLP(metaclass=abc.ABCMeta):
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
        params[-1][dim_out:] += precision
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
        var = np.log(1 + np.exp(logvar))
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
        params += GaussianResidualMLP.init_residual_params(dim_in, dim_out)
        return params

    @staticmethod
    def forward(params, activation, data):
        """Forward an input matrix through the Gaussian residual MLP."""
        gauss_params, res_params = params[:-1], params[-1]
        mean, var = GaussianMLP.forward(gauss_params, activation, data)
        mean = mean + np.dot(data, res_params)
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

