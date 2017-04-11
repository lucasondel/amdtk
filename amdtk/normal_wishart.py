"""
Implementation of a Normal-Wishart density prior.

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

# NOTE
# ----
# phi(mean, precision) = [
#     -(1/2) precision
#     precision * mean
#     -(1/2) mean * precision * mean
#     (1/2) ln det(precision)
# ]
#
# natural_params(kappa, mean, pmean, dfree) = [
#     pmean^(-1) + kappa mean mean^T
#     kappa * mean
#     kappa
#     (dfree - dim)
# ]


import numpy as np
import theano
import theano.tensor as T
from .efd import EFDPrior


class NormalWishart(EFDPrior):
    """Normal-Wishart density prior."""

    def __init__(self, mean, kappa, pmean,
                 dfree):
        """Initialize a Normal-Wishart Distribution.

        Parameters
        ----------
        mean : array_like
            Mean of the Normal density.
        kappa : float
            Scale of the precision matrix of the Normal density.
        pmean : 2D array_like
            Mean of the precision matrix.
        dfree : float
            Degree of freedom of the Wishart density. Should be greater
            than D - 1 with D being the dimension of "precision_mean".

        """
        # Dimension of the random variable of the Normal distribution.
        self.dim = len(mean)

        # Natural parameters.
        np1 = theano.shared(
                np.asarray(kappa * mean[:, None].dot(mean[None, :]) +
                           np.linalg.inv(pmean),
                           dtype=theano.config.floatX),
                borrow=True
        )
        np2 = theano.shared(
                np.asarray(kappa * mean, dtype=theano.config.floatX),
                borrow=True
        )
        np3 = theano.shared(
                np.asarray(kappa, dtype=theano.config.floatX),
                borrow=True
        )
        np4 = theano.shared(
                np.asarray(dfree - self.dim, dtype=theano.config.floatX),
                borrow=True
        )
        self._nparams = [np1, np2, np3, np4]
        self._natural_params = np.hstack([
            np1.get_value().flatten(),
            np2.get_value(),
            np3.get_value(),
            np4.get_value()
        ])

        # Symbolic expression of the log-partition function.
        idxs = np.arange(self.dim) + 1
        W = T.nlinalg.matrix_inverse(np1 - (1./np3) * T.outer(np2, np2))
        log_Z = .5 * (np4 + self.dim) * T.log(T.nlinalg.det(W))
        log_Z += .5 * (np4 + self.dim) * self.dim * np.log(2)
        log_Z += .5 * self.dim * (self.dim - 4)
        log_Z += T.sum(T.gammaln(.5 * (np4 + self.dim + 1 - idxs)))
        log_Z += -.5 * self.dim * T.log(np3)

        # Log-partition function and its gradient.
        self._log_partition_func = theano.function([], log_Z)
        self._log_partition = self._log_partition_func()

        gradients = T.grad(log_Z, self._nparams)
        self._grad_log_partition_func = theano.function(
            [], outputs=gradients
        )
        grads = self._grad_log_partition_func()
        self._grad_log_partition = np.hstack([
            grads[0].flatten(),
            grads[1],
            grads[2],
            grads[3]
        ])

    # EFDPrior interface.
    # ------------------------------------------------------------------

    @property
    def natural_params(self):
        return self._natural_params

    @natural_params.setter
    def natural_params(self, value):
        self._natural_params = value

        # Update the log-partition and its gradient.
        np1 = value[:self.dim**2].reshape(self.dim, self.dim)
        self._nparams[0].set_value(np1)
        self._nparams[1].set_value(value[self.dim**2:(self.dim**2) + self.dim])
        self._nparams[2].set_value(value[-2])
        self._nparams[3].set_value(value[-1])
        self._log_partition = self._log_partition_func()
        grads = self._grad_log_partition_func()
        self._grad_log_partition = np.hstack([
            grads[0].flatten(),
            grads[1],
            grads[2],
            grads[3]
        ])

    @property
    def log_partition(self):
        return self._log_partition

    @property
    def grad_log_partition(self):
        return self._grad_log_partition

    # ------------------------------------------------------------------
