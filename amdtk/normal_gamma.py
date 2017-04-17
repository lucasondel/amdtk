
"""
Implementation of a Normal-Gamma density prior.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

# NOTE
# ----
# phi(mean, precision) = [
#     - precision / 2
#     precision * mean
#     -precision * (mean ** 2) / 2
#     (1/2) * ln precision
# ]
#
# natural_params(kappa, mean, rate, scale) = [
#     kappa * (mean ** 2) + 2 * scale
#     kappa * mean
#     kappa
#     2 * (rate - 1/2)
# ]
#
# log_partition(kappa, mean, rate, scale) =
#   gammaln(rate) - rate * log(scale) - .5 * log(kappa)
#
# log_partition(np1, np2, np3, np4) =
#   gammaln(.5 * (np4 + 1)) - .5 * (np4 + 1) log(.5 *
#       (np1 - (np2 ** 2) / np3)) - .5 * log(np3)
#


import numpy as np
import theano
import theano.tensor as T
from .efd import EFDPrior


class NormalGamma(EFDPrior):
    """Normal-Gamma density prior."""

    def __init__(self, mean, kappa, rate, scale):
        """Initialize a Normal-Gamma Distribution.

        Parameters
        ----------
        mean : numpy.ndarray
            Mean of the Normal density.
        kappa : float
            Scale of the precision Normal density.
        rate : float
            Rate parameter of the Gamma density.
        scale : float
            scale parameter of the Gamma density.

        """
        # Natural parameters.
        np1 = theano.shared(
                np.asarray(kappa * (mean ** 2) + 2 * scale,
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
                np.asarray(2 * (rate - 1./2), dtype=theano.config.floatX),
                borrow=True
        )
        self._nparams = [np1, np2, np3, np4]

        # Compile the model.
        self._build()

    def _build(self):
        self._natural_params = np.hstack([
            self._nparams[0].get_value(),
            self._nparams[1].get_value(),
            self._nparams[2].get_value(),
            self._nparams[3].get_value()
        ])

        log_Z, self._log_partition_func = \
            NormalGamma._get_log_partition_func(self._nparams)
        self._log_partition = self._log_partition_func()


        self._grad_log_partition_func = \
            NormalGamma._get_grad_log_partition_func(log_Z, self._nparams)
        self._grad_log_partition = self._grad_log_partition_func()

    @staticmethod
    def _get_log_partition_func(nparams):
        np1, np2, np3, np4 = nparams
        log_Z = T.gammaln(.5 * (np4 + 1))
        log_Z += - .5 * (np4 + 1) * T.log(.5 * (np1 - (np2 ** 2) / np3))
        log_Z += -.5 * T.log(np3)
        log_Z = T.sum(log_Z)
        return log_Z, theano.function([], log_Z)

    @staticmethod
    def _get_grad_log_partition_func(log_Z, nparams):
        gradients = T.grad(log_Z, nparams)
        return theano.function([], outputs=T.concatenate(gradients))

    # PersistentModel interface implementation.
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            'np1': self._nparams[0].get_value(),
            'np2': self._nparams[1].get_value(),
            'np3': self._nparams[2].get_value(),
            'np4': self._nparams[3].get_value(),
        }

    @staticmethod
    def load_from_dict(model_data):
        model = NormalGamma.__new__(NormalGamma)
        np1 = theano.shared(model_data['np1'], borrow=True)
        np2 = theano.shared(model_data['np2'], borrow=True)
        np3 = theano.shared(model_data['np3'], borrow=True)
        np4 = theano.shared(model_data['np4'], borrow=True)
        model._nparams = [np1, np2, np3, np4]
        model._build()

        return model

    # EFDPrior interface implementation.
    # ------------------------------------------------------------------

    @property
    def natural_params(self):
        return self._natural_params

    @natural_params.setter
    def natural_params(self, value):
        self._natural_params = value

        # Update the log-partition and its gradient.
        nparams = value.reshape(4, -1)
        self._nparams[0].set_value(nparams[0])
        self._nparams[1].set_value(nparams[1])
        self._nparams[2].set_value(nparams[2])
        self._nparams[3].set_value(nparams[3])
        self._log_partition = self._log_partition_func()
        self._grad_log_partition = self._grad_log_partition_func()

    @property
    def log_partition(self):
        return self._log_partition

    @property
    def grad_log_partition(self):
        return self._grad_log_partition

    # ------------------------------------------------------------------

