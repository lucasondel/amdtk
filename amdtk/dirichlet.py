"""
Implementation of the Dirichlet distribution prior.

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

import numpy as np
import theano
import theano.tensor as T
from scipy.special import digamma
from .efd import EFDPrior


class Dirichlet(EFDPrior):
    """Dirichlet Distribution."""

    def __init__(self, prior_counts):
        """Initialize a Dirichlet Distribution.

        Parameters
        ----------
        prior_counts : array_like
            Prior counts for each category (i.e. dimension of the
            random variable).

        """
        # Natural parameters.
        self._np1 = theano.shared(
            np.asarray(prior_counts - 1, dtype=theano.config.floatX),
            borrow=True
        )
        self._natural_params = self._np1.get_value()

        # Compile the model.
        self._build()

    def _build(self):
        self._natural_params = self._np1.get_value()

        log_Z, self._log_partition_func = \
            Dirichlet._get_log_partition_func(self._np1)
        self._log_partition = self._log_partition_func()


        self._grad_log_partition_func = \
            Dirichlet._get_grad_log_partition_func(log_Z, self._np1)
        self._grad_log_partition = self._grad_log_partition_func()

    @staticmethod
    def _get_log_partition_func(np1):
        log_Z = T.sum(T.gammaln(np1 + 1.)) - T.gammaln(T.sum(np1 + 1))
        return log_Z, theano.function([], log_Z)

    @staticmethod
    def _get_grad_log_partition_func(log_Z, np1):
        gradients = T.grad(log_Z, np1)
        return theano.function([], outputs=gradients)

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {'np1': self._np1.get_value()}

    @staticmethod
    def load_from_dict(model_data):
        model = Dirichlet.__new__(Dirichlet)
        np1 = theano.shared(model_data['np1'], borrow=True)
        model._np1 = np1
        model._build()

        return model

    # EFDPrior interface implementation.
    # -----------------------------------------------------------------

    @property
    def natural_params(self):
        return self._natural_params

    @natural_params.setter
    def natural_params(self, value):
        self._natural_params = value
        self._np1.set_value(value)
        self._log_partition = self._log_partition_func()
        self._grad_log_partition = self._grad_log_partition_func()

    @property
    def log_partition(self):
        return self._log_partition

    @property
    def grad_log_partition(self):
        return self._grad_log_partition

    # ------------------------------------------------------------------

