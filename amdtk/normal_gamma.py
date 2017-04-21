
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


import autograd.numpy as np
from autograd.scipy.special import gammaln
from autograd import grad
from .efd import EFDPrior


def _log_partition_func(np1, np2, np3, np4):
    log_Z = np.sum(gammaln(.5 * (np4 + 1)))
    log_Z += np.sum(- .5 * (np4 + 1) * np.log(.5 * (np1 - (np2 ** 2) / np3)))
    log_Z += np.sum(-.5 * np.log(np3))
    return log_Z


_grad_log_partition_func0 = grad(_log_partition_func, argnum=0)
_grad_log_partition_func1 = grad(_log_partition_func, argnum=1)
_grad_log_partition_func2 = grad(_log_partition_func, argnum=2)
_grad_log_partition_func3 = grad(_log_partition_func, argnum=3)


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
        self._nparams = [
            np.asarray(kappa * (mean ** 2) + 2 * scale, dtype=float),
            np.asarray(kappa * mean, dtype=float),
            np.asarray(kappa, dtype=float),
            np.asarray(2 * (rate - 1./2), dtype=float)
        ]

        # Compile the model.
        self._build()

    def _build(self):
        self._natural_params = np.hstack([
            self._nparams[0],
            self._nparams[1],
            self._nparams[2],
            self._nparams[3]
        ])

        self._log_partition = _log_partition_func(*self._nparams)

        self._grad_log_partition = np.r_[
            _grad_log_partition_func0(*self._nparams),
            _grad_log_partition_func1(*self._nparams),
            _grad_log_partition_func2(*self._nparams),
            _grad_log_partition_func3(*self._nparams),
        ]

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'np1': self._nparams[0],
            'np2': self._nparams[1],
            'np3': self._nparams[2],
            'np4': self._nparams[3],
        }

    @staticmethod
    def load_from_dict(model_data):
        model = NormalGamma.__new__(NormalGamma)
        model._nparams = [
            model_data['np1'],
            model_data['np2'],
            model_data['np3'],
            model_data['np4']
        ]
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

        # Update the log-partition and its gradient.
        nparams = value.reshape(4, -1)
        self._nparams = [np.asarray(param, dtype=float) for param in nparams]
        self._log_partition = _log_partition_func(*self._nparams)

        # Update the expected value of the sufficient statistics.
        self._grad_log_partition = np.r_[
            _grad_log_partition_func0(*self._nparams),
            _grad_log_partition_func1(*self._nparams),
            _grad_log_partition_func2(*self._nparams),
            _grad_log_partition_func3(*self._nparams),
        ]

    @property
    def log_partition(self):
        return self._log_partition

    @property
    def grad_log_partition(self):
        return self._grad_log_partition

    # -----------------------------------------------------------------

