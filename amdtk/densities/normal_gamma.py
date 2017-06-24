
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


import theano
import theano.tensor as T
import numpy as np
from .efd import EFDPrior


def _log_partition_symfunc():
    natural_params = T.vector()
    size = natural_params.shape[0] // 4
    np1, np2, np3, np4 = T.split(natural_params, 4 * [size], 4)

    log_Z = T.sum(T.gammaln(.5 * (np4 + 1)))
    log_Z += T.sum(- .5 * (np4 + 1) * T.log(.5 * (np1 - (np2 ** 2) / np3)))
    log_Z += T.sum(-.5 * T.log(np3))

    func = theano.function([natural_params], log_Z)
    grad_func = theano.function([natural_params],
                                T.grad(T.sum(log_Z), natural_params))
    return func, grad_func

_lp_func, _grad_lp_func = _log_partition_symfunc()


class NormalGamma(EFDPrior):
    """Normal-Gamma density prior."""

    @staticmethod
    def _log_partition_func(natural_params):
        return _lp_func(natural_params)

    @staticmethod
    def _grad_log_partition_func(natural_params):
        return _grad_lp_func(natural_params)

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
        EFDPrior.__init__(self)

        self.mean = mean
        self.kappa = kappa
        self.rate = rate
        self.scale = scale

        self._fixed_variance = False

        self.natural_params = np.hstack([
            np.asarray(kappa * (mean ** 2) + 2 * scale, dtype=float),
            np.asarray(kappa * mean, dtype=float),
            np.asarray(kappa, dtype=float),
            np.asarray(2 * (rate - 1./2), dtype=float)
        ])

    @property
    def fixed_variance(self):
        return self._fixed_variance

    @fixed_variance.setter
    def fixed_variance(self, value):
        self._fixed_variance = value

    # EFDPrior interface implementation.
    # -----------------------------------------------------------------

    def correct_np_value(self, value):
        # Separate the natural parameters.
        r_value = value.reshape(4, -1)

        # Convert them to the standard parameters.
        kappa = r_value[2]
        mean = r_value[1] / kappa
        rate = (r_value[3] / 2) + .5
        scale = (r_value[0] - kappa * (mean ** 2)) / 2

        if self.fixed_variance:
            # If the variance is fixed don't update it.
            kappa = self.kappa
            rate = self.rate
            scale = self.scale
        else:
            # Project back the parameters into their domain.
            kappa = np.maximum(kappa, 1)
            rate = np.maximum(rate, 1)
            scale = np.maximum(scale, 1)

        # Return the corrected standard parameters in their natural
        # form.
        return np.hstack([
            kappa * (mean ** 2) + 2 * scale,
            kappa * mean,
            kappa,
            2 * (rate - 1./2)
        ])

    # -----------------------------------------------------------------


