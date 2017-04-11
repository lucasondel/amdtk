"""
Implementation of a Bayesian Normal density with a Normal-Wishart prior.

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

import numpy as np
import theano
import theano.tensor as T
from .efd import EFDPrior, EFDLikelihood


class Normal(object):
    """Normal distribution with Normal-Wishart prior."""

    @staticmethod
    def get_sufficient_stats(data):
        """Return the sufficient statistics of the Normal distribution.

        The statistics will be different if the Normal is full
        covariance of not.

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        s_stats : numpy.ndarray
            Sufficient statistics of the model.

        """
        length = len(data)

        # Quadratic expansion of the data.
        data_quad = np.array([a[:, None].dot(a[None, :]).ravel()
                              for a in data])

        return np.c_[data_quad, data, np.ones(length),
                     np.ones(length)]

    def __init__(self, prior, posterior):
        """Initialize a Normal density.

        Parameters
        ----------
        prior : :class:`NormalWishart`
            Normal-Wishart prior over the mean and precision matrix.
        posterior : :class:`NormalWishart`
            Normal-Wishart posterior over the mean and precision matrix.

        """
        self._prior = prior
        self._posterior = posterior

    @property
    def prior(self):
        """Conjugate prior."""
        return self._prior

    @property
    def posterior(self):
        """Conjugate posterior."""
        return self._posterior

    @posterior.setter
    def posterior(self, value):
        self._posterior = value


class NormalDiag(EFDLikelihood):
    """Normal distribution with diagonal covariance matrix."""

    @staticmethod
    def get_sufficient_stats(data):
        """Sufficient statistics of the Normal density.

        For the case of diagonal covariance matrix, the sufficient
        statistics are: (x, x**2)^T.

        """
        length = len(data)
        return np.c_[data**2, data, np.ones((length, 2 * data.shape[1]))]

    def __init__(self, prior, posterior):
        """Initialize a Normal density.

        Parameters
        ----------
        prior : :class:`NormalWishart`
            Normal-Wishart prior over the mean and precision matrix.
        posterior : :class:`NormalWishart`
            Normal-Wishart posterior over the mean and precision matrix.

        """
        self._prior = prior
        self._posterior = posterior

    @property
    def prior(self):
        """Conjugate prior."""
        return self._prior

    @property
    def posterior(self):
        """Conjugate posterior."""
        return self._posterior

    @posterior.setter
    def posterior(self, value):
        self._posterior = value
