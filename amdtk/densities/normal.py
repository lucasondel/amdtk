
"""
Normal density with conjugate prior.

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
from .efd import EFDPrior, EFDLikelihood


class Normal(object):
    """Normal distribution with Normal-Wishart prior."""

    @staticmethod
    def get_sufficient_stats(data):
        """Return the sufficient statistics of the Normal distribution.

        For the case of full covariance matrix, the sufficient
        statistics are: (x, vec(xx^T)^T.

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
        EFDLikelihood.__init__(self, prior, posterior)


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
        EFDLikelihood.__init__(self, prior, posterior)

