
"""
Dirichlet density prior.

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

import theano
import theano.tensor as T
import numpy as np
from .efd import EFDPrior


def _log_partition_symfunc():
    natural_params = T.vector()
    log_Z = T.sum(T.gammaln(natural_params + 1.)) -\
        T.gammaln(T.sum(natural_params + 1))

    func = theano.function([natural_params], log_Z)
    grad_func = theano.function([natural_params],
                                T.grad(T.sum(log_Z), natural_params))
    return func, grad_func


_lp_func, _grad_lp_func = _log_partition_symfunc()


class Dirichlet(EFDPrior):
    """Dirichlet Distribution."""

    @staticmethod
    def _log_partition_func(natural_params):
        return _lp_func(natural_params)

    @staticmethod
    def _grad_log_partition_func(natural_params):
        return _grad_lp_func(natural_params)

    def __init__(self, prior_counts):
        EFDPrior.__init__(self)
        self.natural_params = np.asarray(prior_counts - 1, dtype=float)

