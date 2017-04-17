"""
Various implementation of the Stochastic Gradient Ascent.

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
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T

ADAM_EPS = 1e-8


class AdamSGATheano(metaclass=abc.ABCMeta):
    """Base class for ADAM gradient models."""

    def __init__(self, inputs, params, gradients):
        """Initialize the parameters of the gradient ascent.

        Parameters
        ----------
        inputs : list
            List of theano variables to compute the objective function.
        objective : theano variable
            Objective function to maximize.
        params : list
            List of parameters to update.
        gradient : list
            List of the gradients of the objective function for each
            parameter.

        """
        self.pmean = []
        self.pvar = []
        for param in params:
            self.pmean.append(
                theano.shared(np.zeros_like(param.get_value(),
                              dtype=theano.config.floatX))
            )
            self.pvar.append(
                theano.shared(np.zeros_like(param.get_value(),
                              dtype=theano.config.floatX))
            )

        # Parameters of the training.
        b1 = T.scalar()
        b2 = T.scalar()
        lrate = T.scalar()
        epoch = T.iscalar('epoch')

        # Build a function for the update.
        self.adam_sga_update = theano.function(
            inputs=gradients + [b1, b2, lrate, epoch],
            updates=self._get_adam_updates(params, gradients, b1, b2, lrate,
                                           epoch)
        )

    def _get_adam_updates(self, params, gradients, b1, b2, lrate, epoch):
        updates = OrderedDict()

        gamma = T.sqrt(1 - b2**epoch) / (1 - b1**epoch)
        values_iterable = zip(params, gradients, self.pmean, self.pvar)

        for parameter, gradient, pmean, pvar in values_iterable:
            new_m = b1 * pmean + (1. - b1) * gradient
            new_v = b2 * pvar + (1. - b2) * (gradient**2)

            updates[parameter] = parameter + lrate * gamma * \
                new_m / (T.sqrt(new_v) + ADAM_EPS)

            updates[pmean] = new_m
            updates[pvar] = new_v

        return updates

