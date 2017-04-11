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


class StdSGATheano(metaclass=abc.ABCMeta):
    """Base class for Stochastic Gradient Ascent engine (Theano)."""

    def __init__(self, inputs, outputs, objective, params, gradients):
        """Initialize the parameters of the gradient ascent.

        Parameters
        ----------
        inputs : list
            List of theano variables to compute the objective function.
        outputs : theano variable or list
            Output of the update function.
        objective : theano variable
            Objective function to maximize.
        params : list
            List of parameters to update.
        gradient : list
            List of the gradients of the objective function for each
            parameter.

        """
        # Parameters of the standard Stochastic Gradient Ascent.
        self._forgetting_rate = theano.shared(0.51)
        self._delay = theano.shared(0.)
        self._scale = theano.shared(1.)
        self._n_frames = theano.shared(1.)

        # Build a function for the update.
        time_step = T.iscalar('time_step')
        self.std_sga_update = theano.function(
            inputs=inputs + [time_step],
            outputs=outputs,
            updates=self._get_sga_updates(params, gradients, time_step)
        )

    @property
    def forgetting_rate(self):
        """Forgetting rate."""
        return self._forgetting_rate.get_value()

    @forgetting_rate.setter
    def forgetting_rate(self, value):
        self._forgetting_rate.set_value(value)

    @property
    def delay(self):
        """Delay."""
        return self._delay.get_value()

    @delay.setter
    def delay(self, value):
        self._delay.set_value(value)

    @property
    def scale(self):
        """Scale."""
        return self._scale.get_value()

    @scale.setter
    def scale(self, value):
        self._scale.set_value(value)

    @property
    def n_frames(self):
        """Scale."""
        return self._n_frames.get_value()

    @n_frames.setter
    def n_frames(self, value):
        """Number of frames."""
        self._n_frames.set_value(value)

    def _get_sga_updates(self, params, gradients, time_step):
        """Stochastic Gradient updates.

        Parameters
        ----------
        params : list
            List of parameters to update.
        gradients : list
            Gradient of the objective function for each parameter of
            the list.
        time_step : theano int
            Time step of the ascent.

        Returns
        -------
        updates : list
            List of the theano updates, i.e. a list of tupe
            (param, update(param)).
        """
        lrate = self._scale * \
            ((self._delay + time_step)**(-self._forgetting_rate))
        updates = []
        for i, param in enumerate(params):
            updates.append((param, param + lrate * gradients[i]))

        return updates


class AdamSGATheano(metaclass=abc.ABCMeta):
    """Base class for ADAM gradient models."""

    def __init__(self, inputs, outputs, objective, params, gradients):
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
        self._b1 = theano.shared(0.95)
        self._b2 = theano.shared(0.999)
        self._lrate = theano.shared(0.01)

        # Build a function for the update.
        epoch = T.iscalar('epoch')
        self.adam_sga_update = theano.function(
            inputs=inputs + [epoch],
            outputs=outputs,
            updates=self._get_adam_updates(params, gradients, epoch)
        )

    @property
    def b1(self):
        """b1."""
        return self._b1.get_value()

    @b1.setter
    def b1(self, value):
        self._b1.set_value(value)

    @property
    def b2(self):
        """b2."""
        return self._b2.get_value()

    @b2.setter
    def b2(self, value):
        self._b2.set_value(value)

    @property
    def lrate(self):
        """Learning rate."""
        return self._lrate.get_value()

    @lrate.setter
    def lrate(self, value):
        self._lrate.set_value(value)

    def _get_adam_updates(self, params, gradients, epoch):
        """ADAM gradients update.

        Parameters
        ----------
        params : list
            List of parameters to update.
        gradients : list
            Gradient of the objective function for each parameter of
            the list.
        epoch : theano int
            Epoch of the training.

        Returns
        -------
        updates : list
            List of the theano updates.
        """
        updates = OrderedDict()

        gamma = T.sqrt(1 - self._b2**epoch) / (1 - self._b1**epoch)

        values_iterable = zip(params, gradients, self.pmean, self.pvar)

        for parameter, gradient, pmean, pvar in values_iterable:
            new_m = self._b1 * pmean + (1. - self._b1) * gradient
            new_v = self._b2 * pvar + (1. - self._b2) * (gradient**2)

            updates[parameter] = parameter + self._lrate * gamma * \
                new_m / (T.sqrt(new_v) + ADAM_EPS)

            updates[pmean] = new_m
            updates[pvar] = new_v

        return updates


class StdSGAPython(metaclass=abc.ABCMeta):
    """Base class for Stochastic Gradient Ascent."""

    def __init__(self):
        """Initialize the parameters of the gradient ascent.

        Parameters
        ----------

        """
        # Parameters of the standard Stochastic Gradient Ascent.
        self._forgetting_rate = 0.51
        self._delay = 0.
        self._scale = 1.
        self._n_frames = 1

    @property
    def forgetting_rate(self):
        """Forgetting rate."""
        return self._forgetting_rate

    @forgetting_rate.setter
    def forgetting_rate(self, value):
        self._forgetting_rate = value

    @property
    def delay(self):
        """Delay."""
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = value

    @property
    def scale(self):
        """Scale."""
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def n_frames(self):
        """Number of frames of the whole database."""
        return self._n_frames

    @n_frames.setter
    def n_frames(self, value):
        self._n_frames = value

    @abc.abstractproperty
    def after_grad_update(self):
        """Called after each gradient update."""
        pass

    def std_sga_update(self, data, time_step):
        """Standard SGA update.

        Parameters
        ----------
        params : list
            List of parameters to update.
        gradients : list
            Gradient of the objective function for each parameter of
            the list.
        time_step : theano int
            Time step of the ascent.

        Returns
        -------
        objective : float
            Value of the objective function before the update.

        """
        # Compute the learning rate.
        lrate = self.scale * \
            ((self.delay + time_step)**(-self.forgetting_rate))

        # Compute the gradients.
        objective, grads = self.gradients(data)

        # Update the parameters.
        for idx, param in enumerate(self.params):
            param += lrate * grads[idx]

        # Post-processing.
        self.after_grad_update()

        return objective

    def std_sga_update_from_grads(self, grads, time_step):
        """Standard SGA update.

        Parameters
        ----------
        params : list
            List of parameters to update.
        grads : list
            Gradient of the objective function for each parameter of
            the list.
        time_step : theano int
            Time step of the ascent.

        Returns
        -------
        objective : float
            Value of the objective function before the update.

        """
        # Compute the learning rate.
        lrate = self.scale * \
            ((self.delay + time_step)**(-self.forgetting_rate))

        # Update the parameters.
        for idx, param in enumerate(self.params):
            param += lrate * grads[idx]

        # Post-processing.
        self.after_grad_update()


class AdamSGAPython(metaclass=abc.ABCMeta):
    """Base class for ADAM gradient training."""

    def __init__(self):
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
        for param in self.params:
            self.pmean.append(np.zeros_like(param))
            self.pvar.append(np.zeros_like(param))

        # Parameters of the training.
        self._b1 = 0.95
        self._b2 = 0.999
        self._lrate = 0.01

    @property
    def b1(self):
        """b1."""
        return self._b1

    @b1.setter
    def b1(self, value):
        self._b1 = value

    @property
    def b2(self):
        """b2."""
        return self._b2

    @b2.setter
    def b2(self, value):
        self._b2 = value

    @property
    def lrate(self):
        """Learning rate."""
        return self._lrate

    @lrate.setter
    def lrate(self, value):
        self._lrate = value

    @abc.abstractproperty
    def after_grad_update(self):
        """Called after each gradient update."""
        pass

    def adam_sga_update(self, data, epoch):
        """ADAM gradients update.

        Parameters
        ----------
       data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.
        epoch : theano int
            Epoch of the training.

        Returns
        -------
        objective : float
            Value of the objective function before the update.

        """
        # Compute the gradients.
        objective, grads = self.gradients(data)

        # Update the parameters.
        gamma = np.sqrt(1 - self._b2**epoch) / (1 - self._b1**epoch)
        values_iterable = zip(self.params, grads, self.pmean, self.pvar)
        for param, gradient, pmean, pvar in values_iterable:
            new_m = self._b1 * pmean + (1. - self._b1) * gradient
            new_v = self._b2 * pvar + (1. - self._b2) * (gradient**2)
            param += self._lrate * gamma * new_m / (np.sqrt(new_v) + ADAM_EPS)
            pmean = new_m
            pvar = new_v

        # Post-processing.
        self.after_grad_update()

        return objective
