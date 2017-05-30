
"""
Base class for members of the Exponential Family of distribution (EFD).

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
import numpy as np
from ..io.persistent_model import PersistentModel


class EFDPrior(PersistentModel, metaclass=abc.ABCMeta):
    """Abstract base class for a prior from the EFD."""

    @abc.abstractstaticmethod
    def _log_partition_func(natural_params):
        pass

    @abc.abstractstaticmethod
    def _grad_log_partition_func(natural_params):
        pass

    def __init__(self):
        self._natural_params = None
        self._log_partition =  None
        self._grad_log_partition = None

    @property
    def natural_params(self):
        """Vector of natural parameters."""
        return self._natural_params

    @natural_params.setter
    def natural_params(self, value):
        self._natural_params = value
        self._log_partition = self._log_partition_func(value)
        self._grad_log_partition = self._grad_log_partition_func(value)

    @property
    def log_partition(self):
        """Log-partition of the distribution."""
        return self._log_partition

    @property
    def grad_log_partition(self):
        """Gradient of the log-partition.

        Gradient of the log-partition with respect to the natural
        parameters. This corresponds to the expected value of the
        sufficient statistics of the current density/distribution.

        """
        return self._grad_log_partition

    def kl_div(self, dist):
        """Kullback-Leibler divergence.

        Compute the Kullback-Leibler divergence between the current
        distribution and the given distribution.

        Parameters
        ----------
        dist : EFDPrior
            Distribution of the same type as the current distribution
            (i.e. "self").


        -------
        div : float
            Results of the KL divergence in nats.

        """
        # Expected value of the sufficient statistics with respect to
        # the current distribution
        expected_value = self.grad_log_partition

        # Natural parameters of the current and given distributions.
        nparams1 = self.natural_params
        nparams2 = dist.natural_params

        # Log-partition of the current and given distributions.
        log_partition1 = self.log_partition
        log_partition2 = dist.log_partition

        # Compute the KL divergence.
        retval = (nparams1 - nparams2).dot(expected_value)
        retval += log_partition2 - log_partition1

        return retval

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {'natural_params': self._natural_params}

    @classmethod
    def load_from_dict(cls, model_data):
        model = cls.__new__(Dirichlet)
        model.natural_params = model_data['natural_params']
        return model

    # -----------------------------------------------------------------


class EFDLikelihood(PersistentModel, metaclass=abc.ABCMeta):
    """Abstract base class for a likelihood from the EFD."""

    def __init__(self, prior, posterior):
        self._prior = prior
        self._posterior = posterior

    @abc.abstractstaticmethod
    def get_sufficient_stats(data):
        """Sufficient statistics of the current distribution.

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        s_stats : numpy.ndarray
            (NxD2) matrix of sufficient statistics. D2 is the dimension
            of the sufficient statistics for a single features frame.

        """
        pass

    @property
    def prior(self):
        """Conjugate prior."""
        return self._prior

    @prior.setter
    def prior(self, value):
        self._prior = value

    @property
    def posterior(self):
        """Conjugate posterior."""
        return self._posterior

    @posterior.setter
    def posterior(self, value):
        self._posterior = value


    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'class': self.__class__,
            'prior_class': self.prior.__class__,
            'prior_data': self.prior.to_dict(),
            'posterior_class': self.posterior.__class__,
            'posterior_data': self.posterior.to_dict()
        }

    @staticmethod
    def load_from_dict(model_data):
        model = model_data['class'].__new__(model_data['class'])
        prior_cls = model_data['prior_class']
        prior_data = model_data['prior_data']
        model.prior = prior_cls.load_from_dict(prior_data)

        posterior_cls = model_data['posterior_class']
        posterior_data = model_data['posterior_data']
        model.posterior = posterior_cls.load_from_dict(posterior_data)

        return model

    # -----------------------------------------------------------------

