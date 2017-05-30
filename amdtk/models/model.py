
"""
Base class for the generative / discriminative models.

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


class EFDStats(object):
    """Wrapper of a list of accumulated sufficient statistics."""

    def __init__(self, stats):
        """Initialize the list of stats.

        Parameters
        ----------
        stats : list
            List of numpy.ndarray.

        """
        self._stats = stats

    def __setitem__(self, k, value):
        self._stats[k] = value

    def __getitem__(self, k):
        return self._stats[k]

    def __iadd__(self, stats):
        for i in range(len(self._stats)):
            self._stats[i] += stats[i]
        return self

    def __imul__(self, scale):
        for i in range(len(self._stats)):
            self._stats[i] *= scale
        return self


class DiscreteLatentModel(PersistentModel, metaclass=abc.ABCMeta):
    """Abstract base class models with latent discrete variable.

    Concrete implementation of this interface can be either
    a Mixture Model (when the latent variables are independent) or
    a Hidden Markov Model (when the latent variables follow a 1st
    order Markov dynamic).

    NOTE
    ----
    This implementation assume the joint distribution of the
    data and the latent variables to be a member of the Exponential
    family of distribution.

    """

    def __init__(self, latent_prior, latent_posterior, components):
        """Initialize a :class:`DiscreteLatentModel`

        Parameters
        ----------
        latent_prior : :class:`EFDPrior`
            Prior over the latent variable.
        latent_posterior : :class:`EFDPrior`
            Initial guess for the posterior over the latent variable.
        components : list or tuple of :class:`EFDLikelihood`
            List of distributions / densities for the conditional
            likelihood. No check is performed but we assumed that all
            the elements of the list are of the same distribution /
            density type.

        """
        self._latent_prior = latent_prior
        self._latent_posterior = latent_posterior
        self._components = components

        # Pre-compute the expected value of the natural parameters for
        # each components to speed computation during
        # training/decoding.
        self._exp_np_matrix = self._get_components_params_matrix()

    @property
    def latent_prior(self):
        return self._latent_prior

    @latent_prior.setter
    def latent_prior(self, value):
        self._latent_prior = value

    @property
    def latent_prosterior(self):
        return self._latent_posterior

    @latent_posterior.setter
    def latent_posterior(self, value):
        self._latent_posterior = value

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, value):
        self._components = value

    def _get_components_params_matrix(self):
        """Natural parameters of the components as a matrix.

        Expected value of the natural parameters w.r.t. their current
        posterior distribution organized as a matrix.

        Returns
        -------
        exp_np_matrix :
            Expected value of the natural parameters of the components
             as a matrix.

        """
        return np.vstack([comp.posterior.grad_log_partition
                          for idx, comp in enumerate(self.components)])

    def get_sufficient_stats(self, data):
        """Sufficient statistics of the latent model.

        Parameters
        ----------
        data : numpy.ndarray
            (N x D) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        s_stats : numpy.ndarray
            (N x D2) matrix of sufficient statistics. D2 is the
            dimension of the sufficient statistics for a single
            features frame.

        """
        cls = self.components[0].__class__
        s_stats = cls.get_sufficient_stats(data)
        return s_stats

    def components_exp_llh(self, s_stats):
        """Per components expected log-likelihood.

        Parameters
        ----------
        s_stats : numpy.ndarray
            (NxD) matrix of sufficient statistics.

        """
        return self._exp_np_matrix.dot(s_stats.T)

    def kl_div_posterior_prior(self):
        """Sum of KL divergence between posteriors/priors.

        The sum of KL divergence between posterior/prior of the
        parameters of the model.

        Returns
        -------
        kl_div : float
            Kullback-Leibler divergence.
        """
        retval = 0.
        retval += self.latent_posterior.kl_div(self.latent_prior)
        for comp in self.components:
            retval += comp.posterior.kl_div(comp.prior)
        return retval

    def post_update(self):
        """Called after each update of the parameters."""
        self._exp_np_matrix = self._get_components_params_matrix()

    @abc.abstractmethod
    def get_posteriors(self, s_stats, accumulate=False):
        """Compute the posterior distribution of the latent variables.

        Parameters
        ----------
        s_stats : numpy.ndarray
            (N x D) matrix of sufficient statistics.
        accumulate : boolean
            If True, accumulate the sufficient statistics based on the
            posteriors.

        Returns
        -------
        posts : numpy.ndarray
            (N x K) posterior matrix. K is the number of state of the
            latent variables.
        log_evidence : float
            Log evidence of the data. For most, if not all, of the
            models the evidence is intractable and a lower-bound is
            returned instead.
        acc_stats : :class:`EFDStats`
            If accumulate is True, return accumulated sufficent
            statistics.

        """
        pass

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'latent_prior_class': self.latent_prior.__class__,
            'latent_prior_data': self.latent_prior.to_dict(),
            'posterior_class': self.latent_posterior.__class__,
            'posterior_data': self.latent_posterior.to_dict(),
            'components_class': self.components[0].__class__,
            'components': [comp.to_dict() for comp in components]
        }

    @staticmethod
    def load_from_dict(cls, model_data):
        model = cls.__new__(model_data['class'])
        latent_prior_cls = model_data['latent_prior_class']
        latent_prior_data = model_data['latent_prior_data']
        model.latent_prior = \
            latent_prior_cls.load_from_dict(latent_prior_data)

        latent_posterior_cls = model_data['latent_posterior_class']
        latent_posterior_data = model_data['latent_posterior_data']
        model.latent_posterior = \
            latent_posterior_cls.load_from_dict(latent_posterior_data)

        components = []
        components_class = model_data['components_class']
        for comp_data in model_data['components']:
            comp = components_class.load_from_dict(comp_data)
        model.components = components

        model.post_update()

        return model

    # -----------------------------------------------------------------

