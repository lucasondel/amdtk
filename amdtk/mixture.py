"""
Main class of the mixture model.

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
from scipy.special import logsumexp
from .model import PersistentModel
from .efd import EFDStats
from .svae_prior import SVAEPrior


class Mixture(PersistentModel, SVAEPrior):
    """Bayesian Mixture Model.

    Bayesian Mixture Model with a Dirichlet prior over the weights.

    """

    def __init__(self, prior, posterior, components):
        """Initialize the GMM.

        Parameters
        ----------
        prior : :class:`Dirichlet`
            Dirichlet prior of the mixture.
        posterior : :class:`Dirichlet`
            Dirichlet posterior of the mixture.
        compents : list
            List of :class:`Normal`

        """
        self.prior = prior
        self.posterior = posterior
        self.components = components

        # matrix of the components' parameters.
        self.comp_params = self._get_param_matrix()

    def _get_param_matrix(self):
        return np.vstack([comp.posterior.grad_log_partition
                          for idx, comp in enumerate(self.components)])

    def get_sufficient_stats(self, data):
        """Sufficient statistics of GMM.

        The sufficient statistics of the GMM include the ones of the
        mixture's component appended with a '1' (for the Dirichlet
        prior).

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        s_stats : numpy.ndarray
            (NxD)2 matrix of sufficient statistics. D2 is the dimension
            of the sufficient statistics for a single features frame.

        """
        cls = self.components[0].__class__
        normal_stats = cls.get_sufficient_stats(data)
        return normal_stats

    def update_posterior(self, acc_s_stats):
        """Update the posterior distribution.

        Parameters
        ----------
        acc_s_stats : numpy.ndarray
            Accumulated sufficient statistics.

        """
        for idx, stats in enumerate(acc_s_stats[:, :-1]):
            self.components[idx].update_posterior(stats)
        self.posterior.natural_params = self.prior.natural_params + \
            acc_s_stats[:, -1]

    def kl_div_posterior_prior(self):
        """Kullback-Leibler divergence between prior /posterior.

        Returns
        -------
        kl_div : float
            Kullback-Leibler divergence.

        """
        retval = 0.
        retval += self.posterior.kl_div(self.prior)
        for comp in self.components:
            retval += comp.posterior.kl_div(comp.prior)
        return retval

    # SVAEPrior interface.
    # ------------------------------------------------------------------

    def init_resps(self, n_frames):
        """Get the initialize per-frame responsibilities.

        Parameters
        ----------
        n_frames : numpy.ndarray,
            Number of frames for the mini-batch.

        Returns
        -------
        resps : numpy.ndarray
            Initial per-frame responsibilities.

        """
        prob = np.exp(self.posterior.grad_log_partition)
        prob /= prob.sum()
        return np.ones((n_frames, len(self.components))) * prob

    def get_resps(self, s_stats, output_llh=False):
        """Get the components' responisbilities.

        Parameters
        ----------
        s_stats : numpy.ndarray,
            Sufficient statistics.
        output_llh : boolean
            If True, returns the per component log-likelihood.

        Returns
        -------
        log_norm : numpy.ndarray
            Per-frame log normalization constant.
        resps : numpy.ndarray
            Responsibilities.
        exp_llh : boolean
            If output_llh is True, per component log-likelihood.

        """
        # Expected value of the log-likelihood w.r.t. the posteriors.
        exp_llh = self.comp_params.dot(s_stats.T) + \
            self.posterior.grad_log_partition[:, np.newaxis]

        # Softmax.
        log_norm = logsumexp(exp_llh, axis=0)
        resps = np.exp((exp_llh - log_norm))

        return log_norm, resps.T, exp_llh

    def accumulate_stats(self, s_stats, resps, model_data):
        """Accumulate the sufficient statistics.

        Parameters
        ----------
        s_stats : numpy.ndarray
            Sufficient statistics.
        resps : numpy.ndarray
            Per-frame responsibilities.
        model_data : object
            Model speficic data for the training.

        Returns
        -------
        acc_stats : :class:`EFDStats`
            Accumulated sufficient statistics.

        """
        acc_stats1 = resps.sum(axis=0)
        acc_stats2 = resps.T.dot(s_stats)
        return EFDStats([acc_stats1, acc_stats2])

    def grads_from_acc_stats(self, acc_stats):
        """Compute the gradients from the accumulated statistics.

        Parameters
        ----------
        acc_stats : :class:`EFDStats`
            Accumulated sufficient statistics.

        Returns
        -------
        grads : list
            List of all the gradients.

        """
        grads = []
        grads.append(self.prior.natural_params + acc_stats[0] \
                     - self.posterior.natural_params)

        for idx, stats in enumerate(acc_stats[1]):
            component = self.components[idx]
            grad = component.prior.natural_params + stats - \
                component.posterior.natural_params
            grads.append(grad)

        return grads

    # StdSGAPython interface.
    # ------------------------------------------------------------------

    def gradients(self, data):
        """Compute the natural gradient of the VB lower bound.

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        grads : list
            List of gradients.

        """
        # Run the standard VB E-step.
        log_norm, acc_stats = self.vb_e_step(data)

        # Scale the statistics.
        scale = self.n_frames / log_norm.shape[0]
        acc_stats *= scale

        # Estimate the lower bound
        kl_div = self.kl_div_posterior_prior()
        lower_bound = (scale * log_norm.sum() - kl_div) / self.n_frames

        # Compute the (natural) gradients.
        grads = self.grads_from_acc_stats(acc_stats)

        return lower_bound, grads

    def after_grad_update(self):
        """Called after each gradient update."""
        self.posterior.natural_params = self.params[0]
        for idx, component in enumerate(self.components):
            component.posterior.natural_params = self.params[idx + 1]
        self.comp_params = self._get_param_matrix()

    # Variational Bayes training.
    # ------------------------------------------------------------------

    def vb_e_step(self, data):
        """E-step of the standard Variational Bayes algorithm.

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        exp_llh : float
            Expected value of the log-likelihood of the data given the
            model.
        acc_s_stats : numpy.ndarray
            Accumulated sufficient statistics.
        is_sufficient_stats : bool
            Whether the given data are the sufficient statistics
            corresponding to the model.

        """
        s_stats = self.get_sufficient_stats(data)
        log_norm, resps, model_data = self.get_resps(s_stats)
        acc_stats1 = resps.sum(axis=0)

        # Accumulate the statistics.
        return log_norm, self.accumulate_stats(s_stats, resps, model_data)

    def vb_m_step(self, acc_stats):
        """M-step of the standard Variational Bayes algorithm.

        Parameters
        ----------
        acc_s_stats : numpy.ndarray
            Accumulated sufficient statistics.

        """
        self.posterior.natural_params = self.prior.natural_params + \
            acc_stats[0]

        for idx, stats in enumerate(acc_stats[1]):
            self.components[idx].update_posterior(stats)

        self.comp_params = self._get_param_matrix()

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'prior_class': self.prior.__class__,
            'prior_data': self.prior.to_dict(),
            'posterior_class': self.posterior.__class__,
            'posterior_data': self.posterior.to_dict(),
            'components_class': [comp.__class__ for comp in self.components],
            'components_data': [comp.to_dict() for comp in self.components]
        }

    @staticmethod
    def load_from_dict(model_data):
        model = Mixture.__new__(Mixture)

        prior_cls = model_data['prior_class']
        prior_data = model_data['prior_data']
        model.prior = prior_cls.load_from_dict(prior_data)

        posterior_cls = model_data['posterior_class']
        posterior_data = model_data['posterior_data']
        model.posterior = posterior_cls.load_from_dict(posterior_data)

        components_cls = model_data['components_class']
        components_data = model_data['components_data']
        components = []
        for idx in mixture(len(components_cls)):
            component = \
                components_cls[idx].load_from_dict(components_data[idx])
            components.append(component)
        model.components = components

        model.comp_params = self._get_param_matrix()

        return model

    # ------------------------------------------------------------------

