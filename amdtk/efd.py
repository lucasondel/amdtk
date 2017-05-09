"""
Base classes for member of the Exponential Family of distribution (EFD).

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
from .model import PersistentModel


class EFDStats(object):
    """Wrapper of EFD statistics."""

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


class EFDPrior(PersistentModel, metaclass=abc.ABCMeta):
    """Abstract base class for a prior from the EFD."""

    @abc.abstractproperty
    def natural_params(self):
        """Vector of natural parameters."""
        pass

    @abc.abstractproperty
    def log_partition(self):
        """Log-partition of the distribution."""
        pass

    @abc.abstractproperty
    def grad_log_partition(self):
        """Gradient of the log-partition.

        Gradient of the log-partition with respect to the natural
        parameters. This corresponds to the expected value of the
        sufficient statistics of the current density/distribution.

        """
        pass

    @abc.abstractmethod
    def evaluate_log_partition(self, natural_params):
        """Evaluate the log-parition.

        Parameters
        ----------
        natural_params : numpy.ndarray
            Natural parameters of the distribution.

        """
        pass

    def kl_div(self, dist):
        """Kullback-Leibler divergence.

        Compute the Kullback-Leibler divergence between the current
        distribution and the given distribution.

        Parameters
        ----------
        dist : EFDPrior
            Distribution of the same type as the current distribution
            (i.e. "self").

        Returns
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


class SVAEPrior(PersistentModel, metaclass=abc.ABCMeta):
    """Abstract base class for SVAE prior."""

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
        prob = 1. / len(self.components)
        return np.ones((n_frames, len(self.components))) * prob

    @abc.abstractmethod
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
        model_data : object
            Model speficic data for the training.

        """
        pass

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
        pass


class LatentEFD(SVAEPrior, metaclass=abc.ABCMeta):
    """Abstract base class for model with latent EFD model."""

    def __init__(self, prior, posterior, components):
        self._prior = prior
        self._posterior = posterior
        self._components = components
        self._p_matrix = self.get_components_params_matrix()

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

    @property
    def components(self):
        """Components of the model."""
        return self._components

    @components.setter
    def components(self, value):
        self._components = value

    @property
    def components_params_matrix(self):
        """Matrix of components' parameters."""
        return self._p_matrix

    @components_params_matrix.setter
    def components_params_matrix(self, value):
        self._p_matrix = value

    def get_components_params_matrix(self):
        """Get the matrix of parameters for fast evaluation.

        Returns
        -------
        p_matrix : numpy.ndarray
            Matrix where ith row is the expected value of the the
            natural parameters of the ith component.

        """
        return np.vstack([comp.posterior.grad_log_partition
                          for idx, comp in enumerate(self.components)])

    def get_sufficient_stats(self, data):
        """Sufficient statistics of the latent model.

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        s_stats : numpy.ndarray
            (NxD2) matrix of sufficient statistics. D2 is the
            dimension of the sufficient statistics for a single
            features frame.

        """
        cls = self.components[0].__class__
        s_stats = cls.get_sufficient_stats(data)
        return s_stats

    def components_exp_llh(self, s_stats, log_resps=None):
        """Per components expected log-likelihood.

        Parameters
        ----------
        s_stats : numpy.ndarray
            (NxD) matrix of sufficient statistics.
        log_resps : numpy.ndarray
            Matrix of log responsibilities to enforce which components
            can be active.

        """
        retval = self.components_params_matrix.dot(s_stats.T)
        if log_resps is not None:
            retval += log_resps.T
        return retval

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

    def vb_e_step(self, data, log_resps=None):
        """E-step of the standard Variational Bayes algorithm.

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.
        log_resps : numpy.ndarray
            Alignments.

        Returns
        -------
        exp_llh : float
            Expected value of the log-likelihood of the data given the
            model.
        acc_stats : numpy.ndarray
            Accumulated sufficient statistics.

        """
        s_stats = self.get_sufficient_stats(data)
        log_norm, resps, acc_stats = self.get_resps(s_stats, log_resps)
        return log_norm, acc_stats


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
            self.components[idx].posterior.natural_params = \
                self.components[idx].prior.natural_params + stats

        self.vb_post_update()


    def vb_post_update(self):
       """Method called after each update."""
       self.components_params_matrix = self.get_components_params_matrix()


    def natural_grad_update(self, acc_stats, lrate):
        """Natural gradient update."""
        grad = self.prior.natural_params + acc_stats[0]
        grad -= self.posterior.natural_params
        self.posterior.natural_params = \
            self.posterior.natural_params + lrate * grad

        for idx, stats in enumerate(acc_stats[1]):
            comp = self.components[idx]
            grad = comp.prior.natural_params + stats
            grad -= comp.posterior.natural_params
            comp.posterior.natural_params = \
                comp.posterior.natural_params + lrate * grad

        self.vb_post_update()

