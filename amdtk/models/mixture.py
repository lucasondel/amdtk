
"""
Bayesian Mixture Model.

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
import theano
import theano.tensor as T
from .model import EFDStats, DiscreteLatentModel, DiscriminativeModel
from ..densities import Dirichlet, NormalGamma, NormalDiag


class Mixture(DiscreteLatentModel):
    """Bayesian Mixture Model.

    Bayesian Mixture Model with a Dirichlet prior over the weights.

    """

    def create(n_comp, mean, var):
        """Create and initialize a Bayesian Mixture Model.


        Parameters
        ----------
        n_comp : int
            Number of components in the mixture.
        mean : numpy.ndarray
            Mean of the data set to train on.
        var : numpy.ndarray
            Variance of the data set to train on.

        Returns
        -------
        model : :class:`Mixture`
            A new mixture model.

        """
        priors = []
        prior_mean = mean.copy()
        prior_var = var.copy()
        for i in range(n_comp):

            prior = NormalGamma(
                prior_mean,
                np.ones_like(mean),
                np.ones_like(var),
                1 / prior_var,
            )
            priors.append(prior)

        dirichlet_prior = Dirichlet(np.ones(n_comp))
        dirichlet_posterior = Dirichlet(np.ones(n_comp))

        components = []
        cov = np.diag(prior_var)
        for i in range(n_comp):
            s_mean = np.random.multivariate_normal(mean, cov)
            posterior = NormalGamma(
                s_mean,
                np.ones_like(mean),
                np.ones_like(var),
                1 / prior_var
            )
            components.append(NormalDiag(priors[i], posterior))

        return Mixture(dirichlet_prior, dirichlet_posterior, components)

    def __init__(self, latent_prior, latent_posterior, components):
        DiscreteLatentModel.__init__(self, latent_prior, latent_posterior,
                                     components)

    # DiscreteLatentModel interface implementation.
    # -----------------------------------------------------------------

    def get_posteriors(self, s_stats, accumulate=False, alignments=None):
        # Expected value of the log-likelihood.
        exp_llh = self.components_exp_llh(s_stats)
        exp_llh += self.latent_posterior.grad_log_partition[:, np.newaxis]

        # Softmax function to get the posteriors.
        log_norm = logsumexp(exp_llh, axis=0)
        resps = np.exp((exp_llh - log_norm))

        # Accumulate the responsibilties if requested.
        if accumulate:
            if alignments is not None:
                resps = np.zeros_like(resps)
                idxs = np.arange(0, len(resps))
                resps[idxs, alignemnts] = 1.
            acc_stats1 = resps.T.sum(axis=0)
            acc_stats2 = resps.dot(s_stats)
            acc_stats = EFDStats([acc_stats1, acc_stats2])

            return resps.T, log_norm, acc_stats

        return resps.T, log_norm

    # -----------------------------------------------------------------


class DiscriminativeMixture(DiscriminativeModel):
    """Bayesian Discriminative Mixture classifier.

    This is the equivalent discriminative model of the mixture model.
    Note that in the discriminative case, the mixture does not have
    weights.

    """

    def create(n_comp, mean, var, counts=1):
        """Create and initialize a Bayesian Mixture Model.


        Parameters
        ----------
        n_comp : int
            Number of components in the mixture.
        mean : numpy.ndarray
            Mean of the data set to train on.
        var : numpy.ndarray
            Variance of the data set to train on.

        Returns
        -------
        model : :class:`Mixture`
            A new mixture model.

        """
        priors = []
        prior_mean = mean.copy()
        prior_var = var.copy()
        for i in range(n_comp):

            prior = NormalGamma(
                prior_mean,
                np.ones_like(mean) * counts,
                np.ones_like(var) * counts,
                prior_var * counts,
            )
            prior.fixed_variance = True
            priors.append(prior)

        components = []
        cov = np.diag(prior_var)
        for i in range(n_comp):
            s_mean = np.random.multivariate_normal(mean, cov)
            posterior = NormalGamma(
                s_mean,
                np.ones_like(mean) * counts,
                np.ones_like(var) * counts,
                prior_var * counts
            )
            posterior.fixed_variance = True
            components.append(NormalDiag(priors[i], posterior))

        return DiscriminativeMixture(components)

    def __init__(self, components):
        DiscriminativeModel.__init__(self, components)

        values = np.vstack([comp.posterior.grad_log_partition
                            for comp in self.components])
        values = np.asarray(values, dtype=theano.config.floatX)
        self.sym_comp_params_matrix = theano.shared(values, borrow=True)

    # DiscriminativeModel interface implementation.
    # -----------------------------------------------------------------
    def classify_symfunc(self, s_stats):
        return T.nnet.softmax(T.dot(self.sym_comp_params_matrix, s_stats.T).T)

    def accumulate_stats(self, s_stats, posteriors):
        return EFDStats([posteriors.T.dot(s_stats)])

    def get_natural_grads(self, acc_stats):
        grads = []
        for idx, stats in enumerate(acc_stats[0]):
            comp = self.components[idx]
            grad = comp.prior.natural_params + stats
            grad -= comp.posterior.natural_params
            grads.append(grad)
        return grads

    def post_update(self):
        values = np.vstack([comp.posterior.grad_log_partition
                            for comp in self.components])
        values = np.asarray(values, dtype=theano.config.floatX)
        self.sym_comp_params_matrix.set_value(values)


    def natural_grad_update(self, acc_stats, lrate):
        """Natural gradient update.

        Parameters
        ----------
        acc_stats : :class:`EFDStats`
            Accumulated sufficient statistics.
        lrate : float
            Learning rate.

        """
        for idx, stats in enumerate(acc_stats[0]):
            comp = self.components[idx]
            grad = comp.prior.natural_params + stats
            grad -= comp.posterior.natural_params
            comp.posterior.natural_params = \
                comp.posterior.natural_params + lrate * grad

        values = np.vstack([comp.posterior.grad_log_partition
                            for comp in self.components])
        values = np.asarray(values, dtype=theano.config.floatX)
        self.sym_comp_params_matrix.set_value(values)

    # -----------------------------------------------------------------

