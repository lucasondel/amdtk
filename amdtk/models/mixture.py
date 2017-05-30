
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
from .model import EFDStats, DiscreteLatentModel
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
                np.ones_like(precision),
                np.ones_like(precision) * prior_var,
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
                np.ones_like(precision),
                prior_strength / var
            )
        components.append(amdtk.NormalDiag(priors[i], posterior))

        return Mixture(dirichlet_prior, dirichlet_posterior, components)

    def __init__(self, latent_prior, latent_posterior, components):
        DiscreteLatentModel.__init__(self, latent_prior, latent_posterior,
                                     components)

    # DiscreteLatentModel interface implementation.
    # -----------------------------------------------------------------

    def get_posteriors(self, s_stats, accumulate=False):
        # Expected value of the log-likelihood.
        exp_llh = self.components_exp_llh(s_stats, log_resps)
        exp_llh += self.posterior.grad_log_partition[:, np.newaxis]

        # Softmax function to get the posteriors.
        log_norm = logsumexp(exp_llh, axis=0)
        resps = np.exp((exp_llh - log_norm))

        # Accumulate the responsibilties if requested.
        if accumulate:
            acc_stats1 = resps.T.sum(axis=0)
            acc_stats2 = resps.dot(s_stats)
            acc_stats = EFDStats([acc_stats1, acc_stats2])

            return resps, log_norm, acc_stats

        return resps, log_norm

    # -----------------------------------------------------------------

