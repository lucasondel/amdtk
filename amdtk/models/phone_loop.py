"""
Main class of the phone loop model.

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

import numpy as np
from bisect import bisect
from itertools import groupby
from scipy.special import logsumexp

from .hmm_utils import create_phone_loop_transition_matrix
from .hmm_utils import forward_backward
from .hmm_utils import viterbi
from .model import EFDStats, DiscreteLatentModel
from ..densities import Dirichlet, NormalGamma, NormalDiag


class PhoneLoop(DiscreteLatentModel):
    """Bayesian Phone Loop.

    Bayesian Phone Loop with a Dirichlet prior over the weights.

    """

    def create(n_units, n_states, n_comp_per_state, mean, var):
        """Create and initialize a Bayesian Phone Loope Model.

        Parameters
        ----------
        n_units : int
            Number of acoustic units i.e. phones.
        n_states : int
            Number of states for each acoustic unit.
        n_comp_per_state : int
            Number of compent per emission.
        mean : numpy.ndarray
            Mean of the data set to train on.
        var : numpy.ndarray
            Variance of the data set to train on.

        Returns
        -------
        model : :class:`PhoneLoop`
            A new phone-loop model.

        """
        tot_n_states = n_units * n_states
        tot_comp = tot_n_states * n_comp_per_state

        latent_prior = Dirichlet(np.ones(n_units))
        latent_posterior = Dirichlet(np.ones(n_units))

        state_priors = [Dirichlet(np.ones(n_comp_per_state))
                        for _ in range(tot_n_states)]
        state_posteriors = [Dirichlet(np.ones(n_comp_per_state))
                            for _ in range(tot_n_states)]

        priors = []
        prior_mean = mean.copy()
        prior_var = var.copy()
        for i in range(tot_comp):

            prior = NormalGamma(
                prior_mean,
                np.ones_like(mean),
                np.ones_like(var),
                prior_var,
            )
            priors.append(prior)

        components = []
        cov = np.diag(prior_var)
        for i in range(tot_comp):
            s_mean = np.random.multivariate_normal(mean, cov)
            posterior = NormalGamma(
                s_mean,
                np.ones_like(mean),
                np.ones_like(var),
                prior_var
            )
            components.append(NormalDiag(priors[i], posterior))

        return PhoneLoop(latent_prior, latent_posterior, state_priors,
                         state_posteriors, components)

    def __init__(self, latent_prior, latent_posterior, state_priors,
                 state_posteriors, components):
        DiscreteLatentModel.__init__(self, latent_prior, latent_posterior,
                                     components)

        self.n_units = len(latent_prior.natural_params)
        self.n_states = len(state_priors) // self.n_units
        self.n_comp_per_states = len(state_priors[0].natural_params)

        self.state_priors = state_priors
        self.state_posteriors = state_posteriors

        # Will be initialized later.
        self.init_prob = None
        self.trans_mat = None
        self.init_states = None
        self.final_states = None

        self.post_update()

    def post_update(self):
        DiscreteLatentModel.post_update(self)

        # Update the states' weights.
        self.state_log_weights = np.zeros((self.n_units * self.n_states,
                                           self.n_comp_per_states))
        for idx in range(self.n_units * self.n_states):
                self.state_log_weights[idx, :] = \
                    self.state_posteriors[idx].grad_log_partition

        # Update the log transition matrix.
        unigram_lm = np.exp(self.latent_posterior.grad_log_partition)
        unigram_lm /= unigram_lm.sum()
        self.init_prob = unigram_lm
        self.trans_mat, self.init_states, self.final_states = \
            create_phone_loop_transition_matrix(self.n_units, self.n_states,
                                                unigram_lm)

    def _get_state_llh(self, s_stats):
        # Evaluate the Gaussian log-likelihoods.
        exp_llh = self.components_exp_llh(s_stats)

        # Reshape the log-likelihood to get the per-state and per
        # component log-likelihood.
        r_exp_llh = exp_llh.reshape(self.n_units * self.n_states,
                                    self.n_comp_per_states, -1)

        # Emission log-likelihood.
        c_given_s_llh = r_exp_llh + self.state_log_weights[:, :, np.newaxis]
        state_llh = logsumexp(c_given_s_llh, axis=1).T
        c_given_s_resps = np.exp(c_given_s_llh - \
            state_llh.T[:, np.newaxis, :])

        return state_llh, c_given_s_resps

    def units_stats(self, c_llhs, log_alphas, log_betas):
        log_units_stats = np.zeros(self.n_units)
        norm = logsumexp(log_alphas[-1] + log_betas[-1])
        log_A = np.log(self.trans_mat.toarray())

        for n_unit in range(self.n_units):
            index1 = n_unit * self.n_states + 1
            index2 = index1 + 1
            log_prob_trans = log_A[index1, index2]
            log_q_zn1_zn2 = log_alphas[:-1, index1] + c_llhs[1:, index2] + \
                log_prob_trans + log_betas[1:, index2]
            log_q_zn1_zn2 -= norm
            log_units_stats[n_unit] = logsumexp(log_q_zn1_zn2)

        return np.exp(log_units_stats)

    def decode(self, data, state_path=False):
        s_stats = self.get_sufficient_stats(data)

        state_llh, c_given_s_resps = self._get_state_llh(s_stats)

        path = viterbi(
            self.init_prob,
            self.trans_mat,
            self.init_states,
            self.final_states,
            state_llh
        )
        if not state_path:
            path = [bisect(self.init_states, state) for state in path]
            path = [x[0] for x in groupby(path)]

        return path


    # DiscreteLatentModel interface.
    # -----------------------------------------------------------------

    def kl_div_posterior_prior(self):
        """Kullback-Leibler divergence between prior /posterior.

        Returns
        -------
        kl_div : float
            Kullback-Leibler divergence.

        """
        retval = DiscreteLatentModel.kl_div_posterior_prior(self)
        for idx, post in enumerate(self.state_posteriors):
            retval += post.kl_div(self.state_priors[idx])
        return retval

    def get_posteriors(self, s_stats, accumulate=False):
        state_llh, c_given_s_resps = self._get_state_llh(s_stats)

        # Forward-Bacward algorithm.
        log_alphas, log_betas = forward_backward(
            self.init_prob,
            self.trans_mat,
            self.init_states,
            self.final_states,
            state_llh.T
        )

        # Compute the posteriors.
        log_q_Z = (log_alphas + log_betas).T
        log_norm = logsumexp(log_q_Z, axis=0)
        state_resps = np.exp((log_q_Z - log_norm))

        if accumulate:
            tot_resps = state_resps[:, np.newaxis, :] * c_given_s_resps
            gauss_resps = tot_resps.reshape(-1, tot_resps.shape[-1])
            if self.n_states > 1 :
                units_stats = self.units_stats(state_llh, log_alphas,
                                               log_betas)
            else:
                units_stats = resps.sum(axis=0)

            state_stats = tot_resps.sum(axis=2)
            gauss_stats = gauss_resps.dot(s_stats)
            acc_stats = EFDStats([units_stats, state_stats, gauss_stats])

            return state_resps, log_norm[-1], acc_stats

        return state_resps, log_norm[-1]

    def natural_grad_update(self, acc_stats, lrate):
        """Natural gradient update."""
        # Update unigram language model.
        grad = self.latent_prior.natural_params + acc_stats[0]
        grad -= self.latent_posterior.natural_params
        self.latent_posterior.natural_params = \
            self.latent_posterior.natural_params + lrate * grad

        # Update the states' weights.
        for idx, post in enumerate(self.state_posteriors):
            grad = self.state_priors[idx].natural_params + acc_stats[1][idx]
            grad -= post.natural_params
            post.natural_params = post.natural_params + lrate * grad

        # Update Gaussian components.
        for idx, stats in enumerate(acc_stats[2]):
            comp = self.components[idx]
            grad = comp.prior.natural_params + stats
            grad -= comp.posterior.natural_params
            comp.posterior.natural_params = \
                comp.posterior.natural_params + lrate * grad

        self.post_update()

    # -----------------------------------------------------------------

