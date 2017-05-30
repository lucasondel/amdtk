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
from .efd import EFDStats, LatentEFD
from .svae_prior import SVAEPrior
from .hmm_util import create_phone_loop_transition_matrix
from .hmm_util import forward_backward
from .dirichlet import Dirichlet


class PhoneLoop(LatentEFD, SVAEPrior):
    """Bayesian Phone Loop.

    Bayesian Phone Loop with a Dirichlet prior over the weights.

    """

    def __init__(self, prior, posterior, gauss_components, n_states,
                 n_gauss_per_states):
        LatentEFD.__init__(self, prior, posterior, gauss_components)

        self.n_units = len(prior.natural_params)
        self.n_states = n_states
        tot_n_states = self.n_units * self.n_states
        self.n_gauss_per_states = n_gauss_per_states
        tot_gauss = self.n_units * self.n_states * n_gauss_per_states

        self.state_priors = [Dirichlet(np.ones(n_gauss_per_states))
                             for _ in range(tot_n_states)]
        self.state_posteriors = [Dirichlet(np.ones(n_gauss_per_states))
                                 for _ in range(tot_n_states)]

        # Will be initialized later.
        self.init_prob = None
        self.trans_mat = None
        self.init_states = None
        self.final_states = None

        self.vb_post_update()

    def vb_post_update(self):
        LatentEFD.vb_post_update(self)

        # Update the states' weights.
        self.state_log_weights = np.zeros((self.n_units * self.n_states,
                                           self.n_gauss_per_states))
        for idx in range(self.n_units * self.n_states):
                self.state_log_weights[idx, :] = \
                    self.state_posteriors[idx].grad_log_partition

        # Update the log transition matrix.
        unigram_lm = np.exp(self.posterior.grad_log_partition)
        unigram_lm /= unigram_lm.sum()
        self.init_prob = unigram_lm
        self.trans_mat, self.init_states, self.final_states = \
            create_phone_loop_transition_matrix(self.n_units, self.n_states,
                                                unigram_lm)

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


    def viterbi(self, llhs):
        backtrack = np.zeros_like(llhs, dtype=int)
        omega = np.zeros(llhs.shape[1]) + float('-inf')
        omega[self.init_states] = llhs[0, self.init_states] + \
            self.posterior.grad_log_partition
        log_A = np.log(self.trans_mat.toarray())

        for i in range(1, llhs.shape[0]):
            hypothesis = omega + log_A.T
            backtrack[i] = np.argmax(hypothesis, axis=1)
            omega = llhs[i] + hypothesis[range(len(log_A)),
                                         backtrack[i]]

        path = [self.final_states[np.argmax(omega[self.final_states])]]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])

        return path

    def decode(self, data, state_path=False):
        s_stats = self.get_sufficient_stats(data)
        exp_llh = self.components_exp_llh(s_stats)
        r_exp_llh = exp_llh.reshape(self.n_units * self.n_states,
                                    self.n_gauss_per_states, -1)
        c_given_s_llh = r_exp_llh + self.state_log_weights[:, :, np.newaxis]
        state_llh = logsumexp(c_given_s_llh, axis=1).T

        path = self.viterbi(state_llh)
        if not state_path:
            path = [bisect(self.init_states, state) for state in path]
            path = [x[0] for x in groupby(path)]

        return path


    # SVAEPrior interface.
    # ------------------------------------------------------------------

    def kl_div_posterior_prior(self):
        """Kullback-Leibler divergence between prior /posterior.

        Returns
        -------
        kl_div : float
            Kullback-Leibler divergence.

        """
        retval = LatentEFD.kl_div_posterior_prior(self)
        for idx, post in enumerate(self.state_posteriors):
            retval += post.kl_div(self.state_priors[idx])
        return retval

    def get_resps(self, s_stats, log_resps=None):
        if log_resps is not None:
            extended_lresps = np.repeat(log_resps, self.n_states, axis=1)
            log_alphas, log_betas = forward_backward(
                self.init_prob,
                self.trans_mat,
                self.init_states,
                self.final_states,
                extended_lresps.T
            )
            log_q_Z = (log_alphas + log_betas).T
            log_norm = logsumexp(log_q_Z, axis=0)
            extended_lresps = (log_q_Z - log_norm).T
        else:
            extended_lresps = None

        # Evaluate the Gaussian log-likelihoods.
        exp_llh = self.components_exp_llh(s_stats)

        # Reshape the log-likelihood to get the per-state and per
        # component log-likelihood.
        r_exp_llh = exp_llh.reshape(self.n_units * self.n_states,
                                    self.n_gauss_per_states, -1)

        # Emission log-likelihood.
        c_given_s_llh = r_exp_llh + self.state_log_weights[:, :, np.newaxis]
        state_llh = logsumexp(c_given_s_llh, axis=1).T
        c_given_s_resps = np.exp(c_given_s_llh - \
            state_llh.T[:, np.newaxis, :])

        if extended_lresps is not None:
            state_llh = extended_lresps

        # Forward-Bacward algorithm.
        log_alphas, log_betas = forward_backward(
            self.init_prob,
            self.trans_mat,
            self.init_states,
            self.final_states,
            state_llh.T
        )

        # State/Gaussian responsibilities.
        log_q_Z = (log_alphas + log_betas).T
        log_norm = logsumexp(log_q_Z, axis=0)
        state_resps = np.exp((log_q_Z - log_norm))
        tot_resps = state_resps[:, np.newaxis, :] * c_given_s_resps
        gauss_resps = tot_resps.reshape(-1, tot_resps.shape[-1])

        # Accumulate statistics.
        if self.n_states > 1 :
            units_stats = self.units_stats(state_llh, log_alphas, log_betas)
        else:
            units_stats = resps.sum(axis=0)
        state_stats = tot_resps.sum(axis=2)
        gauss_stats = gauss_resps.dot(s_stats)
        acc_stats = EFDStats([units_stats, state_stats, gauss_stats])

        return log_norm[-1], (gauss_resps, state_resps), acc_stats

    def vb_m_step(self, acc_stats):
        self.posterior.natural_params = self.prior.natural_params + \
            acc_stats[0]

        for idx, stats in enumerate(acc_stats[1]):
            self.components[idx].posterior.natural_params = \
                self.components[idx].prior.natural_params + stats

        self.vb_post_update()

    def natural_grad_update(self, acc_stats, lrate):
        """Natural gradient update."""

        # Update unigram language model.
        grad = self.prior.natural_params + acc_stats[0]
        grad -= self.posterior.natural_params
        self.posterior.natural_params = \
            self.posterior.natural_params + lrate * grad

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

        self.vb_post_update()

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'prior_class': self.prior.__class__,
            'prior_data': self.prior.to_dict(),
            'posterior_class': self.posterior.__class__,
            'posterior_data': self.posterior.to_dict(),
            'components_class': [comp.__class__ for comp in self.components],
            'components_data': [comp.to_dict() for comp in self.components],
            'n_units': self.n_units,
            'n_states': self.n_states,
            'trans_mat': self.trans_mat,
            'init_states': self.init_states,
            'final_states': self.final_states
        }

    @staticmethod
    def load_from_dict(model_data):
        model = PhoneLoop.__new__(PhoneLoop)

        prior_cls = model_data['prior_class']
        prior_data = model_data['prior_data']
        model.prior = prior_cls.load_from_dict(prior_data)

        posterior_cls = model_data['posterior_class']
        posterior_data = model_data['posterior_data']
        model.posterior = posterior_cls.load_from_dict(posterior_data)

        components_cls = model_data['components_class']
        components_data = model_data['components_data']
        components = []
        for idx in range(len(components_cls)):
            component = \
                components_cls[idx].load_from_dict(components_data[idx])
            components.append(component)
        model.components = components

        model.n_units = model_data['n_units']
        model.n_states = model_data['n_states']
        model.log_trans_mat = model_data['log_trans_mat']
        model.init_states = model_data['init_states']
        model.final_states = model_data['final_states']

        model.vb_post_update()

        return model

    # -----------------------------------------------------------------

