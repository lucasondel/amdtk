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
from scipy.special import logsumexp
from .efd import EFDStats, LatentEFD
from .svae_prior import SVAEPrior
from bisect import bisect
from itertools import groupby


class PhoneLoop(LatentEFD, SVAEPrior):
    """Bayesian Phone Loop.

    Bayesian Phone Loop with a Dirichlet prior over the weights.

    """

    @staticmethod
    def __log_transition_matrix(n_units, n_states, mixture_weights,
                                prob_final_state=.5):
        """Create a log transition matrix."""
        tot_n_states = n_units * n_states
        init_states = np.arange(0, tot_n_states, n_states)
        final_states = init_states + n_states - 1
        trans_mat = np.zeros((tot_n_states, tot_n_states)) + float('-inf')

        for idx, init_state in enumerate(init_states):
            for offset in range(n_states - 1):
                state = init_state + offset
                trans_mat[state, state: state + n_states - 1] = np.log(.5)
            if n_states > 1:
                trans_mat[final_states[idx], final_states[idx]] = \
                    np.log(prob_final_state)

        for idx, final_state in enumerate(final_states):
            if n_states > 1:
                # Disallow repeating a unit.
                m_weights = mixture_weights.copy()
                m_weights[idx] = 0.
                m_weights /= m_weights.sum()

                trans_mat[final_state, init_states] = \
                    np.log((1 - prob_final_state) * m_weights)
            else:
                trans_mat[final_state, init_states] = np.log(mixture_weights)

        return trans_mat, init_states, final_states

    def __init__(self, prior, posterior, components):
        LatentEFD.__init__(self, prior, posterior, components)

        self.n_units = len(prior.natural_params)
        self.n_states = int(len(components) / self.n_units)
        weights = np.exp(self.posterior.grad_log_partition)
        weights /= weights.sum()
        self.log_trans_mat, self.init_states, self.final_states = \
            PhoneLoop.__log_transition_matrix(self.n_units, self.n_states,
                                              weights)

        self.vb_post_update()

    def vb_post_update(self):
        LatentEFD.vb_post_update(self)

        # Update the log transition matrix.
        weights = np.exp(self.posterior.grad_log_partition)
        weights /= weights.sum()
        prob_fs = np.exp(self.log_trans_mat[self.final_states[0],
                                            self.final_states[0]])
        for idx, final_state in enumerate(self.final_states):
            if self.n_states > 1:
                m_weights = weights.copy()
                m_weights[idx] = 0.
                m_weights /= m_weights.sum()
                self.log_trans_mat[final_state, self.init_states] = \
                    np.log((1 - prob_fs) * m_weights)
            else:
                self.log_trans_mat[final_state, self.init_states] = \
                    weights

    def forward(self, llhs):
        log_init = self.posterior.grad_log_partition
        log_init -= logsumexp(log_init)
        log_alphas = np.zeros_like(llhs) - np.inf
        log_alphas[0, self.init_states] = llhs[0, self.init_states] + log_init
        log_A = self.log_trans_mat
        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += logsumexp(log_alphas[i-1] + log_A.T, axis=1)
        return log_alphas

    def backward(self, llhs):
        log_A = self.log_trans_mat
        log_betas = np.zeros_like(llhs) - np.inf
        log_betas[-1, self.final_states] = 0.
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = logsumexp(log_A + llhs[i+1] + log_betas[i+1],
                                     axis=1)
        return log_betas

    def units_stats(self, c_llhs, log_alphas, log_betas):
        log_units_stats = np.zeros(self.n_units)
        norm = logsumexp(log_alphas[-1] + log_betas[-1])

        for n_unit in range(self.n_units):
            index1 = n_unit * self.n_states + 1
            index2 = index1 + 1
            log_prob_trans = self.log_trans_mat[index1, index2]
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

        for i in range(1, llhs.shape[0]):
            hypothesis = omega + self.log_trans_mat.T
            backtrack[i] = np.argmax(hypothesis, axis=1)
            omega = llhs[i] + hypothesis[range(len(self.log_trans_mat)),
                                         backtrack[i]]

        path = [self.final_states[np.argmax(omega[self.final_states])]]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])

        return path

    def decode(self, data, state_path=False):
        s_stats = self.get_sufficient_stats(data)
        exp_llh = self.components_exp_llh(s_stats)
        path = self.viterbi(exp_llh.T)
        if not state_path:
            path = [bisect(self.init_states, state) for state in path]
            path = [x[0] for x in groupby(path)]

        return path


    # SVAEPrior interface.
    # ------------------------------------------------------------------

    def get_resps(self, s_stats, log_resps=None):
        if log_resps is not None:
            extended_lresps = np.repeat(log_resps, self.n_states, axis=1)
            log_alphas = self.forward(extended_lresps)
            log_betas = self.backward(extended_lresps)
            log_q_Z = (log_alphas + log_betas).T
            log_norm = logsumexp(log_q_Z, axis=0)
            extended_lresps = (log_q_Z - log_norm).T
        else:
            extentd_lresps = None

        exp_llh = self.components_exp_llh(s_stats, extended_lresps)
        log_alphas = self.forward(exp_llh.T)
        log_betas = self.backward(exp_llh.T)
        log_q_Z = (log_alphas + log_betas).T
        log_norm = logsumexp(log_q_Z, axis=0)
        resps = np.exp((log_q_Z - log_norm))

        return log_norm[-1], resps.T, (exp_llh.T, log_alphas, log_betas)

    def accumulate_stats(self, s_stats, resps, model_data):
        if self.n_states > 1 :
            acc_stats1 = self.units_stats(*model_data)
        else:
            acc_stats1 = resps.sum(axis=0)
        acc_stats2 = resps.T.dot(s_stats)
        return EFDStats([acc_stats1, acc_stats2])


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
            'log_trans_mat': self.log_trans_mat,
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

