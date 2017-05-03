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


class PhoneLoop(LatentEFD, SVAEPrior):
    """Bayesian Phone Loop.

    Bayesian Phone Loop with a Dirichlet prior over the weights.

    """

    def __init__(self, prior, posterior, gauss_components, n_states):
        LatentEFD.__init__(self, prior, posterior, gauss_components)

        self.n_units = len(prior.natural_params)
        self.n_states = n_states

        # Will be initialized later.
        self.init_prob = None
        self.trans_mat = None
        self.init_states = None
        self.final_states = None

        self.vb_post_update()

    def vb_post_update(self):
        LatentEFD.vb_post_update(self)

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

    def decode(self, data, log_resps=None, state_path=False):
        if log_resps is not None:
            extended_lresps = np.repeat(log_resps, self.n_states, axis=1)
        else:
            extended_lresps = None
        s_stats = self.get_sufficient_stats(data)
        exp_llh = self.components_exp_llh(s_stats, extended_lresps)
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

        exp_llh = self.components_exp_llh(s_stats, extended_lresps)
        log_alphas, log_betas = forward_backward(
            self.init_prob,
            self.trans_mat,
            self.init_states,
            self.final_states,
            exp_llh
        )
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

    def vb_m_step(self, acc_stats):
        self.posterior.natural_params = self.prior.natural_params + \
            acc_stats[0]

        for idx, stats in enumerate(acc_stats[1]):
            self.components[idx].posterior.natural_params = \
                self.components[idx].prior.natural_params + stats

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

