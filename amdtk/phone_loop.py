
"""Phone-Loop model where each unit is modeled by a left-to-right HMM."""

from bisect import bisect
import numpy as np
from scipy.misc import logsumexp
from scipy.special import psi, gammaln
from .model import Model
from .mixture import Mixture


def __log_transition_matrix(n_units, n_states, mixture_weights, ins_penalty=1.,
                            prob_final_state=.5):
    tot_n_states = n_units * n_states
    init_states = np.arange(0, tot_n_states, n_states)
    final_states = init_states + n_states - 1
    trans_mat = np.zeros((tot_n_states, tot_n_states)) + float('-inf')
    for idx, init_state in enumerate(init_states):
        for offset in range(n_states - 1):
            state = init_state + offset
            trans_mat[state, state: state + 2] = np.log(.5)
        if n_states > 1:
            trans_mat[final_states[idx], final_states[idx]] = \
                np.log(prob_final_state)
    for final_state in final_states:
        if n_states > 1:
            trans_mat[final_state, init_states] = ins_penalty * \
                np.log((1 - prob_final_state) * mixture_weights)
        else:
            trans_mat[final_state, init_states] = np.log(mixture_weights)
    return trans_mat, init_states, final_states


class PhoneLoop(Model):
    """Bayesian Phone (i.e. unit) Loop model. Note that
    the transition probability inside a unit is considered
    to be fixed. The prior over the weights of each phone
    is modeled either by a Dirichlet distribution or by a
    Truncated Dirichlet Process.

    """

    # pylint: disable=too-many-instance-attributes
    # This is a complex model hence the number
    # a lot of parameters.

    def __init__(self, n_units, components, concentration, ins_penalty,
                 dp_prior=False):
        """Initialize the Phone Loop.

        Parameters
        ----------
        n_units : int
            Number of units in the phone loop.
        components : list
            List of emissions.
        concentration : float
            Concentration parameter of the Dirichlet Process or the
            Dirichlet distribution prior.
        ins_penalty : float
            Insertion penalty. Values greater than 1 will prefer to remain
            in the current unit whereas values lower than 1 (and greater
            than 0) will favorize unit to unit transition.

        """
        # pylint: disable=too-many-arguments
        # There are no unnecessary arguments.

        super().__init__()
        self.n_units = n_units
        self.n_states = int(len(components) / n_units)
        self.components = components
        self.dp_prior = dp_prior
        if dp_prior:
            self.hg1 = np.ones(n_units)
            self.hg2 = np.zeros(n_units) + concentration
            self.pg1 = np.ones(n_units)
            self.pg2 = np.zeros(n_units) + concentration
        else:
            self.prior_count = np.ones(n_units) * concentration
            self.posterior_count = np.ones(n_units) * concentration
        weights = np.ones(self.n_units) / self.n_units
        self.log_trans_mat, self.init_states, self.final_states = \
            __log_transition_matrix(n_units, self.n_states, weights,
                                    ins_penalty)
        self.ins_penalty = ins_penalty
        self.optimal_order_idx = None

    def expected_log_weights(self):
        """Expected value of the log of the weights of the DP.

        Returns
        -------
        E_log_pi : float
            Log weights.

        """
        if self.dp_prior:
            breaks = psi(self.pg1) - psi(self.pg1 + self.pg2)
            remainders = psi(self.pg2) - psi(self.pg1 + self.pg2)
            retval = breaks.copy()
            for i in range(1, self.n_units):
                retval[i] += remainders[:i].sum()
        else:
            retval = psi(self.posterior_count) - \
                psi(self.posterior_count.sum())
        if self.optimal_order_idx is not None:
            return retval[self.optimal_order_idx]
        return retval

    def get_stats(self, data, weights, state_weights):
        """Compute the sufficient statistics for the model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.
        weights : numpy.ndarray
            Weights for each frame.
        state_resps : numpy.ndarray
            Per-state weights for each frame.

        Returns
        -------
        stats : dict
            Dictionary where 's0', 's1' and 's2' are the keys for the
            zeroth, first and second order statistics respectively.

        """
        stats_data = {}
        stats_data[self.uid] = {}

        # WARNING: this code is wrong.
        tmp = weights.reshape((weights.shape[0], self.n_units, -1)).sum(axis=2)
        stats_data[self.uid]['s0'] = tmp.sum(axis=0)

        stats_1 = np.zeros_like(stats_data[self.uid]['s0'])
        for i in range(len(stats_data[self.uid]['s0']) - 1):
            stats_1[i] += stats_data[self.uid]['s0'][i + 1:].sum()
        stats_data[self.uid]['s1'] = stats_1
        for i, component in enumerate(self.components):
            stats_data[component.uid] = {}
            comp_weights = (weights[:, i] * state_weights[i].T).T
            stats_data = {**stats_data, **component.get_stats(data,
                                                              comp_weights)}
        return stats_data

    def forward(self, llhs):
        """Forward recursion.

        Parameters
        ----------
        llhs : numpy.ndarray
            (Expected) log-likelihood of each emissions per frame.

        Returns
        -------
        log_alphas : numpy.ndarray
            Log of the results of the forward recursion.

        """
        log_alphas = np.zeros_like(llhs) - np.inf
        log_alphas[0, self.init_states] = self.expected_log_weights()
        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += logsumexp(log_alphas[i-1] + \
                                       self.log_trans_mat.T, axis=1)
        return log_alphas

    def backward(self, llhs):
        """Backward recursion.

        Parameters
        ----------
        llhs : numpy.ndarray
            (Expected) log-likelihood of each emissions per frame.

        Returns
        -------
        log_betas : numpy.ndarray
            Log of the results of the backward recursion.

        """
        log_betas = np.zeros_like(llhs) - np.inf
        log_betas[-1, self.final_states] = 0.
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = logsumexp(self.log_trans_mat + llhs[i+1] + \
                                     log_betas[i+1], axis=1)
        return log_betas

    def viterbi(self, llhs):
        """Find the most likely sequence using the
        the Viterbi algorithm.

        Parameters
        ----------
        llhs : numpy.ndarray
            (Expected) log-likelihood of each emissions per frame.

        Returns
        -------
        path : list
            List of indices of the most likely state sequence.

        """
        backtrack = np.zeros_like(llhs, dtype=int)
        omega = np.zeros(llhs.shape[1]) + float('-inf')
        omega[self.init_states] = llhs[0, self.init_states] + \
            self.expected_log_weights()
        for i in range(1, llhs.shape[0]):
            hypothesis = omega + self.log_trans_mat.T
            backtrack[i] = np.argmax(hypothesis, axis=1)
            omega = llhs[i] + hypothesis[range(len(self.log_trans_mat)),
                                         backtrack[i]]
        path = [self.final_states[np.argmax(omega[self.final_states])]]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])
        return path

    def decode(self, data):
        """Find the most likely sequence of units given the data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.

        Returns
        -------
        path : list
            List of indices of the most likely state sequence.

        """
        c_llhs = np.zeros((data.shape[0], self.n_states * self.n_units))
        for k in range(self.n_states * self.n_units):
            c_llh = self.components[k].expected_log_likelihood(data)
            c_llhs[:, k] = logsumexp(c_llh, axis=1)
        path = self.viterbi(c_llhs)
        path = [bisect(self.init_states, state) for state in path]
        return path

    def mask_from_alignments(self, data, ali):
        """Return a mask to filter the possible path while computing
        the forward-backward recursion.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.
        ali : list of tuple
            Unit level alignment.

        Returns
        -------
        mask : numpy.ndarray
            Matrix where 1s' corresponds to allowed paths.

        """
        mask = np.zeros((data.shape[0], self.n_states * self.n_units))
        for entry in ali:
            index = int(entry[0])
            start = entry[1]
            end = entry[2]
            if end > data.shape[0]:
                break
            tmp = np.zeros((end - start, self.n_states * self.n_units)) \
                + float('-inf')
            tmp[:, index * self.n_states: (index + 1) * self.n_states] = 0.
            mask[start:end] = tmp
        return mask

    def expected_log_likelihood(self, data, ali=None):
        """Expected value of the log likelihood.

        If the unit sequence is provided, apply a mask to narrow
        the possible alignments. This is not very efficient as
        the log-likelihood matrix will be very sparse but it keeps
        the code simple.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.
        ali : list of tuple
            Unit level alignment (optional).

        Returns
        -------
        E_llh : float
            Expected value of the log-likelihood.
        state_resps : numpy.ndarray
            Per-state responsibility.

        """
        if ali is not None:
            mask = self.mask_from_alignments(data, ali)
        c_llhs = np.zeros((data.shape[0], self.n_states * self.n_units))
        comp_resps = []
        for k in range(self.n_states * self.n_units):
            c_llh = self.components[k].expected_log_likelihood(data)
            c_llhs[:, k] = logsumexp(c_llh, axis=1)
            resps = np.exp((c_llh.T - c_llhs[:, k]).T)
            comp_resps.append(resps)
        if ali is not None:
            c_llhs += mask
        log_alphas = self.forward(c_llhs)
        log_betas = self.backward(c_llhs)
        log_q_z = log_alphas + log_betas
        norm = logsumexp(log_q_z, axis=1)
        log_q_z = (log_q_z.T - norm).T
        return norm, np.exp(log_q_z), comp_resps

    def reorder(self):
        """Reorder the units so that the most frequent have
        a small index. This is needed when the weights of
        the units have a Dirichlet Process prior.

        """
        self.optimal_order_idx = None
        expected_log_w = self.expected_log_weights()
        idx = expected_log_w.argsort()[::-1]
        new_components = []
        for i in idx:
            start = i * self.n_states
            for k in range(start, start + self.n_states, 1):
                new_components.append(self.components[k])
        self.components = new_components
        self.optimal_order_idx = idx

    def update(self, stats):
        """Update the posterior parameters given the sufficient
        statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.

        """
        if self.dp_prior:
            self.pg1 = self.hg1 + stats[self.uid]['s0']
            self.pg2 = self.hg2 + stats[self.uid]['s1']
        else:
            self.posterior_count = self.prior_count + stats[self.uid]['s0']
        for component in self.components:
            component.update(stats)
        if self.dp_prior:
            self.reorder()
        expected_log_w = self.expected_log_weights()
        expected_log_w -= logsumexp(expected_log_w)
        prob_fs = np.exp(self.log_trans_mat[self.final_states[0],
                                            self.final_states[0]])
        for final_state in self.final_states:
            if self.n_states > 1:
                self.log_trans_mat[final_state, self.init_states] = \
                    self.ins_penalty * (np.log((1 - prob_fs)) + expected_log_w)
            else:
                self.log_trans_mat[final_state, self.init_states] = \
                    expected_log_w

    def kl_divergence(self):
        """Kullback-Leibler divergence between the posterior and
        the prior density.

        Returns
        -------
        ret : float
            KL(q(params) || p(params)).

        """
        kl_div = 0.
        if self.dp_prior:
            for i in range(self.pg1.shape[0]):
                val1 = np.array([self.pg1[i], self.pg2[i]])
                val2 = np.array([self.hg1[i], self.hg2[i]])
                kl_div += gammaln(np.sum(val1)) - gammaln(np.sum(val2)) - \
                    gammaln(val1).sum() + gammaln(val2).sum()
        else:
            tmp_mixture = Mixture([], self.prior_count)
            tmp_mixture.posterior_count = self.posterior_count
            kl_div = tmp_mixture.kl_divergence()
        for component in self.components:
            kl_div += component.kl_divergence()
        return kl_div
