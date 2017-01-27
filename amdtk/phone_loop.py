
"""Phone-Loop model where each unit is modeled by a left-to-right HMM."""

from bisect import bisect
from itertools import groupby
import numpy as np
from scipy.misc import logsumexp
from scipy.special import psi, gammaln
from .model import Model


def _sample_state(log_prob_trans, log_alpha):
    log_prob = log_alpha + log_prob_trans
    log_prob -= logsumexp(log_prob)
    return np.random.choice(log_prob.shape[0], p=np.exp(log_prob))


def _indices(matrix, threshold=-100):
    # Prune the matrix.
    idx0, idx1 = np.where(matrix > threshold)

    retval = []
    for n in range(len(matrix)):
        idx = np.where(idx0 == n)
        retval.append(idx1[idx])

    return np.array(retval)


def _prune(vector, pruning):
    return vector.argsort()[::-1][:int(pruning)]


def _trim(llhs, n_states, final_states):
    mask = np.zeros_like(llhs[-1], dtype=bool) + True
    mask[final_states] = False
    for i in range(n_states - 1):
        mask[final_states - i] = False
        llhs[-(i + 1), mask] = float('-inf')


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

    @staticmethod
    def __log_transition_matrix(n_units, n_states, mixture_weights, ins_penalty=1.,
                                prob_final_state=.5):
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
        for final_state in final_states:
            if n_states > 1:
                trans_mat[final_state, init_states] = ins_penalty * \
                    np.log((1 - prob_final_state) * mixture_weights)
            else:
                trans_mat[final_state, init_states] = np.log(mixture_weights)
        return trans_mat, init_states, final_states

    def __init__(self, n_units, components, concentration, ins_penalty,
                 pruning_threshold=100):
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
        pruning_threshold : float
            Pruning threshold (default: 100).

        """
        # pylint: disable=too-many-arguments
        # There are no unnecessary arguments.

        super().__init__()
        self.n_units = n_units
        self.n_states = int(len(components) / n_units)
        self.components = components
        self.hg1 = np.ones(n_units)
        self.hg2 = np.zeros(n_units) + concentration
        self.pg1 = np.ones(n_units)
        self.pg2 = np.zeros(n_units) + concentration

        expected_log_w = self.expected_log_weights()
        expected_log_w -= logsumexp(expected_log_w)
        weights = np.exp(expected_log_w)

        self.log_trans_mat, self.init_states, self.final_states = \
            PhoneLoop.__log_transition_matrix(n_units, self.n_states, weights,
                                    ins_penalty)

        self.ins_penalty = ins_penalty
        self.pruning_threshold = pruning_threshold
        self.trans_idx = _indices(self.log_trans_mat, threshold=float('-inf'))
        self.optimal_order_idx = None

    def expected_log_weights(self):
        """Expected value of the log of the weights of the DP.

        Returns
        -------
        E_log_pi : float
            Log weights.

        """
        breaks = psi(self.pg1) - psi(self.pg1 + self.pg2)
        remainders = psi(self.pg2) - psi(self.pg1 + self.pg2)
        retval = breaks.copy()
        for i in range(1, self.n_units):
            retval[i] += remainders[:i].sum()
        return retval

    def get_stats(self, data, units_stats, weights, state_weights):
        """Compute the sufficient statistics for the model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.
        units_stats : numpy.ndarray
            Expected count for the units.
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

        # Counts of the units.
        stats_data[self.uid]['s0'] = units_stats
        for i, component in enumerate(self.components):
            stats_data[component.uid] = {}
            comp_weights = (weights[:, i] * state_weights[i].T).T
            stats_data = {**stats_data, **component.get_stats(data,
                                                              comp_weights)}

        return stats_data

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
        neg_inf = float('-inf')
        init_states = self.init_states
        trans_idx = self.trans_idx
        threshold = self.pruning_threshold
        log_prob_init = self.expected_log_weights()

        hypothesis = np.zeros_like(self.log_trans_mat)
        backtrack = np.zeros_like(llhs, dtype=int)
        omega = np.zeros(llhs.shape[1]) + float('-inf')
        omega[init_states] = llhs[0, init_states] + \
            log_prob_init

        # Indices of the surviving path at the previous time step.
        idx_t0 = init_states[_prune(omega[init_states], threshold)]

        # Indices of the possible path at the current step.
        idx_t1 = np.unique(np.hstack(trans_idx[idx_t0]))

        for i in range(1, llhs.shape[0]):
            trans = self.log_trans_mat[idx_t0[:, np.newaxis], idx_t1]
            hypothesis[idx_t1[:, np.newaxis], idx_t0] = (omega[idx_t0] + trans.T)
            backtrack[i, idx_t1] = idx_t0[
                np.argmax(hypothesis[idx_t1[:, np.newaxis], idx_t0], axis=1)]
            omega.fill(neg_inf)
            omega[idx_t1] = llhs[i, idx_t1] \
                + hypothesis[idx_t1, backtrack[i, idx_t1]]

            # Update the indices of the surviving paths.
            idx_t0 = idx_t1[_prune(omega[idx_t1], threshold)]

            idx_t1 = np.unique(np.hstack(trans_idx[idx_t0]))

        path = [self.final_states[np.argmax(omega[self.final_states])]]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])

        return path, omega[path[-1]]

    def decode(self, data, state_path=False):
        """Find the most likely sequence of units given the data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data (N x D) of N frames with D dimensions.
        state_path : boolean
            If True, return the state path instead of the unit path.

        Returns
        -------
        path : list
            List of indices of the most likely state sequence.

        """
        c_llhs = np.zeros((data.shape[0], self.n_states * self.n_units))
        for k in range(self.n_states * self.n_units):
            c_llh = self.components[k].expected_log_likelihood(data)
            c_llhs[:, k] = logsumexp(c_llh, axis=1)
        path, _ = self.viterbi(c_llhs)
        if not state_path:
            path = [bisect(self.init_states, state) for state in path]
        path = ['a' + str(x[0]+1) for x in groupby(path)]
        return path

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
        idxs : list
            List of the indices of the surviving paths.

        """
        pruning = self.pruning_threshold
        trans_idx = self.trans_idx
        log_alphas = np.zeros_like(llhs) - np.inf
        log_alphas[0, self.init_states] = self.expected_log_weights()

         # Indices of the surviving path at the previous time step.
        idx_t0 = self.init_states[_prune(log_alphas[0, self.init_states], pruning)]

        # Indices of the possible path at the current step.
        idx_t1 = np.unique(np.hstack(trans_idx[idx_t0]))

        idxs = [idx_t0]
        for i in range(1, llhs.shape[0]):
            trans = self.log_trans_mat[idx_t0[:, np.newaxis], idx_t1]
            log_alphas[i, idx_t1] = llhs[i, idx_t1] \
                + logsumexp(trans.T + log_alphas[i-1, idx_t0], axis=1)

            # Pruning: discard paths that are unlikely.
            idx_t0 = idx_t1[_prune(log_alphas[i, idx_t1], pruning)]
            idxs.append(idx_t0)

            # From the suriving paths, get the possible transitions.
            idx_t1 = np.unique(np.hstack(trans_idx[idx_t0]))

        return log_alphas, idxs


    def backward(self, llhs, idxs):
        """Backward recursion.

        Parameters
        ----------
        llhs : numpy.ndarray
            (Expected) log-likelihood of each emissions per frame.
        idxs : List
            List of the indices of the surviving paths from the
            forward step.

        Returns
        -------
        log_betas : numpy.ndarray
            Log of the results of the backward recursion.

        """
        trans_idx = self.trans_idx

        # Indices of the surviving path at the next time step.
        idx_t0 = self.final_states
        log_betas = np.zeros_like(llhs) - np.inf
        log_betas[-1, self.final_states] = 0.
        for i in reversed(range(llhs.shape[0]-1)):
            idx_t0 = idxs[i]
            idx_t1 = idxs[i + 1]
            trans = self.log_trans_mat[idx_t0[:, np.newaxis], idx_t1]
            log_betas[i, idx_t0] = logsumexp(trans + log_betas[i+1, idx_t1] \
                                             + llhs[i+1, idx_t1], axis=1)

        return log_betas

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

    def viterbi_exp(self, data, ali=None):
        """Expectation step of the Viterbi training.

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
        units_stats : numpy.ndarray
            Expected count for the units.
        state_resps : numpy.ndarray
            Per state responsibility.
        comp_resps : numpy.ndarray
            Per sate component responsibility.

        """
        c_llhs = np.zeros((data.shape[0], self.n_states * self.n_units))
        comp_resps = []
        for k in range(self.n_states * self.n_units):
            c_llh = self.components[k].expected_log_likelihood(data)
            c_llhs[:, k] = logsumexp(c_llh, axis=1)
            resps = np.exp((c_llh.T - c_llhs[:, k]).T)
            comp_resps.append(resps)

        if ali is not None:
            mask = self.mask_from_alignments(data, ali)
            c_llhs += mask

        # Trim the end of the log-likelihood to make sure that
        # the pruning will still keep a valid path.
        _trim(c_llhs, self.n_states, self.final_states)

        # Compute the best path.
        state_path, llh = self.viterbi(c_llhs)
        state_path = np.array(state_path)

        # Convert the state path to a unit path.
        path = [bisect(self.init_states, state) for state in state_path]
        path = np.array([x[0] for x in groupby(path)])

        return llh, path, state_path, comp_resps

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
        units_stats : numpy.ndarray
            Expected count for the units.
        state_resps : numpy.ndarray
            Per state responsibility.
        comp_resps : numpy.ndarray
            Per sate component responsibility.

        """
        c_llhs = np.zeros((data.shape[0], self.n_states * self.n_units))
        comp_resps = []
        for k in range(self.n_states * self.n_units):
            c_llh = self.components[k].expected_log_likelihood(data)
            c_llhs[:, k] = logsumexp(c_llh, axis=1)
            resps = np.exp((c_llh.T - c_llhs[:, k]).T)
            comp_resps.append(resps)

        if ali is not None:
            mask = self.mask_from_alignments(data, ali)
            c_llhs += mask

        # Trim the end of the log-likelihood to make sure that
        # the pruning will still keep a valid path.
        _trim(c_llhs, self.n_states, self.final_states)

        log_alphas, idxs = self.forward(c_llhs)
        log_betas = self.backward(c_llhs, idxs)
        log_q_z = log_alphas + log_betas
        norm = logsumexp(log_q_z[-1])
        log_q_z = log_q_z - norm
        units_stats = self.units_stats(c_llhs, log_alphas, log_betas)

        return norm, units_stats, np.exp(log_q_z), comp_resps

    def units_stats(self, c_llhs, log_alphas, log_betas):
        """Extract the statistics needed to re-estimate the
        weights of the units.

        Parameters
        ----------
        c_llhs : numpy.ndarray
            Emissions log-likelihood.
        log_alphas : numpy.ndarray
            Log of the results of the forward recursion.
        log_betas : numpy.ndarray
            Log of the results of the backward recursion.

        Returns
        -------
        units_stats : numpy.ndarray
            Units' statistics.

        """
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

    def update(self, stats, scale=1.):
        """Update the posterior parameters given the sufficient
        statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.
        scale : float
            Scaling factors of the statistics.

        """
        stats_0 = stats[self.uid]['s0'] * scale
        stats_1 = np.zeros_like(stats_0)
        for i in range(len(stats_0) - 1):
            stats_1[i] += stats_0[i + 1:].sum()
        self.pg1 = self.hg1 + stats_0
        self.pg2 = self.hg2 + stats_1

        for component in self.components:
            component.update(stats, scale)

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

    def natural_grad_update(self, stats, scale, lrate):
        """Natural gradient update of the posterior parameters given
        the sufficient statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.
        scale : float
            Scaling factors of the statistics.

        """
        stats_0 = stats[self.uid]['s0'] * scale
        stats_1 = np.zeros_like(stats_0)
        for i in range(len(stats_0) - 1):
            stats_1[i] += stats_0[i + 1:].sum()
            self.pg1 += lrate * (-self.pg1 + self.hg1 + stats_0)
            self.pg2 += lrate * (-self.pg2 + self.hg2 + stats_1)

        for component in self.components:
            component.natural_grad_update(stats, scale, lrate)

        # Update the probabilty of the initial states.
        expected_log_w = self.expected_log_weights()
        expected_log_w -= logsumexp(expected_log_w)

        # Update the log probability transition matrix.
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
        for i in range(self.pg1.shape[0]):
            val1 = np.array([self.pg1[i], self.pg2[i]])
            val2 = np.array([self.hg1[i], self.hg2[i]])
            kl_div += gammaln(np.sum(val1)) - gammaln(np.sum(val2)) - \
            gammaln(val1).sum() + gammaln(val2).sum()
        for component in self.components:
            kl_div += component.kl_divergence()
        return kl_div
