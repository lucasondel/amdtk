
"""Non-parametric Bayesian phone-loop model."""

# NOTE:
# Differing from what has been described in the paper, we do not put any
# prior on the transition matrix as:
#  * optimizing the states of the sub HMM does not help and is
#    time consuming
#  * transition between sub-HMMs are hanlded by weights taken
#    from the Dirichlet process.

import copy
import numpy as np
from scipy.misc import logsumexp
from ..models import DirichletProcessStats
from ..models import TruncatedDirichletProcess
from ..models import HmmGraph
from ..models import AcousticModel

UNIT_PREFIX = 'a'


class BayesianInfinitePhoneLoop(object):

    def __init__(self, truncation, concentration, nstates, gmms,
                 silence_model=None):
        """Create a (infinite) phone loop model.

        Parameters
        ----------
        tdp : :class:`TruncatedDirichletProcess`
            Truncated DirichletProcess prior.
        dgraph : :class:`DecodableGraph`
            Fst structure to use.
        acoustic_model : :class:`AcousticModel`
            Acoustic model corresponding to the decodable graph.

        """
        g1 = np.ones(truncation)
        g2 = np.zeros(truncation) + concentration
        tdp = TruncatedDirichletProcess(g1, g2)

        if silence_model is not None:
            nunits = truncation - 1
            self.has_sil_model = True
        else:
            nunits = truncation
            self.has_sil_model = False

        self.prior = tdp
        self.posterior = copy.deepcopy(tdp)
        self.dgraph = HmmGraph.standardPhoneLoop(UNIT_PREFIX, nunits,
                                                 nstates)

        if silence_model is not None:
            self.dgraph.addSilenceState('sil')
            models = gmms + [silence_model]
        else:
            models = gmms

        unit_names, state_names = self.dgraph.names
        self.acoustic_model = AcousticModel(unit_names, state_names, models)

        self.unit_names = sorted(list(set(unit_names)))
        self.updateParams()

    def updateParams(self):
        """Update the parameters of the model."""
        log_pi = self.posterior.expLogPi()
        pi = np.exp(log_pi)
        pi /= pi
        weights = {}

        for i, unit_name in enumerate(self.unit_names):
            weights[unit_name] = pi[i]

        self.dgraph.setUnigramWeights(weights)

    def evalAcousticModel(self, X):
        """Compute the expected value of the log-likelihood of the
        acoustic model of the phone loop.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.

        Returnsgmm_log_P_Zs
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.
        data : object
            Additional data that the model can re-use when accumulating
            the statistics.

        """
        return self.acoustic_model.evaluate(X)

    def forward(self, llhs):
        name_index = self.acoustic_model.name_index
        index_name = self.acoustic_model.index_name
        log_alphas = llhs.copy()
        log_alphas[0] += self.dgraph.logProbInit(index_name)

        unit_names, state_names = self.dgraph.names
        self.dgraph.setActiveStates(self.dgraph.init_states)
        for t in range(1, len(llhs)):
            for name in state_names:
                idx = self.acoustic_model.name_index[name]
                log_trans = self.dgraph.logProbTransitions(name, name_index)
                log_alphas[t, idx] += logsumexp(log_alphas[t-1] + log_trans)

        return log_alphas

    def backward(self, llhs):
        name_index = self.acoustic_model.name_index
        log_betas = np.zeros_like(llhs) - float('inf')
        for name in self.dgraph.finalNames():
            idx = name_index[name]
            log_betas[-1, idx] = 0.

        unit_names, state_names = self.dgraph.names
        self.dgraph.setActiveStates(self.dgraph.final_states)
        for t in reversed(range(llhs.shape[0]-1)):
            for name in state_names:
                idx = self.acoustic_model.name_index[name]
                log_trans = self.dgraph.logProbTransitions(name, name_index,
                                                           incoming=False)
                log_betas[t, idx] = logsumexp(log_betas[t+1] + log_trans +
                                              llhs[t+1])

        return log_betas

    def forwardBackward(self, am_llhs):
        """Forward-backward algorithm of the phone-loop.

        Parameters
        ----------
        am_llhs : numpy.ndarray
            Acoustic model log likelihood.

        Returns
        -------
        E_log_P_X : float
            The expected value of log probability of the sequence of
            features.
        resp_units ; numpy.ndarray
            The responsibility for each unit per frame.

        """
        # Compute the forward-backward algorithm.
        old_settings = np.seterr(divide='ignore')
        log_alphas = self.forward(am_llhs)
        log_betas = self.backward(am_llhs)
        np.seterr(**old_settings)
        log_P_Z = log_alphas + log_betas
        log_P_Z = (log_P_Z.T - logsumexp(log_P_Z, axis=1)).T
        P_Z = np.exp(log_P_Z)

        # Evaluate the responsibilities for each units.
        resp_units = np.zeros((len(am_llhs), len(self.unit_names)),
                              dtype=float)
        for unit_name, idxs in self.acoustic_model.parent_name_indices.items():
            unit_idx = self.unit_names.index(unit_name)
            for index in idxs:
                resp_units[:, unit_idx] += P_Z[:, index]

        # nframes = len(am_llhs)
        # resp_units = np.exp(log_P_Z).reshape((nframes, dim, -1)).sum(axis=2)

        # log-likelihood of the sequence. We compute it to monitor the
        # convergence of the training.
        E_log_P_X = logsumexp(log_alphas[-1])

        return E_log_P_X, log_P_Z, resp_units

    def viterbi(self, am_llhs):
        """Viterbi algorithm.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM model.

        Returns
        -------
        path : list
            List of the state of the mostr probable path.

        """
        name_index = self.acoustic_model.name_index
        index_name = self.acoustic_model.index_name
        backtrack = np.zeros_like(am_llhs, dtype=int)
        log_omegas = np.zeros_like(am_llhs, dtype=float) - float('inf')
        log_omegas[0] = am_llhs[0] + self.dgraph.logProbInit(index_name)

        unit_names, state_names = self.dgraph.names
        self.dgraph.setActiveStates(self.dgraph.init_states)
        import pdb
        pdb.set_trace()
        for t in range(1, am_llhs.shape[0]):
            for name in state_names:
                idx = self.acoustic_model.name_index[name]
                log_trans = self.dgraph.logProbTransitions(name, name_index)
                hypothesis = log_omegas[t - 1] + log_trans
                backtrack[t, idx] = np.argmax(hypothesis)
                log_omegas[t, idx] += np.max(hypothesis)

        import pdb
        pdb.set_trace()
        path = [self.final_states[np.argmax(log_omegas[self.final_states])]]
        for i in reversed(range(1, len(am_llhs))):
            path.insert(0, backtrack[i, path[0]])
        return path

    def stats(self, X, unit_log_resps, hmm_log_resps, am_log_resps):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.
        am_log_resps : nlist of umpy.ndarray
            Responsibility for each gmm of the model.
        hmm_log_resps : numpy.ndarray
            Responsibility for each state of the hmm.
        unit_log_resps : numpy.ndarray
            Responsility for each unit of the model.

        Returns
        -------
        E_log_P_X : float
            The expected value of log probability of the sequence of
            features.
        resp_units ; numpy.ndarray
            The responsibility for each unit per frame.

        """
        # Evaluate the statistics for the truncated DP.
        tdp_stats = DirichletProcessStats(np.exp(unit_log_resps))

        # Evaluate the statistics of the acoustic model.
        gmm_stats, gauss_stats = self.acoustic_model.stats(X, hmm_log_resps,
                                                           am_log_resps)

        return tdp_stats, gmm_stats, gauss_stats

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns:class:MixtureStats
        -------
        KL : float
            KL divergence.

        """
        return self.acoustic_model.KLPosteriorPrior() + \
            self.posterior.KL(self.prior)

    def updatePosterior(self, tdp_stats, gmm_stats, gauss_stats):
        """Update the parameters of the posterior distribution according
        to the accumulated statistics.

        Parameters
        ----------
        tdp_stats : :class:MixtureStats
            Statistics for the truncated DP.

        """
        self.posterior = self.prior.newPosterior(tdp_stats)
        self.acoustic_model.updatePosterior(gmm_stats, gauss_stats)
        self.updateParams()
