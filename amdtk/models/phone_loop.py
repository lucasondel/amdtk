
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
from ..models import MixtureStats
from ..models import GaussianDiagCovStats

# Watchout before changing this as we expect the silence name to be
# lower (alphabetically speaking) than the unit prefix.
UNIT_PREFIX = 'a'
SILENCE_NAME = 'sil'


class BayesianInfinitePhoneLoop(object):
    """Bayesian Infinite Phone Loop model.

    Attributes
    ----------
    prior : :class:`TruncatedDirichletProcess`
        Prior over the weights of each phone (i.e. unit) model.
    posterior : :class:`TruncatedDirichletProcess`
        Posterior over the weights of each phone (i.e. unit) model.
    dgraph : :class:`HmmGraph`
        HMM model corresponding to the phone loop model.
    name_model : dict
        Mapping "hmm state name" -> "emission probability model".
    id_model : dict
        Mapping "gmm unique id" -> "emission probability model".
    name_id : dict
        Mapping "hmm state name" -> "gmm unique id"
    unit_names : list
        Name assoctiated for each unit.

    """

    def __init__(self, truncation, concentration, nstates, gmms,
                 silence_model=None):
        """Create a (infinite) phone loop model.

        Parameters
        ----------
        truncation : int
            Order of the truncation for the Truncated Dirichlet Process
            posterior.
        concentration : float
            Concentration parameter of the Dirichlet Process prior.
        nstates : int
            Number of states for each HMM representing a single unit.
        silence_model : model
            If provided, the silence will be model explicitely in the
            phone-loop. The silence state will be the only initial
            state and final state of the HMM.

        """
        g1 = np.ones(truncation)
        g2 = np.zeros(truncation) + concentration
        tdp = TruncatedDirichletProcess(g1, g2)

        if silence_model is not None:
            nunits = truncation - 1
        else:
            nunits = truncation

        self.prior = tdp
        self.posterior = copy.deepcopy(tdp)
        self.dgraph = HmmGraph.standardPhoneLoop(UNIT_PREFIX, nunits,
                                                 nstates)

        if silence_model is not None:
            self.dgraph.addSilenceState(SILENCE_NAME, nstates)
            models = [silence_model] * nstates
            models += gmms
        else:
            models = gmms

        unit_names, state_names = self.dgraph.names
        self.name_model = {}
        self.id_model = {}
        self.name_id = {}
        for i, state_name in enumerate(sorted(state_names, reverse=True)):
            self.name_model[state_name] = models[i]
            self.id_model[i] = models[i]
            self.name_id[state_name] = i
        self.dgraph.setEmissions(self.name_model)

        self.unit_names = sorted(list(set(unit_names)), reverse=True)
        self.unit_name_index = {}
        for i, unit_name in enumerate(self.unit_names):
            self.unit_name_index[unit_name] = i

        self.bigram = None

        self.updateWeights()

    def setBigramLM(self, bigram):
        self.bigram = bigram
        self.updateWeights()

    def updateWeights(self):
        """Update the weights of the phone loop."""
        if self.bigram is None:
            log_pi = self.posterior.expLogPi()
            weights = {}
            for i, unit_name in enumerate(self.unit_names):
                weights[unit_name] = log_pi[i]
            self.dgraph.setUnigramWeights(weights)
        else:
            self.dgraph.setBigramWeights(self.bigram)

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
        return self.dgraph.evaluateEmissions(X)

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
        log_P_Z : numpy.ndarray
            Log responsibility per frame for each state of the HMM.
        resp_units ; numpy.ndarray
            The responsibility for each unit per frame.

        """
        # Compute the forward-backward algorithm.
        log_alphas = self.dgraph.forward(am_llhs)
        log_betas = self.dgraph.backward(am_llhs)
        log_P_Z = log_alphas + log_betas
        log_P_Z = (log_P_Z.T - logsumexp(log_P_Z, axis=1)).T
        P_Z = np.exp(log_P_Z)

        # Evaluate the responsibilities for each units.
        resp_units = np.zeros((len(am_llhs), len(self.unit_names)),
                              dtype=float)
        for idx, state in enumerate(self.dgraph.states):
            unit_idx = self.unit_name_index[state.parent_name]
            resp_units[:, unit_idx] += P_Z[:, idx]

        # log-likelihood of the sequence. We compute it to monitor the
        # convergence of the training.
        E_log_P_X = logsumexp(log_alphas[-1])

        return E_log_P_X, log_P_Z, resp_units

    def viterbi(self, am_llhs, output_states=False):
        """Viterbi algorithm.

        Parameters
        ----------
        am_llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM model.
        output_states : boolean
            If true, the path will have the name of the HMM states
            rather than the units' name (default: False).

        Returns
        -------
        path : list
            Ordered sequence of unit name corresponding to the most
            likely sequence.

        """
        return self.dgraph.viterbi(am_llhs, not output_states)

    def stats(self, X, unit_log_resps, hmm_log_resps, am_log_resps):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.
        unit_log_resps : numpy.ndarray
            Responsility for each unit of the model.
        hmm_log_resps : numpy.ndarray
            Responsibility for each state of the hmm.
        am_log_resps : nlist of umpy.ndarray
            Responsibility for each gmm of the model.

        Returns
        -------
        tdp_stats : :class:`DirichletProcessStats`
            Statistics for the (Truncated) Dirichlet Process posterior.
        gmm_stats : dict
            Statistics for each GMM of the HMM.
        gauss_stats : dict
            Statistics for each Gaussian of the HMM.

        """
        tdp_stats = DirichletProcessStats(np.exp(unit_log_resps))

        gmm_stats = {}
        gauss_stats = {}
        for i, state in enumerate(self.dgraph.states):
            gmm = state.model
            log_weights = (hmm_log_resps[:, i] + am_log_resps[i].T).T
            weights = np.exp(log_weights)
            key = state.name
            if key not in gmm_stats.keys():
                gmm_stats[key] = MixtureStats(weights)
                gauss_stats[key] = {}
                for j in range(gmm.k):
                    gauss_stats[key][j] = \
                        GaussianDiagCovStats(X, weights[:, j])
            else:
                gmm_stats[key] += MixtureStats(weights)
                for j in range(gmm.k):
                    gauss_stats[key][j] += \
                        GaussianDiagCovStats(X, weights[:, j])


        return tdp_stats, gmm_stats, gauss_stats

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        KL = 0
        for name, gmm in self.name_model.items():
            KL += gmm.KLPosteriorPrior()
        return KL + self.posterior.KL(self.prior)

    def updatePosterior(self, tdp_stats, gmm_stats, gauss_stats):
        """Update the parameters of the posterior distributions
        according to the accumulated statistics.

        Parameters
        ----------
        tdp_stats : :class:`DirichletProcessStats`
            Statistics for the (Truncated) Dirichlet Process posterior.
        gmm_stats : dict
            Statistics for each GMM of the HMM.
        gauss_stats : dict
            Statistics for each Gaussian of the HMM.

        """
        self.posterior = self.prior.newPosterior(tdp_stats)
        for name, stats in gmm_stats.items():
            gmm = self.name_model[name]
            gmm.updatePosterior(stats)

        for name, data in gauss_stats.items():
            gmm = self.name_model[name]
            for j, stats in data.items():
                gmm.components[j].updatePosterior(stats)
        self.updateWeights()

