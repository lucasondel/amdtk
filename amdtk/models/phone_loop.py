
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


class BayesianInfinitePhoneLoop(object):

    def __init__(self, tdp, dgraph, acoustic_model):
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
        self.prior = tdp
        self.posterior = copy.deepcopy(tdp)
        self.dgraph = dgraph
        self.acoustic_model = acoustic_model

        # Use the prior over the units to set transitions of the
        # decoding graph.
        self.updateParams()

    def updateParams(self):
        """Update the parameters of the model."""
        pass
        # TODOs :
        #   * update the decoding graph
        #   * update the GMMs' posterior

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

        for t in range(1, len(llhs)):
            for name in self.dgraph.names:
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

        for t in reversed(range(llhs.shape[0]-1)):
            for name in self.dgraph.names:
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

        # Evaluate the responsibilities for each units.
        dim = self.prior.truncation
        nframes = len(am_llhs)
        resp_units = np.exp(log_P_Z).reshape((nframes, dim, -1)).sum(axis=2)

        # log-likelihood of the sequence. We compute it to monitor the
        # convergence of the training.
        E_log_P_X = logsumexp(log_alphas[-1])

        return E_log_P_X, log_P_Z, resp_units

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
