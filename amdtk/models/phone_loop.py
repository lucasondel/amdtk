
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
        log_alphas = self.dgraph.forward(am_llhs)
        log_betas = self.dgraph.backward(am_llhs)
        log_P_Z = log_alphas + log_betas
        log_P_Z = (log_P_Z.T - logsumexp(log_P_Z, axis=1)).T

        # Evaluate the responsibilities for each units.
        dim = self.dgraph.nunits
        nframes = len(am_llhs)
        resp_units = np.exp(log_P_Z).reshape((nframes, dim, -1)).sum(axis=2)

        # log-likelihood of the sequence. We compute it to monitor the
        # convergence of the training.
        E_log_P_X = logsumexp(log_alphas[-1])

        return E_log_P_X, log_P_Z, resp_units

    def stats(self, X, am_log_resps, hmm_log_resps, unit_log_resps):
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
        gmm_stats, am_stats = self.acoustic_model.stats(X, am_log_resps,
                                                        hmm_log_resps)

        return tdp_stats, gmm_stats, am_stats

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns:class:MixtureStats
        -------
        KL : float
            KL divergence.

        """
        KL = 0
        for gmm in self.components:
            KL += gmm.KLPosteriorPrior()
        return KL + self.posterior.KL(self.prior)

    def updatePosterior(self, tdp_stats):
        """Update the parameters of the posterior distribution according
        to the accumulated statistics.

        Parameters
        ----------
        tdp_stats : :class:MixtureStats
            Statistics for the truncated DP.

        """
        self.posterior = self.prior.newPosterior(tdp_stats)
        self.updateParams()
