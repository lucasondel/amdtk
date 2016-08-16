
"""Acoustic modeling for acoustic unit discovery. Properly
speaking, the objects implemented here are not "model" but rather
structure that embed a set of model (i.e. gmm) that simplify the
interation with any AUD graph."""

import abc
import numpy as np
from ..models import MixtureStats
from ..models import GaussianDiagCovStats


class AcousticModel(metaclass=abc.ABCMeta):
    """"Base class for any acoustic model."""

    @abc.abstractproperty
    def ngmms(self):
        pass

    @abc.abstractmethod
    def evaluate(self, X):
        """Compute the expected value of the log-likelihood of the
        acoustic model of the phone loop.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.

        Returns
        -------
        E_llh : numpy.ndarray
            The (expected value of the) log-likelihood per frame.
        data : object
            Additional data that the model can re-use when accumulating
            the statistics.

        """
        pass

    @abc.abstractmethod
    def stats(self, X, am_log_resps, hmm_log_resps):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.
        am_log_resps : nlist of umpy.ndarray
            Responsibility for each gmm of the model.
        hmm_log_resps : numpy.ndarray
            Responsibility for each state of the hmm.

        Returns
        -------
        E_log_P_X : float
            The expected value of log probability of the sequence of
            features.
        resp_units ; numpy.ndarray
            The responsibility for each unit per frame.

        """
        pass


class GMMAcousticModel(AcousticModel):
    """Acoustic model where each state of the AUD graph has its own
    GMM."""

    def __init__(self, nunits, nstates, gmms):
        """Create GMMs acoustic model.

        Parameters
        ----------
        nunits : int
            Maximum number of units in the AUD model.
        nstates : int
            Number of states per sub-HMM.
        gmms : list
            List of GMMs. The total number of element in the list should
            be nunits x nstates.

        """
        self.nunits = nunits
        self.nstates = nstates
        self.gmms = gmms

    @property
    def ngmms(self):
        return len(self.gmms)

    def evaluate(self, X):
        E_log_p_X_given_Z = np.zeros((X.shape[0], self.ngmms))
        log_resps = []
        for i, gmm in enumerate(self.gmms):
            llh, log_resp = gmm.expLogLikelihood(X)
            E_log_p_X_given_Z[:, i] = llh
            log_resps.append(log_resp)

        return E_log_p_X_given_Z, log_resps

    def stats(self, X, am_log_resps, hmm_log_resps):
        # Evaluate the statistics of the GMMs.
        gmm_stats = {}
        gauss_stats = {}
        for i, gmm in enumerate(self.gmms):
            log_weights = (hmm_log_resps[:, i] + am_log_resps[i].T).T
            weights = np.exp(log_weights)
            gmm_stats[i] = MixtureStats(weights)
            for j in range(gmm.k):
                gauss_stats[(i, j)] = GaussianDiagCovStats(X, weights[:, j])
        return gmm_stats, gauss_stats
