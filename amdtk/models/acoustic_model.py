
"""Acoustic modeling for acoustic unit discovery. Properly
speaking, the objects implemented here are not "model" but rather
structure that embed a set of model (i.e. gmm) that simplify the
interation with any AUD graph."""

import abc
import numpy as np


class AcousticModel(metaclass=abc.ABCMeta):
    """"Base class for any acoustic model."""

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
            The expected value of the log-likelihood per frame.
        data : obect
            Additional information that the model can re-use while
            accumulating the statistics.

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
