
"""Acoustic modeling for acoustic unit discovery."""

import abc
import numpy as np
from ..models import MixtureStats
from ..models import GaussianDiagCovStats


class AcousticModel(metaclass=abc.ABCMeta):

    def __init__(self, names, gmms):
        """Create GMMs acoustic model.

        Parameters
        ----------
        names : list
            List of name associated with each model.
        gmms : list
            List of GMMs. The total number of element in the list should
            be nunits x nstates.

        """
        self.name_model = {}
        for i, name in enumerate(names):
            self.name_model[name] = gmms[i]
        self.n_models = len(gmms)

    def evaluate(self, X):
        E_log_p_X_given_Z = np.zeros((X.shape[0], self.n_models))
        log_resps = []
        index_name = {}
        for i, name in enumerate(self.name_model):
            index_name[i] = name
            gmm = self.name_model[name]
            llh, log_resp = gmm.expLogLikelihood(X)
            E_log_p_X_given_Z[:, i] = llh
            log_resps.append(log_resp)

        return E_log_p_X_given_Z, log_resps, index_name

    def stats(self, X, hmm_log_resps, am_log_resps, index_name):
        gmm_stats = {}
        gauss_stats = {}
        for i, name in index_name.items():
            gmm = self.name_model[name]
            log_weights = (hmm_log_resps[:, i] + am_log_resps[i].T).T
            weights = np.exp(log_weights)
            gmm_stats[i] = MixtureStats(weights)
            for j in range(gmm.k):
                gauss_stats[(i, j)] = GaussianDiagCovStats(X, weights[:, j])
        return gmm_stats, gauss_stats
