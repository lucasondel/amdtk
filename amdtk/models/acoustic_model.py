
"""Acoustic modeling for acoustic unit discovery."""

import abc
import numpy as np
from ..models import MixtureStats
from ..models import GaussianDiagCovStats


class AcousticModel(metaclass=abc.ABCMeta):

    def __init__(self, parent_names, state_names, gmms):
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
        self.index_name = {}
        self.name_index = {}
        self.parent_name_indices = {}
        for i, name in enumerate(state_names):
            self.name_model[name] = gmms[i]
            self.index_name[i] = name
            self.name_index[name] = i
            try:
                self.parent_name_indices[parent_names[i]].append(i)
            except KeyError:
                self.parent_name_indices[parent_names[i]] = [i]
        self.n_models = len(gmms)

    def evaluate(self, X):
        E_log_p_X_given_Z = np.zeros((X.shape[0], self.n_models))
        log_resps = []
        for i, name in self.index_name.items():
            gmm = self.name_model[name]
            llh, log_resp = gmm.expLogLikelihood(X)
            E_log_p_X_given_Z[:, i] = llh
            log_resps.append(log_resp)

        return E_log_p_X_given_Z, log_resps

    def stats(self, X, hmm_log_resps, am_log_resps):
        gmm_stats = {}
        gauss_stats = {}
        for i, name in self.index_name.items():
            gmm = self.name_model[name]
            log_weights = (hmm_log_resps[:, i] + am_log_resps[i].T).T
            weights = np.exp(log_weights)
            gmm_stats[name] = MixtureStats(weights)
            gauss_stats[name] = {}
            for j in range(gmm.k):
                gauss_stats[name][j] = GaussianDiagCovStats(X, weights[:, j])
        return gmm_stats, gauss_stats

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns:class:MixtureStats
        -------
        KL : float
            KL divergence.

        """
        KL = 0
        for name, gmm in self.name_model.items():
            KL += gmm.KLPosteriorPrior()
        return KL

    def updatePosterior(self, gmm_stats, gauss_stats):
        for name, stats in gmm_stats.items():
            gmm = self.name_model[name]
            gmm.updatePosterior(stats)

        for name, data in gauss_stats.items():
            gmm = self.name_model[name]
            for j, stats in data.items():
                gmm.components[j].updatePosterior(stats)
