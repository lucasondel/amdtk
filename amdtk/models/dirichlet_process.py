
"""Dirichlet Process (DP) prior."""

import numpy as np
from scipy.special import psi
from .dirichlet import Dirichlet


class DirichletProcessStats(object):
    """Sufficient statistics for :class:`TruncatedDirichletProcess`.

    Methods
    -------
    __getitem__(key)
        Index operator.
    __add__(stats)
        Addition operator.
    __iadd__(stats)
        In-place addition operator.

    """

    def __init__(self, E_P_Z):
        stats1 = E_P_Z.sum(axis=0)
        stats2 = np.zeros_like(stats1)
        for i in range(len(stats1)-1):
            stats2[i] += stats1[i+1:].sum()
        self.__stats = [stats1, stats2]

    def __getitem__(self, key):
        if type(key) is not int:
            raise KeyError()
        if key < 0 or key > 2:
            raise IndexError()
        return self.__stats[key]

    def __add__(self, other):
        new_stats = DirichletProcessStats(len(self.__stats[0]))
        new_stats += self
        new_stats += other
        return new_stats

    def __iadd__(self, other):
        self.__stats[0] += other.__stats[0]
        self.__stats[1] += other.__stats[1]
        return self


class TruncatedDirichletProcess(object):
    """Truncated Dirichlet process.

    In this model, the maximum number of component is limited in order
    to apply variational bayesian inference.

    Attributes
    ----------
    g1 : float
        First shape parameter of the Beta distribution for the
        stick-breaking construction of the DP.
    g2 : float
        First shape parameter of the Beta distribution for the
        stick-breaking construction of the DP.

    Methods
    -------
    expLogPi()
        Expected value of the log of the weights of the DP.
    KL(pdf)
        KL divergence between the current and the given densities.
    newPosterior(stats)
        Create a new posterior distribution.

    """

    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

    def expLogPi(self):
        """Expected value of the log of the weights of the DP.

        Returns
        -------
        E_log_pi : float
            Log weights.

        """
        n = self.g1.shape[0]
        v = psi(self.g1) - psi(self.g1+self.g2)
        nv = psi(self.g2) - psi(self.g1+self.g2)
        for i in range(1, n):
            v[i] += nv[:i].sum()
        return v

    def KL(self, pdf):
        """KL divergence between the current and the given densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        KL = 0
        for i in range(len(self.g1)):
            a1 = np.array([self.g1[i], self.g2[i]])
            a2 = np.array([pdf.g1[i], pdf.g2[i]])
            KL += Dirichlet(a1).KL(Dirichlet(a2))
        return KL

    def newPosterior(self, stats):
        """Create a new posterior distribution.

        Create a new posterior (a Dirichlet density) given the
        parameters of the current model and the statistics provided.

        Parameters
        ----------
        stats : :class:MultivariateGaussianDiagCovStats
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:Dirichlet
            New Dirichlet density.
        """
        return TruncatedDirichletProcess(self.g1+stats[0], self.g2+stats[1])
