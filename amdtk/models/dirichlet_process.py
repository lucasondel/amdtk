
"""Dirichlet Process (DP) prior."""

import numpy as np
from scipy.special import psi
from .dirichlet import Dirichlet
from .model import Model
from .model import InvalidModelError
from .model import InvalidModelParameterError
from .model import MissingModelParameterError
from .prior import Prior
from .prior import PriorStats


class DirichletProcessStats(PriorStats):
    """Sufficient statistics for :class:`DirichletProcess`."""

    def __init__(self, X, weights=None):
        if weights is None:
            weighted_X = X
        else:
            weighted_X = (np.asarray(weights)*X.T).T

        stats1 = weighted_X.sum(axis=0)
        stats2 = np.zeros_like(stats1)
        for i in range(len(stats1)-1):
            stats2[i] += stats1[i+1:].sum()

        self.__stats = [stats1, stats2]

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError()
        if key < 0 or key > 1:
            raise IndexError()
        return self.__stats[key]

    def __iadd__(self, other):
        self.__stats[0] += other.__stats[0]
        self.__stats[1] += other.__stats[1]
        return self


class DirichletProcess(Model, Prior):
    """(Truncated) Dirichlet process.

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

    """

    @classmethod
    def loadParams(cls, config, data):
        """Load the parameters of the model.

        Parameters
        ----------
        config : dict like
            Dictionary like object containing specific values of the
            model.
        data : dict
            Extra data that may be used for initializing the model.

        Returns
        -------
        params : dict
            Dictioanry of the model's parameters.

        """
        params = {}
        params['T'] = config.getint('truncation')
        params['gamma'] = config.getfloat('concentration')
        return params

    def __init__(self, params):
        """Initialize a (truncated) Dirichlet process.

        Attributes
        ----------
        params : dict
            Dictionary containing:
              * T: truncation parameter
              * gamma: concentration parameter

        """
        super().__init__(params)
        missing_param = None
        try:
            if self.T <= 0:
                raise InvalidModelParameterError(self, 'T', self.T)
            if self.gamma <= 0:
                raise InvalidModelParameterError(self, 'gamma', self.gamma)
        except KeyError as e:
            missing_param = e.__str__()

        if missing_param is not None:
            raise MissingModelParameterError(self, missing_param)

        self.g1 = np.ones(self.T)
        self.g2 = np.zeros(self.T) + self.gamma

    @property
    def T(self):
        return self.params['T']

    @property
    def gamma(self):
        return self.params['gamma']

    def stats(self, stats, X, data, weights, model_id=None):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        stats : dict
            Dictionary where to store the stats for each model.
        X : numpy.ndarray
            Ignored.
        data : dict
            Data specific to each sub-model with to use to accumulate
            the stats.
        weights : numpy.ndarray
            Weights to apply when building the stats.
        model_id : int
            Use the specified model_id to store the statistics.

        """
        if model_id is None:
            model_id = self.uuid
        try:
            stats[model_id] += DirichletProcessStats(data, weights)
        except KeyError:
            stats[model_id] = DirichletProcessStats(data, weights)

    def expectedX(self):
        """Expected value of the weights.

        Returns
        -------
        E_weights : numpy.ndarray
            Expected value of weights.
        """
        return self.g1 / (self.g1 + self.g2)

    def expectedLogX(self):
        """Expected value of the logarithm of the weights.

        Returns
        -------
        E_log weights : numpy.ndarray
            Expected value of the log of the weights.

        """
        v = psi(self.g1) - psi(self.g1+self.g2)
        nv = psi(self.g2) - psi(self.g1+self.g2)
        for i in range(1, self.T):
            v[i] += nv[:i].sum()
        return v

    def KL(self, q):
        """KL divergence between the current and the given density.

        Parameters
        ----------
        q : :class:`DirichletProcess`
            Dirichlet processwith which to compute the KL divergence.
        Returns
        -------
        KL : float
            KL divergence.

        """
        if not isinstance(q, DirichletProcess):
            raise InvalidModelError(q, self)

        p = self
        KL = 0
        for i in range(p.T):
            a1 = np.array([p.g1[i], p.g2[i]])
            a2 = np.array([q.g1[i], q.g2[i]])
            d1 = Dirichlet({'alphas': a1})
            d2 = Dirichlet({'alphas': a2})
            KL += d1.KL(d2)
        return KL

    def newPosterior(self, stats):
        """Create a new Dirichlet process giventhe parameters of the
        current model and the statistics provided.

        Parameters
        ----------
        stats : :class:DirichletProcessStats
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:`DirichletProcess`
            New Dirichlet process.

        """
        new_params = {
            'T': self.T,
            'gamma': self.gamma
        }
        dp = DirichletProcess(new_params)
        dp.g1 = self.g1 + stats[0]
        dp.g2 = self.g2 + stats[1]
        return dp
