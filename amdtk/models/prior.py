
"""Base class for prior model."""

import abc
from .model import Model


class PriorStats(metaclass=abc.ABCMeta):
    """Abstract base class for the prior statistics."""

    @abc.abstractmethod
    def __init__(self, X, weights=None):
        pass

    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @abc.abstractmethod
    def __iadd__(self, other):
        pass


class Prior(metaclass=abc.ABCMeta):
    """Abstract base class for prior model."""

    @abc.abstractmethod
    def expectedX(self):
        """Expected value of the the random variable of the prior.

        Returns
        -------
        E_X : object
            E[X]

        """
        pass

    @abc.abstractmethod
    def expectedLogX(self):
        """Expected value of the log of the random variable of the
        prior.

        Returns
        -------
        E_X : object
            E[log X]

        """
        pass

    @abc.abstractmethod
    def KL(self, other):
        """Kullback-Leibler divergence

        Parameters
        ----------
        other : :class:`Prior`
            Other density/distribution with which to compute the KL
            divergence.

        Returns
        -------
        kl : float
            KL(self, other)
        """
        pass

    @abc.abstractmethod
    def newPosterior(self, stats):
        """Create a new posterior distribution from the prior.

        Parameters
        ----------
        stats : stats object
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:`Model`
            New posterior distribution.

        """
        pass

class PriorStats(metaclass=abc.ABCMeta):
    """Abstract base class for the prior statistics."""

    @abc.abstractmethod
    def __init__(self, X, weights=None):
        pass

    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @abc.abstractmethod
    def __iadd__(self, other):
        pass


class EmptyPrior(Prior, Model):
    """Prior that does nothing. Helpful for some classes to behave as is
    they had prior in VB algorithm."""

    @classmethod
    def loadParams(cls, config):
        return {}

    def __init__(self, params):
        super().__init__(params)

    def stats(self, stats, X, data, weights, model_id=None):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        stats : dict
            Dictionary where to store the stats for each model.
        X : numpy.ndarray
            Data on which to accumulate the stats.
        data : dict
            Data specific to each sub-model with to use to accumulate
            the stats.
        weights : numpy.ndarray
            Weights to apply when building the stats.
        model_id : int
            Use the specified model_id to store the statistics.

        """
        pass

    def expectedX(self):
        """Expected value of the the random variable of the prior.

        Returns
        -------
        E_X : object
            E[X]

        """
        return 0.

    def expectedLogX(self):
        """Expected value of the log of the random variable of the
        prior.

        Returns
        -------
        E_X : object
            E[log X]

        """
        return 0.

    def KL(self, other):
        """Kullback-Leibler divergence

        Parameters
        ----------
        other : :class:`Prior`
            Other density/distribution with which to compute the KL
            divergence.

        Returns
        -------
        kl : float
            KL(self, other)
        """
        return 0.

    def newPosterior(self, stats):
        """Create a new posterior distribution from the prior.

        Parameters
        ----------
        stats : stats object
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:`Model`
            New posterior distribution.

        """
        return self
