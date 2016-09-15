
"""Variational Bayes algorithm."""

import abc
from ..models import VBModel
from ..models import InvalidModelError


class VariationalBayes(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def expectation(self, model, X):
        """Variational Bayes expectation step.

        Parameters
        ----------
        model : :class:`VBModel`
            Model to use for the VB expectation.
        X : numpy.ndarray
            The data. A matrix (NxD) of N frames with D dimensions.

        Returns
        -------
        E_log_p_X, stats : scalar, tuple
            The expected value of the log-evidence.
        stats : dict
            Statistics for each model.

        """
        pass

    @abc.abstractmethod
    def maximization(self, model, X):
        """Maximization step of the variational bayes training.

        Parameters
        ----------
        model : :class:`VBModel`
            Model to update.
        stats : dictionary
            Accumulated statistics for the model.

        """
        pass


class StandardVariationalBayes(VariationalBayes):

    def expectation(self, model, X, weight):
        """Variational Bayes expectation step.

        Parameters
        ----------
        model : :class:`VBModel`
            Model to use for the VB expectation.
        X : numpy.ndarray
            The data. A matrix (NxD) of N frames with D dimensions.
        weight : float
            Scaling factor that can be used by the underlying model.

        Returns
        -------
        E_log_p_X, stats : scalar, tuple
            The expected value of the log-evidence.
        stats : dict
            Statistics for each model.

        """
        if not isinstance(model, VBModel):
            raise InvalidModelError(model, VBModel)

        E_llh, data = model.expectedLogLikelihood(X, weight=weight)
        stats = {}
        model.stats(stats, X, data, weights=1.0)
        return E_llh, stats

    def maximization(self, model, stats):
        """Maximization step of the variational bayes training.

        Parameters
        ----------
        model : :class:`VBModel`
            Model to update.
        stats : dictionary
            Accumulated statistics for the model.

        """
        if not isinstance(model, VBModel):
            raise InvalidModelError(model, VBModel)
        model.updatePosterior(stats)
