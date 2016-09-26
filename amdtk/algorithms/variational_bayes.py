
"""Variational Bayes algorithm."""

import abc
from ..models import VBModel
from ..models import InvalidModelError

class UnknownVariationalBayesAlgorithmError(Exception):
    """Raised when the algorithm name in a configuration does not match
    any known algorithm.

    """

    def __init__(self, alg_name):
        self.alg_name = alg_name

    def __str__(self):
        return self.alg_name

class VariationalBayes(metaclass=abc.ABCMeta):

    @classmethod
    def create(cls, config):
        alg_name = config['type']
        amdtk_module = __import__('amdtk')
        algorithms = getattr(amdtk_module, 'algorithms')
        try:
            alg_class = getattr(algorithms, alg_name)
            failed = False
        except AttributeError:
            failed = True

        if failed:
            raise UnknownVariationalBayesAlgorithmError(alg_name)
        
        return alg_class(config)

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

    def __init__(self, config):
        pass

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

class StochasticGradientVariationalBayes(VariationalBayes):

    def __init__(self, config):
        self.nframes = config['nframes']
        self.tau = config['tau']
        self.kappa = config['kappa']
        try:
            self.step = config['step']
        except KeyError:
            self.step = 0

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
        stats['grad_nframes'] = X.shape[0]
        return E_llh/X.shape[0], stats

    def maximization(self, model, stats):
        """Maximization step of the variational bayes training.

        Parameters
        ----------
        step : int
            Step number of the gradient descent.
        model : :class:`VBModel`
            Model to update.
        stats : dictionary
            Accumulated statistics for the model.

        """
        if not isinstance(model, VBModel):
            raise InvalidModelError(model, VBModel)
        #print('total # frames:', self.nframes, 'grad # frames:', 
        #    stats['grad_nframes'])
        lrate = (self.tau + self.step)**(-self.kappa)
        model.gradUpdatePosterior(stats, lrate, self.nframes, 
            stats['grad_nframes'])

