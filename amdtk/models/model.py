
"""Base class for all the models."""

import uuid
import abc


class ModelError(Exception):
    "Base class for Model exceptions."""
    pass


class InvalidModelError(ModelError):
    """Raised when the model is of a different type from the expected
    one.

    """

    def __init__(self, obj, expected_obj):
        self.obj = obj
        self.expected_obj = expected_obj

    def __str__(self):
        return "expected {0} got {1}".format(self.expected_obj,
                                             self.obj)


class InvalidModelParameterError(ModelError):
    """Raised when the model is initialized with an invalid parameter.

    """

    def __init__(self, obj, name, value):
        self.obj = obj
        self.name = name
        self.value = value

    def __str__(self):
        return "invalid value: {0} for parameter {1} of {2} model".format(
            self.value, self.name, self.obj)


class MissingModelParameterError(ModelError):
    """Raised when a parameter is missing when initializing the model.

    """
    def __init__(self, obj, name):
        self.obj = obj
        self.name = name

    def __str__(self):
        return "missing value for parameter {0} of {1} model".format(
            self.name, self.obj)


class DiscreteLatentModelEmptyListError(ModelError):
    """Raised when attempting to create a DiscreteLatentModel."""

    def __init__(self, obj):
        self.obj = obj

    def __str__(self):
        return "Creating a {0} model with not components.".format(self.obj)


class Model(metaclass=abc.ABCMeta):
    """Base class for all the models.

    Attributes
    ----------
    uuid

    """

    @abc.abstractmethod
    def __init__(self, params):
        """Initialize a model.

        Parameters
        ----------
        params : dict
            Dictionary containing the parameters of the object.

        """
        self.params = params
        self.__uuid = uuid.uuid4().int >> 64

    @property
    def uuid(self):
        """Unique identifier of the model."""
        return self.__uuid

    @abc.abstractmethod
    def stats(self, stats, X, data, weights):
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

        """
        pass

    def __str__(self):
        return self.__class__.__name__


class VBModel(metaclass=abc.ABCMeta):
    """Base class for all the models that can be trained with
    Variational Bayes algorithm.

    """

    @abc.abstractproperty
    def prior(self):
        pass

    @abc.abstractproperty
    def posterior(self):
        pass

    @abc.abstractmethod
    def expectedLogLikelihood(self, X, weight=1.0):
        """Expected value of the log-likelihood of X.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.
        weight : float
            Weight to apply to the expected log-likelihood.

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood.
        data : object
            Data needed by the model to evaluate the statistics for the
            VB update.

        """
        pass

    @abc.abstractmethod
    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns
        -------
        KL : float
            KL divergence.

        """
        return self.posterior.KL(self.prior)

    @abc.abstractmethod
    def updatePosterior(self, stats):
        """Update the parameters of the posterior density given the
        accumulated statistics.

        Parameters
        ----------
        stats : obj
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:`Prior`
            New posterior density/distribution.

        """
        self.posterior = self.prior.newPosterior(stats)


class DiscreteLatentModel(metaclass=abc.ABCMeta):
    """Base class for model having discrete latent variable.

    Attributes
    ----------
    k : int
        Number of state for the hidden variable.
    components : list like
        Model associated for each specific state of the hidden variable.

    """

    @abc.abstractproperty
    def components(self):
        pass

    @property
    def k(self):
        """Number of state for the hidden variable."""
        return len(self.components)
