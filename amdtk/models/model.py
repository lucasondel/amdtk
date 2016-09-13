
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
        return "invalid value: {0} for parameter {1} for {2}".format(
            self.value, self.name, self.obj)


class MissingModelParameterError(ModelError):
    """Raised when a parameter is missing when initializing the model.

    """
    def __init__(self, obj, name):
        self.obj = obj
        self.name = name

    def __str__(self):
        return "missing value for parameter {0} for {1}".format(
            self.name, self.obj)


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
