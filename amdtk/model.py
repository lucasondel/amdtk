
"""Base class for all the models."""

import abc


class Model(metaclass=abc.ABCMeta):
    """This base class is mainly used to assign
    a unique ID (per session) to each model.

    """

    # pylint: disable=too-few-public-methods
    # This base class is only here to provied a
    # unique identifier to each model. There is
    # meed for more methods.

    # Total number of model in this session.
    n_model = 0

    def __init__(self):
        self.uid = Model.n_model + 1
        Model.n_model += 1

