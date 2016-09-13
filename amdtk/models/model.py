
"""Base class for all the models."""

import uuid


class Model(object):
    """Base class for all the models.
    
    Attributes
    ----------
    uuid
    
    """
    
    def __init__(self):
        self.__uuid = uuid.uuid4().int >> 64
    
    @property
    def uuid(self):
        """Unique identifier of the model."""
        return self.__uuid
    