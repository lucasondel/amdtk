
"""Generic implementation of a discrete latent model."""

class DiscreteLatentModelError(Exception):
    "Base class for DiscreteLatentModel exceptions."""
    pass


class DiscreteLatentModelEmptyListError(DiscreteLatentModelError):
    """Raised when attempting to create a DiscreteLatentModel."""
    
    def __init__(self, obj, message):
        self.obj = obj
        self.message = message
    

class DiscreteLatentModel(object):
    """Base class for model having discrete latent variable.

    Attributes
    ----------
    k : int
        Number of state for the hidden variable.
    components : list like
        Model associated for each specific state of the hidden variable.

    """

    def __init__(self, components):
        if len(components) == 0:
            raise DiscreteLatentModelEmptyListError(self, "empty list")
        self.components = components

    @property
    def k(self):
        """Number of state for the hidden variable."""
        return len(self.components)
