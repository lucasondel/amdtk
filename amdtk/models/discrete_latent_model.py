
"""Generic implementation of a discrete latent model."""


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
        self.components = components

    @property
    def k(self):
        """Number of state for the hidden variable."""
        return len(self.components)
