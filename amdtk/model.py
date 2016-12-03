
"""Base class for the models."""


class Model(object):
    """This base class is mainly used to assign 
    a unique ID (per session) to each model.
    
    """
    
    n_model = 0

    def __init__(self):
        self.id = Model.n_model + 1
        Model.n_model += 1
        