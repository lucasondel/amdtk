
"""Dirichlet density."""

import numpy as np
from scipy.special import gammaln, psi
from .model import Model
from .model import InvalidModelError
from .model import InvalidModelParameterError
from .model import MissingModelParameterError
from .prior import Prior
from .prior import PriorStats


class DirichletStats(PriorStats):
    """Sufficient statistics for the NormalGamma."""

    def __init__(self, X, weights=None):
        if weights is None:
            weighted_X = X
        else:
            weighted_X = (np.asarray(weights)*X.T).T
        self.__stats = (weighted_X).sum(axis=0)

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError()
        if key < 0 or key > 0:
            raise IndexError
        return self.__stats

    def __iadd__(self, other):
        self.__stats += other.__stats
        return self


class Dirichlet(Model, Prior):
    """Dirichlet density.

    Attributes
    ----------
    alphas : numpy.ndarray
        Parameters of the Dirichlet density.

    """

    @classmethod
    def loadParams(cls, config, data):
        """Load the parameters of the model.

        Parameters
        ----------
        config : dict like
            Dictionary like object containing specific values of the
            model.
        data : dict
            Extra data that may be used for initializing the model.

        Returns
        -------
        params : dict
            Dictioanry of the model's parameters.

        """
        params = {}
        dim = int(config['dim'])
        params['alphas'] = np.ones(dim) * float(config['alpha'])
        return params

    def __init__(self, params):
        """Initialize a Dirichlet density.

        Parameters
        ----------
        params : dict
            Dictionary containing:
              * alphas: parameters of the Dirichlet density

        """
        super().__init__(params)
        missing_param = None
        try:
            if (self.alphas <= 0).any():
                raise InvalidModelParameterError(self, 'alphas', self.alphas)
        except KeyError as e:
            missing_param = e.__str__()

        if missing_param is not None:
            raise MissingModelParameterError(self, missing_param)

    @property
    def alphas(self):
        return self.params['alphas']

    def stats(self, stats, X, data, weights):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        stats : dict
            Dictionary where to store the stats for each model.
        X : numpy.ndarray
           Ignored.
        data : dict
            Data specific to each sub-model with to use to accumulate
            the stats.
        weights : numpy.ndarray
            Weights to apply when building the stats.

        """
        try:
            stats[self.uuid] += DirichletStats(data, weights)
        except KeyError:
            stats[self.uuid] = DirichletStats(data, weights)

    def expectedX(self):
        """Expected value of the weights.

        Returns
        -------
        E_weights : numpy.ndarray
            Expected value of weights.
        """
        return self.alphas / self.alphas.sum()

    def expectedLogX(self):
        """Expected value of the logarithm of the weights.

        Returns
        -------
        E_log weights : numpy.ndarray
            Expected value of the log of the weights.

        """
        return psi(self.alphas) - psi(self.alphas.sum())

    def KL(self, q):
        """KL divergence between the current and the given density.

        Parameters
        ----------
        q : :class:`Dirichlet`
            Dirichlet density with which to compute the KL divergence.
        Returns
        -------
        KL : float
            KL divergence.

        """
        if not isinstance(q, Dirichlet):
            raise InvalidModelError(q, self)

        p = self

        E_log_weights = p.expectedLogX()
        val = gammaln(p.alphas.sum())
        val -= gammaln(q.alphas.sum())
        val -= gammaln(p.alphas).sum()
        val += gammaln(q.alphas).sum()
        val += (E_log_weights * (p.alphas - q.alphas)).sum()
        return val

    def newPosterior(self, stats):
        """Create a new Dirichlet density given the parameters of the
        current model and the statistics provided.

        Parameters
        ----------
        stats : :class:DirichletStats
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:`Dirichlet`
            New Dirichlet density.

        """
        new_params = {
            'alphas': self.alphas + stats[0]
        }
        return Dirichlet(new_params)
