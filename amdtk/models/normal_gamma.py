
""" Normal-Gamma density."""

import numpy as np
from scipy.special import gammaln, psi
from .model import Model
from .model import InvalidModelError
from .model import InvalidModelParameterError
from .model import MissingModelParameterError
from .prior import Prior
from .prior import PriorStats


class NormalGammaStats(PriorStats):
    """Sufficient statistics for the NormalGamma."""

    def __init__(self, X, weights=None):
        if weights is None:
            weighted_X = X
            counts = X.shape[0]
        else:
            weighted_X = (np.asarray(weights)*X.T).T
            if not isinstance(weights, np.ndarray):
                counts = X.shape[0]
            else:
                counts = weights.sum()
        self.__stats = [
            counts,
            weighted_X.sum(axis=0),
            (weighted_X*X).sum(axis=0)
        ]

    def __getitem__(self, key):
        if type(key) is not int:
            raise TypeError()
        if key < 0 or key > 2:
            raise IndexError()
        return self.__stats[key]

    def __iadd__(self, other):
        self.__stats[0] += other.__stats[0]
        self.__stats[1] += other.__stats[1]
        self.__stats[2] += other.__stats[2]
        return self


class NormalGamma(Model, Prior):
    """Normal-Gamma density.

    Attributes
    ----------
    mu : numpy.ndarray
        Mean of the Gaussian density.
    kappa : float
        Factor of the precision matrix.
    alpha : float
        Shape parameter of the Gamma density.
    beta : numpy.ndarray
        Rate parameters of the Gamma density.

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
        params['mu'] = data['mean']
        params['kappa'] = config.getfloat('kappa')
        params['alpha'] = config.getfloat('alpha')
        params['beta'] = config.getfloat('beta') * data['var']
        return params

    def __init__(self, params):
        """Initialize a NormalGamma Prior.

        Parameters
        ----------
        params : dict
            Dictionary containing:
              * mu: mean of the Normal density
              * kappa: coefficient of the precision of the Normal
              density
              * alpha: shape parameter of the Gamma density.
              * beta: Rate parametesr of the Gamma density.

        """
        super().__init__(params)
        missing_param = None
        try:
            # Empty statement just to make sure that the parameter was
            # specified in params.
            self.mu

            if self.kappa <= 0:
                raise InvalidModelParameterError(self, 'kappa', self.kappa)
            if self.alpha <= 0:
                raise InvalidModelParameterError(self, 'alpha', self.alpha)
            if (self.beta <= 0).any():
                raise InvalidModelParameterError(self, 'beta', self.beta)
        except KeyError as e:
            missing_param = e.__str__()

        if missing_param is not None:
            raise MissingModelParameterError(self, missing_param)

    @property
    def mu(self):
        return self.params['mu']

    @property
    def kappa(self):
        return self.params['kappa']

    @property
    def alpha(self):
        return self.params['alpha']

    @property
    def beta(self):
        return self.params['beta']

    def stats(self, stats, X, data, weights):
        """Compute the sufficient statistics for the training..

        Parameters
        ----------
        stats : dict
            Dictionary where to store the stats for each model.
        X : numpy.ndarray
            Data on which to accumulate the stats.
        data : dict
            Ignored.
        weights : numpy.ndarray
            Weights to apply per frame.

        Returns
        -------
        stats : dict
            Dictionary containing the mapping model_id -> stats.

        """
        try:
            stats[self.uuid] += NormalGammaStats(X, weights)
        except KeyError:
            stats[self.uuid] = NormalGammaStats(X, weights)

    def expectedX(self):
        """Expected value of the mean and precision.

        Returns
        -------
        E_mean : numpy.ndarray
            Expected value of the mean.
        E_prec : numpy.ndarray
            Expected value of the precision.
        """
        return self.mu, self.alpha / self.beta

    def expectedLogX(self):
        '''Expected value of the logarithm of the precision.

        NOTE: because the mean can be null or negative the expected
        value of the mean is not always defined. Thus, we do not compute
        this expectation for the mean.

        Returns
        -------
        E_log mean : None
            None
        E_log prec : numpy.ndarray
            Expected value of the log precision.
        '''
        return None, psi(self.alpha) - np.log(self.beta)

    def KL(self, q):
        """KL divergence between the current and the given density.

        Parameters
        ----------
        q : :class:`NormalGamma`
            NormalGamma density with which to compute the KL divergence.
        Returns
        -------
        KL : float
            KL divergence.

        """
        if not isinstance(q, NormalGamma):
            raise InvalidModelError(q, self)
        p = self

        E_mean, E_prec = p.expectedX()
        _, E_log_prec = p.expectedLogX()

        val = .5 * (np.log(p.kappa) - np.log(q.kappa))
        val += - .5*(1 - q.kappa * (1. / p.kappa + E_prec * (p.mu - q.mu)**2))
        val += - gammaln(p.alpha) - gammaln(q.alpha)
        val += p.alpha * np.log(p.beta) - q.alpha * np.log(q.beta)
        val += E_log_prec * (p.alpha - q.alpha)
        val += - E_prec * (p.beta - q.beta)
        val = val.sum()

        return val

    def newPosterior(self, stats):
        """Create a new Normal-Gamma density given the parameters of the
        current model and the statistics provided.

        Parameters
        ----------
        stats : :class:NormalGammaStats
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:NormalGamma
            New NormalGamma density.

        """
        s0 = stats[0]
        s1 = stats[1]
        s2 = stats[2]
        v = (self.kappa * self.mu + s1)**2
        v /= (self.kappa + s0)
        new_params = {
            'mu': (self.kappa * self.mu + s1) / (self.kappa + s0),
            'kappa': self.kappa + s0,
            'alpha': self.alpha + .5 * s0,
            'beta': self.beta + 0.5*(-v + s2 + self.kappa * self.mu**2)
        }
        return NormalGamma(new_params)
