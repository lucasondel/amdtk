
"""Base class for prior model."""

import abc


class Prior(metaclass=abc.ABCMeta):
    """Abstract base class for prior model."""
    
    @abc.abstractmethod
    def expectedX(self):
        """Expected value of the the random variable of the prior.

        Returns
        -------
        E_X : object
            E[X]
            
        """
        pass
        
    @abc.abstractmethod
    def expectedLogX(self):
        """Expected value of the log of the random variable of the 
        prior.
        
        Returns
        -------
        E_X : object
            E[log X]
        
        """
        pass
    
    @abc.abstractmethod
    def KL(self, other):
        """Kullback-Leibler divergence 
        
        Parameters
        ----------
        other : :class:`Prior`
            Other density/distribution with which to compute the KL 
            divergence.
            
        Returns
        -------
        kl : float
            KL(self, other)
        """
        pass
    
    @abc.abstractmethod
    def newPosterior(self, stats):
        """Create a new posterior distribution from the prior.

        Parameters
        ----------
        stats : stats object
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:`Model`
            New posterior distribution.

        """
        pass
    