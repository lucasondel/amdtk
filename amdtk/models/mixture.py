
"""Bayeisna Mixture of Gaussian."""

import numpy as np
from scipy.misc import logsumexp
from scipy.special import psi, gammaln
from .model import Model


class Mixture(Model):
    """Bayesian mixture of Gaussian with a Dirichlet prior."""

    def __init__(self, components, alphas):    
        """Initialize the mixture.
        
        Parameters
        ----------
        gaussians : list
            List of :class:`Gaussian` components.
        alphas : numpy.ndarray
            Hyper-parameters of the Dirichlet prior.
        
        """
        # Initialize the base class "Model".
        super().__init__()
        
        self.halphas = alphas
        self.palphas = alphas
        self.components = components

    def get_stats(self, X, resps):
        """Compute the sufficient statistics for the model.
        
        Parameters
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        resps : numpy.ndarray
            Weights (N x K) for each frame and each component.
        
        Returns
        -------
        stats : dict
            Nested dictionaries. Statistics for a specific model
            are accesible by the key (model.id) of the model.
            
        """
        stats_data = {}
        stats_data[self.id] = {}
        stats_data[self.id]['s0'] = resps.sum(axis=0)
        for i, component in enumerate(self.components):
            stats_data[component.id] = {}
            stats_data[component.id] = component.get_stats(X, resps[:, i])
        
        return stats_data
    
    def log_likelihood(self, X):
        """Log likelihood.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        
        Returns
        -------
        llh : float
            Log-likelihood.
            
        """ 
        ncomps = len(self.components)
        llh = np.zeros((X.shape[0], ncomps))
        
        log_weights = np.log(self.weights)
        for i, component in enumerate(self.components):
            llh[:, i] = log_weights[i] + component.log_likelihood(X)
        
        return llh
       
    def expected_log_likelihood(self, X):
        """Expected value of the log likelihood.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        
        Returns
        -------
        E_llh : float
            Expected value of the log-likelihood.
            
        """ 
        ncomps = len(self.components)
        llh = np.zeros((X.shape[0], ncomps))
        
        log_weights = psi(self.palphas) - psi(self.palphas.sum())
        for i, component in enumerate(self.components):
            llh[:, i] = log_weights[i] + component.expected_log_likelihood(X)
        
        return llh
    
    def comp_expected_log_likelihood(self, X):
        """Component-wise (no weights) Expected value of the log 
        likelihood.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        
        Returns
        -------
        E_llh : float
            Expected value of the log-likelihood.
            
        """ 
        ncomps = len(self.components)
        llh = np.zeros((X.shape[0], ncomps))
        for i, component in enumerate(self.components):
            llh[:, i] = component.expected_log_likelihood(X)
        
        return llh
    
    def log_predictive(self, X):
        """Log of the predictive distribution given the current state 
        of the posterior's parameters.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        
        Returns
        -------
        log_pred : float
            log predictive density.
            
        """
        ncomps = len(self.components)
        llh = np.zeros((X.shape[0], ncomps))
        log_weights = np.log(self.palphas) - np.log(self.palphas.sum())
        for i, component in enumerate(self.components):
            llh[:, i] = log_weights[i] + component.log_predictive(X)
        
        norm = logsumexp(llh, axis=1)
        return norm
    
    def comp_log_predictive(self, X):
        """Component-wise (no weights) log of the predictive distribution 
        given the current state of the posterior's parameters.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        
        Returns
        -------
        log_pred : float
            log predictive density.
            
        """
        ncomps = len(self.components)
        llh = np.zeros((X.shape[0], ncomps))
        for i, component in enumerate(self.components):
            llh[:, i] = component.log_predictive(X)
        
        return llh

    def update(self, stats):
        """ Update the posterior parameters given the sufficient
        statistics.
        
        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.
        
        """
        self.palphas = self.halphas + stats[self.id]['s0']
        for component in self.components:
            component.update(stats)
    
    def KL(self):
        """Kullback-Leibler divergence between the posterior and 
        the prior density.
        
        Returns 
        -------
        ret : float
            KL(q(params) || p(params)).
        
        """
        KL = 0.
        KL = gammaln(self.palphas.sum())
        KL -= gammaln(self.halphas.sum())
        KL -= gammaln(self.palphas).sum()
        KL += gammaln(self.halphas).sum()
        log_weights = psi(self.palphas) - psi(self.palphas.sum())
        KL += (self.palphas - self.halphas).dot(log_weights)
        for component in self.components:
            KL += component.KL()
        return KL 
