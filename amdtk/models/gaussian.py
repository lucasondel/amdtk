
"""Gaussian distribution with a GaussianWishart prior."""

import logging
import numexpr as ne
from numba import jit
import numpy as np
from scipy.special import psi, gammaln
from .model import Model

def sample_wishart(W, nu):
    """Returns a sample from the Wishart distn, conjugate prior for
    precision matrices.

    Parameters
    ----------
    W : numpy.ndarray
        Matrix parameter of the Wishart.
    nu : float
        Number of degree of freedom of the Wishart.

    """
    n = W.shape[0]
    chol = np.linalg.cholesky(W)

    # use matlab's heuristic for choosing between the two different 
    # sampling schemes
    if (nu <= 81+n) and (nu == round(nu)):
        X = np.dot(chol, np.random.normal(size=(n, int(nu))))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(nu - np.arange(0, n), size=n)))
        A[np.tri(n, k=-1, dtype=bool)] = np.random.normal(size=int(n*(n-1)/2.))
        X = np.dot(chol, A)

    return np.dot(X, X.T)

def logB(W, nu):
    """Log of the partition function of the Wishart density
    
    Parameters
    ----------
    W : numpy.ndarray
        Matrix parameter of the Wishart.
    nu : float
        Number of degree of freedom of the Wishart.
        
    Returns
    -------
    retval : float
        Log of the normalization constant of the Wishart.
        
    """
    D = W.shape[0]
    idx = np.arange(1, D + 1, 1)
    s, ldet = np.linalg.slogdet(W)
    retval = -.5 * nu * s * ldet
    tmp = .5 * D * nu * np.log(2) + .25 * D * (D - 1) * np.log(np.pi)
    tmp += gammaln(.5*(nu + 1 - idx)).sum()
    return retval - tmp


class Gaussian(Model):
    """Multivariate Gaussian density with a Gaussian-WIshart Prior."""

    def __init__(self, hmean, hkappa, hnu, hW):
        """Initializat the Gaussian.
        
        Parameters
        ----------
        hmean : numpy.ndarray
            Prior mean.
        hkappa : float
            Prior count for the mean. 
        hnu : float
            Prior counts for the precision matrix.
        hW : numpy.ndarray
            Prior precision matrix.
        
        """
        # Initialize the base class "Model".
        super().__init__()
        
        # Hyper-parameters
        self.hmean = hmean
        self.hkappa = hkappa
        self.hnu = hnu
        self.hW = hW
        
        # Initialize the posterior's parameters to their 
        # corresponding priors.
        self.pmean = hmean
        self.pkappa = hkappa
        self.pnu = hnu
        self.pW = hW

        # Sample parameters from the posteriors.
        self.sample_params()
        
    def sample_params(self):
        """Sample new parameters from the current posterior 
        distribution.

        """
        self.precision = sample_wishart(self.pW, self.pnu)
        cov = np.linalg.inv(self.pkappa * self.precision)
        self.mean = np.random.multivariate_normal(self.pmean, cov)
        
    def get_stats(self, X, resps):
        """Compute the sufficient statistics for the model.
        
        Parameters
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        resps : numpy.ndarray
            Weights for each frame.
        
        Returns
        -------
        stats : dict
            Dictionary where 's0', 's1' and 's2' are the keys for the
            zeroth, first and second order statistics respectively.
        
        """
        s0 = float(resps.sum())
        s1 = (resps*X.T).T.sum(axis=0)
        s2 = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            x = X[i,:,np.newaxis]
            s2 += resps[i] * x.dot(x.T)
        
        stats_data = {
            's0': s0,
            's1': s1,
            's2': s2
        }
        return stats_data
    
    def log_likelihood(self, X):
        """Log-likelihood of the data given the current parameters.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        
        Returns
        -------
        llh : float
           Log-likelihood.
            
        """ 
        # Expected value of the log of the precision matrix.
        dim = X.shape[1]
        s, ldet = np.linalg.slogdet(self.precision)
        ldet *= s 
        
        llh = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            # Log normalizer.
            llh[n] = .5 * ldet - .5 * dim * np.log(2 * np.pi) 
            
            # Log of the Gaussian kernel.
            x_m = (X[n] - self.mean)
            llh[n] += -.5 * (x_m).T.dot(self.precision.dot(x_m))

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
        # Expected value of the log of the precision matrix.
        dim = X.shape[1]
        idx = np.arange(1, dim + 1, 1)
        log_prec = psi(.5*(self.pnu + 1 - idx)).sum() + dim * np.log(2) 
        det = np.linalg.det(self.pW)
        log_prec += np.log(det)
        
        llh = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            llh[n] = .5 * log_prec - .5 * dim * np.log(2 * np.pi) -.5 * dim / self.pkappa
            
            # Expected value of the log of the Gaussian kernel.
            x_m = (X[n] - self.pmean)
            llh[n] += -.5 * self.pnu*(x_m).T.dot(self.pW.dot(x_m))

        return llh.sum(axis=1)
    
    def log_predictive(self, X):
        """Log of the predictive density given the current state 
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
        dim = X.shape[1]
        
        # Compute the parameters of the T distribution.
        mean = self.pmean
        L = self.pW * ((self.pnu + 1 - dim) * self.pkappa / (1 + self.pkappa))
        nu = self.pnu + 1 - dim

        # Pre-compute the log of the determinant of the precision matrix.
        s, ldet = np.linalg.slogdet(L)
        ldet = s * ldet
         
        # Log of the partition function Z
        log_norm = gammaln(.5 * (dim + nu)) - gammaln(.5 * nu)
        log_norm += .5 * ldet
        log_norm -= .5 * dim * (np.log(nu * np.pi))
        
        # Kernel 
        log_llh = np.zeros(X.shape[0])
        for n in range(X.shape[0]):    
            x_mean = X[n] - mean
            kernel = 1 + (1/nu) * x_mean.T.dot(L).dot(x_mean)
            log_kernel = -.5 * (nu + dim) * np.log(kernel)
            log_llh[n] = log_norm + log_kernel
        
        return log_llh
   
    def log_marginal_evidence(self):
        N = self.pkappa - self.hkappa
        D = self.pmean.shape[0]
        lme = - .5 * D * N * np.log(2*np.pi)
        lme += .5 * D * (np.log(self.hnu) - np.log(self.pnu))
        lme += logB(self.hW, self.hnu) - logB(self.pW, self.pnu)
        return lme
    
    def update(self, stats):
        """ Update the posterior parameters given the sufficient
        statistics.
        
        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.
        
        """
        N = stats[self.id]['s0']
        if N > .1:
            mean = stats[self.id]['s1'] / N
            tmp_mean = mean[:, np.newaxis]
            cov = (stats[self.id]['s2'] / N) - tmp_mean.dot(tmp_mean.T)
        else:
            mean = np.zeros_like(stats[self.id]['s1'])
            cov = np.zeros_like(stats[self.id]['s2'])

        # compute the posterior's parameters.
        self.pkappa = self.hkappa + N
        self.pmean = (self.hkappa * self.hmean + N*mean) / self.pkappa
        mean_m = ((mean - self.hmean)[:, np.newaxis])
        self.pnu = self.hnu + N + 1
        W_inv = np.linalg.inv(self.hW) + N*cov 
        W_inv += ((self.hkappa * N) / self.pkappa) * (mean_m.dot(mean_m.T))
        self.pW = np.linalg.inv(W_inv)
        
        # Sample parameters from the posteriors.
        self.sample_params()
        
    def KL(self):
        """Kullback-Leibler divergence between the posterior and 
        the prior density.
        
        Returns 
        -------
        ret : float
            KL(q(params) || p(params)).
        
        """
        # Expected value of the log of the precision matrix.
        dim = self.pW.shape[1]
        idx = np.arange(1, dim + 1, 1)
        log_prec = psi(.5*(self.pnu + 1 - idx)).sum() + dim * np.log(2) 
        s, ldet = np.linalg.slogdet(self.pW)
        log_prec += s * ldet
        
        mk_m0 = self.pmean - self.hmean
        
        KL = 0
        KL -= .5 * dim * (np.log(self.hkappa) - np.log(2*np.pi) - \
             self.hkappa / self.pkappa)  
        KL -= .5 * log_prec
        KL += .5 * self.hkappa * self.pnu * mk_m0.T.dot(self.pW).dot(mk_m0)
        KL -= logB(self.hW, self.hnu)
        KL -= .5 * (self.hnu - dim - 1) * log_prec
        KL += .5 * self.pnu * np.trace(np.linalg.inv(self.hW).dot(self.pW))
        
        KL += .5 * log_prec
        KL += .5 * dim * (np.log(self.pkappa) -np.log(2*np.pi) - 1)
        
        # Negative entropy of the Wishart posterior.
        KL += logB(self.pW, self.pnu)
        KL += .5 * (self.pnu - dim - 1) * log_prec 
        KL -= .5 * dim * self.pnu
        
        return KL
 
                
class GaussianDiagCov(Model):
    """Bayesian multivariate Gaussian with a diagonal covariance matrix.
    The prior over the mean and variance for each dimension is a
    Normal-Gamma density.
    
    """
    
    def __init__(self, hmean, hkappa, ha, hb):
        """Initializat the Gaussian.
        
        Parameters
        ----------
        hmean : numpy.ndarray
            Prior mean.
        hkappa : float
            Prior count for the mean. 
        hnu : float
            Prior counts for the precision matrix.
        hW : numpy.ndarray
            Prior precision matrix.
        
        """
        # Initialize the base class "Model".
        super().__init__()
        
        # Hyper-parameters
        self.hmean = hmean
        self.hkappa = hkappa
        self.ha = ha
        self.hb = hb
        
        # Initialize the posterior's parameters to their 
        # corresponding priors.
        self.pmean = hmean
        self.pkappa = hkappa
        self.pa = ha
        self.pb = hb

        # Sample parameters from the posteriors.
        self.sample_params()
        
        # For optimization.
        self.log_predictive_precompiled = False
        
    def sample_params(self):
        """Sample new parameters from the current posterior 
        distribution.

        """
        self.diag_prec = np.zeros_like(self.hmean)
        self.mean = np.zeros_like(self.hmean)
        for d in range(len(self.diag_prec)):
            self.diag_prec[d] = np.random.gamma(self.pa, 1/self.pb[d])
            self.mean[d] = np.random.normal(self.pmean, np.sqrt(self.pb[d]))[0]
        
    def get_stats(self, X, resps):
        """Compute the sufficient statistics for the model.
        
        Parameters
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        resps : numpy.ndarray
            Weights for each frame.
        
        Returns
        -------
        stats : dict
            Dictionary where 's0', 's1' and 's2' are the keys for the
            zeroth, first and second order statistics respectively.
        
        """
        s0 = float(resps.sum())
        w_X = (resps * X.T).T
        s1 = w_X.sum(axis=0)
        s2 = (w_X*X).sum(axis=0)
        
        stats_data = {
            's0': s0,
            's1': s1,
            's2': s2
        }
        return stats_data
    
    def log_likelihood(self, X):
        """Log-likelihood of the data given the current parameters.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        
        Returns
        -------
        llh : float
           Log-likelihood.
            
        """ 
        log_norm = (-np.log(2 * np.pi) + np.log(self.diag_prec))
        return .5 * (log_norm - self.diag_prec * (X - self.mean)**2).sum(axis=1)
    
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
                         
        E_log_prec = psi(self.pa) - np.log(self.pb)
        E_prec = self.pa/self.pb
        log_norm = (E_log_prec - 1/self.pkappa)
        return .5*(log_norm - E_prec*(X - self.pmean)**2).sum(axis=1)
    
    def log_predictive(self, X):
        """Log of the predictive density given the current state 
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
        pa = self.pa
        pb = self.pb
        pmean = self.pmean
        pkappa = self.pkappa
        pi = np.pi
        
        na = pa + .5
        #nb = np.zeros_like(X, dtype=np.float64)
        #ne.evaluate('pb + .5 * ((X - pmean)**2) + (pmean**2) / (1 + pkappa) ', out=nb)
        #nb = self.pb + (.5*((X - self.pmean)**2) + (self.pmean**2) / (1 + self.pkappa))
        
        retval = np.zeros(len(X))
        gln = gammaln(na) - gammaln(pa)
        ne.evaluate('sum(-.5 * log(2*pi) +  pa * log(pb) - na * log(pb + .5 * ((X - pmean)**2) + (pmean**2) / (1 + pkappa) ) + gln, axis=1)', 
                    out=retval)
        #retval = -.5 * np.log(2*np.pi)
        #retval += self.pa * np.log(self.pb) - na * np.log(nb) 
        #retval += gammaln(na) - gammaln(self.pa)
        
        return retval
        #return retval.sum(axis=1)
   
    def log_marginal_evidence(self):
        retval = -.5 * np.log(2*np.pi)
        retval += self.ha * np.log(self.hb) - self.pa * np.log(self.pb) 
        retval += gammaln(self.pa) - gammaln(self.ha)
        
        return retval.sum()
    
    def update(self, stats):
        """ Update the posterior parameters given the sufficient
        statistics.
        
        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.
        
        """
        s0 = stats[self.id]['s0']
        s1 = stats[self.id]['s1']
        s2 = stats[self.id]['s2']
        
        if s0 < .1:
            self.pkappa = self.hkappa
            self.pmean = self.hmean
            self.pa = self.ha
            self.pb = self.hb
        else:
            self.pkappa = self.hkappa + s0
            self.pmean = (self.hkappa * self.hmean + s1) / self.pkappa
            self.pa = self.ha+ .5 * s0
            v = (self.hkappa * self.hmean + s1)**2
            v /= (self.pkappa)
            self.pb = self.hb + 0.5*(-v + s2 + self.hkappa * self.hmean**2)

    def KL(self):
        """Kullback-Leibler divergence between the posterior and 
        the prior density.
        
        Returns 
        -------
        ret : float
            KL(q(params) || p(params)).
        
        """
        p = self

        E_log_prec = psi(self.pa) - np.log(self.pb)
        E_prec = self.pa/self.pb

        return (.5 * (np.log(self.pkappa) - np.log(self.hkappa))
                - .5 * (1 - self.hkappa * (1./self.pkappa + E_prec * (self.pmean - self.hmean)**2))
                - (gammaln(self.pa) - gammaln(self.ha))
                + (self.pa * np.log(self.pb) - self.ha * np.log(self.hb))
                + E_log_prec * (self.pa - self.ha)
                - E_prec * (self.pb - self.hb)).sum()      
