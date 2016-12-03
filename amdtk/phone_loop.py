
"""Phone-Loop model where each unit is modeled by a left-to-right HMM."""

import logging
from bisect import bisect
import numpy as np
from scipy.misc import logsumexp
from scipy.special import psi, gammaln
from .model import Model

def _log_transition_matrix(n_units, n_states, mixture_weights, ins_penalty=1., 
                          prob_final_state=.5):
    tot_n_states = n_units * n_states
    init_states = np.arange(0, tot_n_states, n_states)
    final_states = init_states + n_states - 1
    A = np.zeros((tot_n_states, tot_n_states)) + float('-inf')

    for ix, init_state in enumerate(init_states):
        for offset in range(n_states - 1):
            state = init_state + offset
            A[state, state: state+2] = np.log(.5)
        if n_states > 1:
            A[final_states[ix], final_states[ix]] = np.log(prob_final_state)
    
    # Looping arcs
    for final_state in final_states:
        if n_states > 1:
            A[final_state, init_states] = ins_penalty * np.log((1 - prob_final_state) * \
                                                               mixture_weights)
        else:
            A[final_state, init_states] = np.log(mixture_weights)
    
    return A, init_states, final_states

                
class PhoneLoop(Model):
    """Hidden Markov Model.
    
    """
    
    def __init__(self, n_units, components, alpha, ins_penalty, dp_prior=False):
        """Initialize the HMM.
        
        Parameters
        ----------
        n_units : int
            Number of units in the phone loop.
        components : list
            List of emissions.
        alpha : float
            Hyper-parameters for the symmetrix Dirichlet prior of the 
            mixture weights.
        ins_penalty : float
            Insertion penalty. Values greater than 1 will prefer to remain 
            in the current unit whereas values lower than 1 (and greater 
            than 0) will favorize unit to unit transition.
            
        """
        # Initialize the base class "Model".
        super().__init__()
        
        # Guess the number of states per units.
        self.n_units = n_units
        self.n_states = int(len(components) / n_units)
        
        # Store the Gaussian.
        self.components = components
        
        # Hyper-parameter prior/posterior.
        self.dp_prior = dp_prior
        if dp_prior:
            self.hg1 = np.ones(n_units)
            self.hg2 = np.zeros(n_units) + alpha
            self.pg1 = np.ones(n_units)
            self.pg2 = np.zeros(n_units) + alpha
        else:
            self.halphas = np.ones(n_units) * alpha
            self.palphas = np.ones(n_units) * alpha
        
        # Expected value of the log weights.
        weights = np.ones(self.n_units) / self.n_units
        
        # Create the log transition matrix.
        self.log_A, self.init_states, self.final_states = \
            _log_transition_matrix(n_units, self.n_states, weights, 
                                   ins_penalty)
       
        self.ins_penalty = ins_penalty
        
        # A priori, no optimal ordering of the components.
        self.optimal_order_idx = None
    
    def expected_log_weights(self):
        """Expected value of the log of the weights of the DP.

        Returns
        -------
        E_log_pi : float
            Log weights.

        """
        if self.dp_prior:
            n = self.pg1.shape[0]
            v = psi(self.pg1) - psi(self.pg1+self.pg2)
            nv = psi(self.pg2) - psi(self.pg1+self.pg2)
            for i in range(1, n):
                v[i] += nv[:i].sum()
            retval =  v
        else:
            retval = psi(self.palphas) - psi(self.palphas.sum())
        
        if self.optimal_order_idx is not None:
            return retval[self.optimal_order_idx]
        else:
            return retval
        
    def get_stats(self, X, resps, state_resps):
        """Compute the sufficient statistics for the model.
        
        Parameters
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        resps : numpy.ndarray
            Weights for each frame.
        state_resps : numpy.ndarray
            Per-state weights for each frame.
        
        Returns
        -------
        stats : dict
            Dictionary where 's0', 's1' and 's2' are the keys for the
            zeroth, first and second order statistics respectively.
        
        """
        stats_data = {}
        stats_data[self.id] = {}
        tmp = resps.reshape((resps.shape[0], self.n_units, -1)).sum(axis=2)
        stats_data[self.id]['s0'] = tmp.sum(axis=0)
        s1 = np.zeros_like(stats_data[self.id]['s0'])
        for i in range(len(stats_data[self.id]['s0'])-1):
            s1[i] += stats_data[self.id]['s0'][i+1:].sum()
        stats_data[self.id]['s1'] = s1
        for i, component in enumerate(self.components):
            stats_data[component.id] = {}
            s_resps = state_resps[i]
            weights = (resps[:, i] * s_resps.T).T
            stats_data = {**stats_data, **component.get_stats(X, weights)}
        return stats_data
    
    def forward(self, llhs):
        """Forward recursion.
        
        """
        log_alphas = np.zeros_like(llhs) - np.inf
        log_alphas[0, self.init_states] = self.expected_log_weights()
        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += logsumexp(log_alphas[i-1] + self.log_A.T, axis=1)
        return log_alphas
    
    def backward(self, llhs):
        """Backward recursion.
        
        """
        log_betas = np.zeros_like(llhs) - np.inf
        log_betas[-1, self.final_states] = 0.
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = logsumexp(self.log_A + llhs[i+1] + log_betas[i+1],
                                     axis=1)
        return log_betas
    
    def viterbi(self, llhs):
        backtrack = np.zeros_like(llhs, dtype=int)
        omega = np.zeros(llhs.shape[1]) + float('-inf')
        omega[self.init_states] = llhs[0, self.init_states] + self.expected_log_weights()
        for i in range(1, llhs.shape[0]):
            hypothesis = omega + self.log_A.T
            backtrack[i] = np.argmax(hypothesis, axis=1)
            omega = llhs[i] + hypothesis[range(len(self.log_A)),
                                            backtrack[i]]
        path = [self.final_states[np.argmax(omega[self.final_states])]]
        for i in reversed(range(1, len(llhs))):
            path.insert(0, backtrack[i, path[0]])
        return path

    def decode(self, X, state_label=False):
        c_llhs = np.zeros((X.shape[0], self.n_states * self.n_units))
        comp_resps = []
        for k in range(self.n_states * self.n_units):
            c_llh = self.components[k].expected_log_likelihood(X)
            c_llhs[:, k] = logsumexp(c_llh, axis=1)
        
        path = self.viterbi(c_llhs)
        path = [bisect(self.init_states, state) for state in path]
        return path
        
    def expected_log_likelihood(self, X, ali=None):
        """Expected value of the log likelihood.
        
        Parameters 
        ----------
        X : numpy.ndarray
            Data (N x D) of N frames with D dimensions.
        ali : list of tuple
            Unit level alignment (optional).
        
        Returns
        -------
        E_llh : float
            Expected value of the log-likelihood.
        state_resps : numpy.ndarray
            Per-state responsibility.
            
        """ 
        # If the unit sequence is provided, build a mask to narrow 
        # the search path of the alignments.
        if ali is not None:
            mask = np.zeros((X.shape[0], self.n_states * self.n_units))
            for entry in ali:
                index = int(entry[0])
                start = entry[1]
                end = entry[2]
                if end > X.shape[0]:
                    break
                tmp = np.zeros((end - start, self.n_states * self.n_units)) \
                    + float('-inf')
                tmp[:, index * self.n_states: (index + 1) * self.n_states] = 0.
                mask[start:end] = tmp
            
        c_llhs = np.zeros((X.shape[0], self.n_states * self.n_units))
        comp_resps = []
        for k in range(self.n_states * self.n_units):
            c_llh = self.components[k].expected_log_likelihood(X)
            c_llhs[:, k] = logsumexp(c_llh, axis=1)
            resps = np.exp((c_llh.T - c_llhs[:, k]).T)
            comp_resps.append(resps)
        
        # If the unit sequence is provided, filter the possible paths.
        if ali is not None:
            c_llhs += mask
            
        log_alphas = self.forward(c_llhs)
        log_betas = self.backward(c_llhs)
        log_q_z = log_alphas + log_betas
        norm = logsumexp(log_q_z, axis=1)
        log_q_z = (log_q_z.T - norm).T

        return norm, np.exp(log_q_z), comp_resps 
    
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
        state_resps : numpy.ndarray
            Per-state responsibility.
            
        """        
        for k in range(self.n_states):
            g_llhs[:, k] = self.components[k].log_predictive(X)
        
        log_alphas = self.forward(g_llhs)
        log_betas = self.backward(g_llhs)
        log_q_z = log_alphas + log_betas
        norm = logsumexp(log_q_z, axis=1)
        log_q_z = (log_q_z.T - norm).T
        
        return norm
   
    def log_marginal_evidence(self):
        for k in range(self.n_states):
            g_llhs[:, k] = self.components[k].log_predictive(X)
        
        log_alphas = self.forward(g_llhs)
        log_betas = self.backward(g_llhs)
        log_q_z = log_alphas + log_betas
        norm = logsumexp(log_q_z, axis=1)
        
        return norm
   
    def reorder(self):
        self.optimal_order_idx = None
        E_log_w = self.expected_log_weights()     
        idx = E_log_w.argsort()[::-1]
        new_components = []
        for i in idx:
            start = i * self.n_states
            for k in range(start, start + self.n_states, 1):
                new_components.append(self.components[k])
        self.components = new_components
        
        self.optimal_order_idx = idx
    
    def update(self, stats):
        """Update the posterior parameters given the sufficient
        statistics.
        
        Parameters
        ----------
        stats : dict
            Dictionary of sufficient statistics.
        
        """
        if self.dp_prior:
            self.pg1 = self.hg1 + stats[self.id]['s0']
            self.pg2 = self.hg2 + stats[self.id]['s1']
        else:
            self.palphas = self.halphas + stats[self.id]['s0']
        
        for component in self.components:
            component.update(stats)
        
        # Sort the unit from the most to the least likely.
        if self.dp_prior:
            self.reorder()
        
        # Update the transition matrix.
        E_log_w = self.expected_log_weights()
        E_log_w -= logsumexp(E_log_w)
        prob_fs = np.exp(self.log_A[self.final_states[0], self.final_states[0]])
        for fs in self.final_states:
            if self.n_states > 1:
                self.log_A[fs, self.init_states] = \
                    self.ins_penalty * (np.log((1 - prob_fs)) + E_log_w)
            else:
                self.log_A[fs, init_states] = E_log_w
        
    def KL(self):
        """Kullback-Leibler divergence between the posterior and 
        the prior density.
        
        Returns 
        -------
        ret : float
            KL(q(params) || p(params)).
        
        """
        KL = 0.
        
        if self.dp_prior:
            for i in range(self.pg1.shape[0]):
                a1 = np.array([self.pg1[i], self.pg2[i]])
                a2 = np.array([self.hg1[i], self.hg2[i]])
                KL += gammaln(a1.sum()) - gammaln(a2.sum()) - gammaln(a1).sum() + gammaln(a2).sum()
        else:
            KL = gammaln(self.palphas.sum())
            KL -= gammaln(self.halphas.sum())
            KL -= gammaln(self.palphas).sum()
            KL += gammaln(self.halphas).sum()
            log_weights = self.expected_log_weights()
            KL += (self.palphas - self.halphas).dot(log_weights)
        
        for component in self.components:
            KL += component.KL()
        return KL 
