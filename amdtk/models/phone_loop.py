
"""non-parametric Bayesian phone-loop model."""

# NOTE:
# Differing from what has been described in the paper, we do not put any
# prior on the transition matrix as:
#  * optimizing the states of the sub HMM does not help and is
#    time consuming
#  * transition between sub-HMMs are hanlded by weights taken
#    from the Dirichlet process.

import numpy as np
from scipy.linalg import block_diag
from scipy.misc import logsumexp
from .multivariate_gaussian import BayesianGaussianDiagCov
from .mixture import BayesianMixture
from .discrete_latent_model import DiscreteLatentModel
from .dirichlet_process import TruncatedDirichletProcess
from .hpyp import EMPTY_CONTEXT


def createHmmSkeleton(nstates, max_jump=1):
    """Create a left-to-rigth HMM skeleton.

    The skeleton of a HMM is a binary transition matrix where the 1s
    indicate a possible transition. This method creates only skeleton
    based on the left-to-right topology.

    Parameters
    ----------
    nstates : int
        The number of states of the HMM.
    max_jump : int
        The maximum number of state that a given state can reach. For
        example, if the HMM has 3 states and 'max_jump' is 2 then the
        first state will be connected to the second and third states.

    Returns
    -------
    skeleton : numpy.ndarray
        Matrix of nstates x nstates elements.

    """
    skeleton = np.zeros((nstates, nstates), dtype=np.int)
    for i in range(nstates):
        skeleton[i, i:i+max_jump+1] = 1
    return skeleton


class BayesianInfinitePhoneLoop(DiscreteLatentModel):
    """Bayesian Infinite phone-loop model.

    Attributes
    ----------
    nstates : int
        Number of states per sub-HMM (i.e. units).
    ncomponents : int
        Number of gaussians per states.
    nunits : int
        Number of acoustic units in the loop.
    prior : :class:`TruncatedDirichletProcess`
        Prior of the probability of the units.
    posterior : :class:`TruncatedDirichletProcess`
        POsterrior of the probability of the units.
    init_states : list
        Indices of all possible initial states.
    final_states : list
        Indices of all possible final states.
    log_A : numpy.ndarray
        Expected value of the log transition matrix.

    Methods
    -------
    updateParams()
        Update the parameters of the model.
    evalAcousticModel(X)
        Compute the expected value of the log-likelihood of the
        acoustic model of the phone loop.
    evalLanguageModel(am_llh)
        Rescore the acoustic model using the language model of the
        phone-loop.
    forward(llhs)
        Forward recursion given the log emission probabilities and
        the HMM model.
    backward(llhs)
        Backward recursion given the log emission probabilities and
        the HMM model.
    viterbi(am_llhs)
        Viterbi algorithm.
    KLPosteriorPrior()
        KL divergence between the posterior and the prior densities.
    updatePosterior(tdp_stats, gmm_stats, gauss_stats)
        Update the parameters of the posterior distribution according
        to the accumulated statistics.
    """

    def __init__(self, trunc, ctrt, eta, nstates, alpha, ncomponents, mu,
                 kappa, a, b, mean, cov):
        """Create a (infinite) mixture of HMM/GMM

        Parameters
        ----------
        trunc : int
            Order of the truncation of the Dirichlet process. In other
            words, maximum number of component in the "infinite" mixture
            model.
        ctrt : float
            Concentration parameter of the Dirichlet process.
        eta : float
            Hyper parameter for the symmetric Dirichlet distribution of
            each row of the HMM transition matrix.
        nstates : int
            Number of states for the HMM component.
        alpha : float
            Hyper parameter for the symmetric Dirichlet distribution of
            the weights for each mixture.
        ncomponents : int
            Number of Gaussian per mixture.
        mu : numpy.ndarray
            Hyper parameter for the mean of each Gaussian.
        kappa : float
            Coefficient of the precision of the norma-gamma
            distribution.
        a : float
            Scale parameter of the gamma distribution.
        b : numpy.ndarray
            Rate parameter of the gamma distribution.
        mean : numpy.ndarray
            Mean of the data set for the initialization.
        cov : numpy.ndarray
            Diagonal of the covariance matrix of the data set for the
            initialization.

        Returns
        -------
        model : tuple
            The created model composed of a Dirichlet process and a HMM
            model.

        """
        self.nstates = nstates
        self.ncomponents = ncomponents
        self.nunits = trunc

        # Initialize the prior of the the phone loop weights.
        g1 = np.ones(trunc)
        g2 = np.zeros(trunc) + ctrt
        self.prior = TruncatedDirichletProcess(g1, g2)
        self.posterior = TruncatedDirichletProcess(g1.copy(), g2.copy())

        # Initialize all the GMMs.
        gmms = []
        for i in range(trunc):
            for j in range(nstates):
                gaussians = []
                for k in range(ncomponents):
                    dc = np.diag(cov)
                    mu_n = mu + np.random.multivariate_normal(mean, dc)
                    gaussian = BayesianGaussianDiagCov(mu, kappa, a, b, mu_n,
                                                       kappa, a, b.copy())
                    gaussians.append(gaussian)
                gmms.append(BayesianMixture(alpha*np.ones(ncomponents),
                                            gaussians))

        # Initialize the DiscreteLatentModel super class.
        super().__init__(gmms)

        # Define the possible initial and final states.
        self.init_states = [0]
        self.final_states = [nstates - 1]
        for i in range(1, trunc):
            self.final_states.append(self.final_states[i-1] + nstates)
            self.init_states.append(self.final_states[-1]-nstates+1)

        # Create the graph of the phone loop.
        l2r_skeleton = createHmmSkeleton(nstates, max_jump=1)
        skeletons = [l2r_skeleton for i in range(trunc)]
        skeleton = block_diag(*skeletons)
        for final_state in self.final_states:
            for init_state in self.init_states:
                skeleton[final_state, init_state] = 1

        # We disable the warnings as we divide by 0. This is not an
        # error and
        # the '-inf' results is handled later on.
        old_settings = np.seterr(divide='ignore')
        self.log_A = np.log((skeleton.T/skeleton.sum(axis=1)).T)
        np.seterr(**old_settings)
        
        # By default no language model.
        self.lm = None

        # Correct the transition matrix and evaluate the probability of
        # transition between sub HMMs.
        self.updateParams()

    @property
    def lm(self):
        return self.__lm

    @lm.setter
    def lm(self, value):
        self.__lm = value

    def getStateIndex(self, unit_name):
        """ inverse func of below one """
        return (int(unit_name[1:]) - 1) * self.nstates

    def getUnitName(self, unit_index):
        return 'a' + str(int(unit_index/self.nstates) + 1)

    def getUnitId(self, unit_index):
        unit_name = self.getUnitName(unit_index)
        try:
            unit_id = self.lm.vocab[unit_name]
        except KeyError:
            unit_id = len(self.lm.vocab) + 1
            self.lm.vocab[unit_name] = unit_id
        return unit_id

    def createLinearTransitionMatrix(self, seq_len):
        """ seq length """
        
        # Probability of initial state.
        self.log_pi = np.zeros(seq_len * self.nstates) - np.inf
        self.log_pi[0] = 0.

        # Transition matrix
        self.log_A = np.zeros((self.log_pi.shape[0], 
                               self.log_pi.shape[0]), dtype=float) 
        self.log_A -= np.inf
        dim, _ = self.log_A.shape
        diag_indices = np.diag_indices(dim)
        self.log_A[diag_indices] = np.log(0.5)
        self.log_A[diag_indices[0][:dim-1], 
                   diag_indices[1][:dim-1] + 1] = np.log(0.5)
        self.log_A[-1, -1] = np.log(1.)

        # Update the initial/final state litss.
        self.init_states = [0]
        self.final_states = [dim-1]

    def logProbLm(self, s_unit_index, e_unit_index):
        s_unit_id = self.getUnitId(s_unit_index)
        e_unit_id = self.getUnitId(e_unit_index)
        context = tuple([e_unit_id])
        prob = self.lm.predictiveProbability(self.lm.order, context,
                                             s_unit_id)
        return np.log(prob)

    def updateParams(self):
        """Update the parameters of the model."""
        self.log_pi = np.zeros(self.k) - np.inf
        if self.lm is None:
            exp_log_weights = self.posterior.expLogPi()
            self.log_pi[self.init_states] = exp_log_weights
            for fs in self.final_states:
                self.log_A[fs, self.init_states] = exp_log_weights
        else:
            # Beginning of utterance.
            level = self.lm.order
            context = tuple([EMPTY_CONTEXT])
            for ss in self.init_states:
                s_unit_id = self.getUnitId(ss)
                prob = self.lm.predictiveProbability(level, context, s_unit_id)
                self.log_pi[ss] = np.log(prob)
            
            # Bigram transitions. 
            for ss in self.init_states:
                for fs in self.final_states:
                    self.log_A[fs, ss] = self.logProbLm(ss, fs)


    def evalAcousticModel(self, X):
        """Compute the expected value of the log-likelihood of the
        acoustic model of the phone loop.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix of N frames with D dimensions.

        Returns
        -------
        E_llh : numpy.ndarray
            The expected value of the log-likelihood for each frame.
        log_p_Z ; numpy.ndarray
            Log probability of the discrete latent variables of the
            acoustic  model.

        """
        gmm_E_log_p_X_given_W = np.zeros((X.shape[0], self.k))
        gmm_log_P_Zs = []
        for i, gmm in enumerate(self.components):
            llh, resps = gmm.expLogLikelihood(X)
            gmm_E_log_p_X_given_W[:, i] = llh
            gmm_log_P_Zs.append(resps)

        return gmm_E_log_p_X_given_W, gmm_log_P_Zs

    def evalLanguageModel(self, am_llh):
        """Rescore the acoustic model using the language model of the
        phone-loop.

        Parameters
        ----------
        am_llh : numpy.ndarray
            Acoustic model log likelihood.

        Returns
        -------
        E_log_P_Z : numpy.ndarray
            The expected value of responsibility of each state.
        log_alpha ; numpy.ndarray
            Log of the forward values.
        log_beta ; numpy.ndarray
            Log of the backward values.

        """
        log_alphas = self.forward(am_llh)
        log_betas = self.backward(am_llh)
        log_P_Z = log_alphas + log_betas
        log_P_Z = (log_P_Z.T - logsumexp(log_P_Z, axis=1)).T

        return log_P_Z, log_alphas, log_betas

    def forward(self, llhs):
        """Forward recursion given the log emission probabilities and
        the HMM model.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM model.

        Returns
        -------
        log_alphas : numpy.ndarray
            The log alphas values of the recursions.

        """
        log_alphas = np.zeros_like(llhs) - np.inf
        log_alphas[0] = llhs[0] + self.log_pi
        for i in range(1, llhs.shape[0]):
            log_alphas[i] = llhs[i]
            log_alphas[i] += logsumexp(log_alphas[i-1] + self.log_A.T, axis=1)
        return log_alphas

    def backward(self, llhs):
        """Backward recursion given the log emission probabilities and
        the HMM model.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM model.

        Returns
        -------
        log_alphas : numpy.ndarray
            The log alphas values of the recursions.

        """
        log_betas = np.zeros_like(llhs) - np.inf
        log_betas[-1, self.final_states] = 0.
        for i in reversed(range(llhs.shape[0]-1)):
            log_betas[i] = logsumexp(self.log_A + llhs[i+1] + log_betas[i+1],
                                     axis=1)
        return log_betas

    def viterbi(self, am_llhs):
        """Viterbi algorithm.

        Parameters
        ----------
        llhs : numpy.ndarray
            Log of the emission probabilites with shape (N x K) where N
            is the number of frame in the sequence and K is the number
            of state in the HMM model.

        Returns
        -------
        path : list
            List of the state of the mostr probable path.

        """
        backtrack = np.zeros_like(am_llhs, dtype=int)
        omega = am_llhs[0] + self.log_pi
        for i in range(1, am_llhs.shape[0]):
            hypothesis = omega + self.log_A.T
            backtrack[i] = np.argmax(hypothesis, axis=1)
            omega = am_llhs[i] + hypothesis[range(len(self.log_A)),
                                            backtrack[i]]
        path = [self.final_states[np.argmax(omega[self.final_states])]]
        for i in reversed(range(1, len(am_llhs))):
            path.insert(0, backtrack[i, path[0]])
        return path

    def KLPosteriorPrior(self):
        """KL divergence between the posterior and the prior densities.

        Returns:class:MixtureStats
        -------
        KL : float
            KL divergence.

        """
        KL = 0
        for gmm in self.components:
            KL += gmm.KLPosteriorPrior()
        return KL + self.posterior.KL(self.prior)

    def updatePosterior(self, tdp_stats):
        """Update the parameters of the posterior distribution according
        to the accumulated statistics.

        Parameters
        ----------
        tdp_stats : :class:MixtureStats
            Statistics for the truncated DP.

        """
        self.posterior = self.prior.newPosterior(tdp_stats)
        self.updateParams()
