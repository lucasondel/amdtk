
"""Set of operations for training and using the  Bayesian phone-loop."""

import numpy as np
from scipy.misc import logsumexp
from ..models import DirichletProcessStats
from ..models import MixtureStats
from ..models import GaussianDiagCovStats


def phoneLoopVbExpectation(model, X, Y=None):
    """Estimate the expected value of the different latent variables of
    the model.

    Parameters
    ----------
    model : tuple
        Tuple containing the Dirichlet process and HMM models. See
        :func:`create_model`.
    X : numpy.ndarray
        The data. A matrix (NxD) of N frames with D dimensions.
    Y : numpy.ndarray
        Data on which to compute the accumulated statistics. If none
        the statistics will be accumulated on 'X'.

    Returns
    -------
    E_log_p_X, stats : scalar, tuple
        The expected value of the log-evidence and the statistics to
        update the parameters.

    """
    # Evaluate the log-likelihood of the acoustic model.
    am_llhs, gmm_log_P_Zs = model.evalAcousticModel(X)

    # Forward-backward algorithm.
    E_log_P_X, hmm_log_P_Z, unit_log_resps = model.forwardBackward(am_llhs)

    # If no other features are provided accumulate the stats on the 'X'.
    if Y is None:
        Y = X

    stats = model.stats(Y, unit_log_resps, hmm_log_P_Z, gmm_log_P_Zs)

    return E_log_P_X, stats


def phoneLoopVb1BestExpectation(model, X, seq):
    """Estimate the expected value of the different latent variables of
    the model.

    Parameters
    ----------
    model : tuple
        Tuple containing the Dirichlet process and HMM models. See
        :func:`create_model`.
    X : numpy.ndarray
        The data. A matrix (NxD) of N frames with D dimensions.
   seq : str
         Sequence of units.

    Returns
    -------
    E_log_p_X, stats : scalar, tuple
        The expected value of the log-evidence and the statistics to
        update the parameters.

    """
    # Save tthe total number of component per GMM and the total number
    # of units in the phone loop model.
    ncomponents = model.ncomponents
    nunits = model.nunits

    # Get the index of the initial state of each unit.
    unit_idxs = [model.getStateIndex(unit_name) for unit_name in seq]

    # Unit state indices
    unit_state_idxs = []
    for idx in unit_idxs:
        unit_state_idxs += [idx, idx + 1, idx + 2]

    # Mapping state index to index in the transition matrix..
    state_index = {}
    for i, idx in enumerate(unit_state_idxs):
        try:
            state_index[idx].append(i)
        except KeyError:
            state_index[idx] = [i]

    # Change the model to accept the given sequence of unit.
    model.createLinearTransitionMatrix(len(unit_idxs))

    # Update the component of the model to fit the sequence.
    new_components = [model.components[i] for i in unit_state_idxs]
    model.components = new_components

    # Evaluate the log-likelihood of the acoustic model.
    gmm_E_log_p_X_given_W, gmm_log_P_Zs = model.evalAcousticModel(X)

    # Evaluate the log-likelihood of the HMM states.
    hmm_log_P_Z, log_alphas, log_betas = \
        model.evalLanguageModel(gmm_E_log_p_X_given_W)

    # log-likelihood of the sequence. We compute it to monitor the
    # convergence of the training.
    E_log_P_X = logsumexp(log_alphas[-1])

    # Evaluate the probability of the latent variable of the inifinite
    # mixture.
    dim = int(model.k/model.nstates)
    dp_P_Z = np.exp(hmm_log_P_Z).reshape((X.shape[0], dim, -1)).sum(axis=2)
    dp_P_Z = dp_P_Z.sum(axis=0)
    resp = np.zeros(nunits)
    for i in range(len(dp_P_Z)):
        idx = int(unit_idxs[i]/model.nstates)
        resp[idx] += dp_P_Z[i]

    # Evaluate the statistics for the truncated DP.
    tdp_stats = DirichletProcessStats(resp[:, np.newaxis])

    # Evaluate the statistics of the GMMs.
    gmm_stats = {}
    gaussian_stats = {}
    for state in state_index:
        weights = np.zeros((X.shape[0], ncomponents))
        for index in state_index[state]:
            log_weights = (hmm_log_P_Z[:, index] + gmm_log_P_Zs[index].T).T
            weights += np.exp(log_weights)
        gmm_stats[state] = MixtureStats(weights)
        for j in range(ncomponents):
            gaussian_stats[(state, j)] = \
                GaussianDiagCovStats(X, weights[:, j])
    return E_log_P_X, (tdp_stats, gmm_stats, gaussian_stats)


def phoneLoopVbMaximization(model, stats):
    """Maximization step of the variational bayes training.

    Parameters
    ----------
    model : tuple
        Tuple containing the Dirichlet process and HMM models. See
        :func:`create_model`.
    stats : dictionary
        Accumulated statistics for earch component of the model. See
        :func:`phoneLoopVbExpectation`.

    """
    # Update the Truncated Dirichlet Process.
    model.updatePosterior(stats[0], stats[1], stats[2])


def phoneLoopDecode(model, X, output_states=False):
    """Label the segments using the Viterbi algorithm.

    Parameters
    ----------
    model : tuple
        Tuple containing the Dirichlet process and HMM models. See
        :func:`create_model`.
    X : numpy.ndarray
        The data. A matrix (NxD) of N frames with D dimensions.
    output_states : boolean
        If true, output the path of the HMM state sequence otherwise,
        simply output the path of clusters.
    lscale : float
        Log scale for the external (i.e. cluster) transition weights.
    lscale_full : float
        Log scale for all the transition weights (internal and external).

    Returns
    -------
    path : list
        List of the state of the most probable path.

    """
    # Evaluate the log-likelihood of the acoustic model.
    gmm_E_log_p_X_given_W, gmm_log_P_Zs = model.evalAcousticModel(X)

    # Evaluate the log-likelihood of the HMM states.
    path = model.viterbi(gmm_E_log_p_X_given_W, output_states)

    return path


def phoneLoopPosteriors(model, X, output_states=False):
    """Compute the posterior of the model's unit..

    Parameters
    ----------
    model : :class:`BayesianInfinitePhoneLoop`
        Bayesian Infinite phone-loop model.
    X : numpy.ndarray
        The data. A matrix (NxD) of N frames with D dimensions.
    output_states : boolean
        If true, output the states posteriors.

    Returns
    -------
    path : list
        List of the state of the most probable path.

    """
    # Evaluate the log-likelihood of the acoustic model.
    gmm_E_log_p_X_given_W, gmm_log_P_Zs = model.evalAcousticModel(X)

    # Normalize the log-likelihood of the HMM states to get their
    # posteriors.
    log_norm = logsumexp(gmm_E_log_p_X_given_W, axis=1)
    gmm_E_log_P_Z = np.exp(gmm_E_log_p_X_given_W.T - log_norm).T

    # Merge the inner states of the units to output only the units
    # posteriors.
    if not output_states:
        gmm_E_log_P_Z = gmm_E_log_P_Z.reshape((X.shape[0], model.nunits, -1))
        gmm_E_log_P_Z = gmm_E_log_P_Z.sum(axis=2)

    return gmm_E_log_P_Z


def phoneLoopForwardBackwardPosteriors(model, X, output_states=False):
    """Compute the hmm states posteriors.

    Parameters
    ----------
    model : :class:`BayesianInfinitePhoneLoop`
        Bayesian Infinite phone-loop model.
    X : numpy.ndarray
        The data. A matrix (NxD) of N frames with D dimensions.
    output_states : boolean
        If true, output the states posteriors.

    Returns
    -------
    path : list
        List of the state of the most probable path.

    """
    # Evaluate the log-likelihood of the acoustic model.
    gmm_E_log_p_X_given_W, gmm_log_P_Zs = model.evalAcousticModel(X)

    # Evaluate the log-likelihood of the HMM states.
    hmm_log_P_Z, log_alphas, log_betas = \
        model.evalLanguageModel(gmm_E_log_p_X_given_W)
    hmm_P_Z = np.exp(hmm_log_P_Z)

    # Merge the inner states of the units to output only the units
    # posteriors.
    if not output_states:
        hmm_P_Z = hmm_P_Z.reshape((X.shape[0], model.nunits, -1))
        hmm_P_Z = hmm_P_Z.sum(axis=2)

    return hmm_P_Z
