
"""Set of operations for training and using the  Bayesian phone-loop."""

import numpy as np
from scipy.misc import logsumexp


def phoneLoopVbExpectation(model, X, Y=None, ac_weight=1.0):
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
    ac_weight : float
            Scaling of the acoustic scores.

    Returns
    -------
    E_log_p_X, stats : scalar, tuple
        The expected value of the log-evidence and the statistics to
        update the parameters.

    """
    # Evaluate the log-likelihood of the acoustic model.
    am_llhs, gmm_log_P_Zs = model.evalAcousticModel(X, ac_weight=ac_weight)

    # Forward-backward algorithm.
    E_log_P_X, hmm_log_P_Z, unit_log_resps = model.forwardBackward(am_llhs)

    # If no other features are provided accumulate the stats on the 'X'.
    if Y is None:
        Y = X

    stats = model.stats(Y, unit_log_resps, hmm_log_P_Z, gmm_log_P_Zs)

    return E_log_P_X, stats


def phoneLoopVb1BestExpectation(model, seq, X, Y=None, ac_weight=1.0):
    """Estimate the expected value of the different latent variables of
    the model given a specific path of unit.

    Parameters
    ----------
    model : tuple
        Tuple containing the Dirichlet process and HMM models. See
        :func:`create_model`.
    seq : str
         Sequence of units.
    X : numpy.ndarray
        The data. A matrix (NxD) of N frames with D dimensions.
    Y : numpy.ndarray
        Data on which to compute the accumulated statistics. If none
        the statistics will be accumulated on 'X'.
   ac_weight : float
            Scaling of the acoustic scores.

    Returns
    -------
    E_log_p_X, stats : scalar, tuple
        The expected value of the log-evidence and the statistics to
        update the parameters.

    """
    # Set the decoding graph to match the sequence of unit.
    model.setLinearDecodingGraph(seq)

    # Evaluate the log-likelihood of the acoustic model.
    am_llhs, gmm_log_P_Zs = model.evalAcousticModel(X, ac_weight=ac_weight)

    # Forward-backward algorithm.
    E_log_P_X, hmm_log_P_Z, unit_log_resps = model.forwardBackward(am_llhs)

    # If no other features are provided accumulate the stats on the 'X'.
    if Y is None:
        Y = X

    stats = model.stats(Y, unit_log_resps, hmm_log_P_Z, gmm_log_P_Zs)

    return E_log_P_X, stats


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
        gmm_E_log_P_Z = gmm_E_log_P_Z.reshape((X.shape[0],
                                               len(model.unit_names), -1))
        gmm_E_log_P_Z = gmm_E_log_P_Z.sum(axis=2)

    return gmm_E_log_P_Z


def phoneLoopForwardBackwardPosteriors(model, X, ac_weight=1.0,
                                       output_states=False):
    """Compute the hmm states posteriors.

    Parameters
    ----------
    model : :class:`BayesianInfinitePhoneLoop`
        Bayesian Infinite phone-loop model.
    X : numpy.ndarray
        The data. A matrix (NxD) of N frames with D dimensions.
    ac_weight : float
            Scaling of the acoustic scores.
    output_states : boolean
        If true, output the states posteriors.

    Returns
    -------
    path : list
        List of the state of the most probable path.

    """
    # Evaluate the log-likelihood of the acoustic model.
    am_llhs, gmm_log_P_Zs = model.evalAcousticModel(X, ac_weight=ac_weight)

    # Forward-backward algorithm.
    E_log_P_X, hmm_log_P_Z, unit_log_resps = model.forwardBackward(am_llhs)
    
    if not output_states:
        posts = np.exp(unit_log_resps)
    else:
        posts = np.exp(hmm_log_P_Z)

    return posts
