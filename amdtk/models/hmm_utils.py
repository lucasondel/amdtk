
"""
Utilities for HMM.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
from scipy.special import logsumexp
from scipy.sparse import csc_matrix


def create_phone_loop_transition_matrix(n_units, n_states, unigram_lm):
    """Create phone loop transition matrix."""
    tot_n_states = n_units * n_states
    init_states = np.arange(0, tot_n_states, n_states)
    final_states = init_states + n_states - 1
    trans_mat = np.zeros((tot_n_states, tot_n_states))

    for idx, init_state in enumerate(init_states):
        for offset in range(n_states - 1):
            state = init_state + offset
            trans_mat[state, state: state + 2] = 0.5
        if n_states > 1:
            trans_mat[final_states[idx], final_states[idx]] = 0.5

    for idx, final_state in enumerate(final_states):
        if n_states > 1:
            # Disallow repeating a unit.
            weights = unigram_lm.copy()
            weights[idx] = 0.
            weights /= weights.sum()

            trans_mat[final_state, init_states] = .5 * weights
        else:
            trans_mat[final_state, init_states] = unigram_lm

    return csc_matrix(trans_mat), init_states, final_states


def create_linear_transition_matrix(n_units, n_states):
    """Create "linear" transition matrix."""
    tot_n_states = n_units * n_states
    init_states = np.array([0])
    final_states = np.array([tot_n_states - 1])
    trans_mat = np.zeros((tot_n_states, tot_n_states))
    for idx in range(n_units):
        init_state = idx * n_states
        for offset in range(n_states):
            state = init_state + offset
            trans_mat[state, state:state + 2] = 0.5
    trans_mat[-1, -1] = 1.
    return csc_matrix(trans_mat)


def forward(init_log_prob, log_trans_mat, init_states, llhs):
    log_alphas = np.zeros_like(llhs) - np.inf
    log_alphas[0, init_states] = llhs[0, init_states] + init_log_prob
    for i in range(1, llhs.shape[0]):
        log_alphas[i] = llhs[i]
        log_alphas[i] += logsumexp(log_alphas[i-1] + log_trans_mat.T, axis=1)
    return log_alphas


def backward(log_trans_mat, final_states, llhs):
    log_betas = np.zeros_like(llhs) - np.inf
    log_betas[-1, final_states] = 0.
    for i in reversed(range(llhs.shape[0]-1)):
        log_betas[i] = logsumexp(log_trans_mat + llhs[i+1] + log_betas[i+1],
                                 axis=1)
    return log_betas


def forward_backward(init_prob, trans_mat, init_states,
                     final_states, llhs):
    # Take the log of initial/transition probabilities.
    init_log_prob = np.log(init_prob)
    log_trans_mat = np.log(trans_mat.toarray())

    # Scaled forward-backward algorithm.
    log_alphas = forward(init_log_prob, log_trans_mat, init_states, llhs.T)
    log_betas = backward(log_trans_mat, final_states, llhs.T)

    return log_alphas, log_betas

def viterbi(init_prob, trans_mat, init_states, final_states, llhs):
    backtrack = np.zeros_like(llhs, dtype=int)
    omega = np.zeros(llhs.shape[1]) + float('-inf')
    omega[init_states] = llhs[0, init_states] + np.log(init_prob)
    log_A = np.log(trans_mat.toarray())

    for i in range(1, llhs.shape[0]):
        hypothesis = omega + log_A.T
        backtrack[i] = np.argmax(hypothesis, axis=1)
        omega = llhs[i] + hypothesis[range(len(log_A)),
                                     backtrack[i]]

    path = [final_states[np.argmax(omega[final_states])]]
    for i in reversed(range(1, len(llhs))):
        path.insert(0, backtrack[i, path[0]])

    return path

