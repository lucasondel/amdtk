
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


def forward(init_prob, trans_mat, init_states, lhs):
    alphas = np.zeros_like(lhs)
    consts = np.zeros(len(lhs))
    res = lhs[0, init_states] * init_prob
    consts[0] = res.sum()
    alphas[0, init_states] = res / consts[0]
    for i in range(1, lhs.shape[0]):
        res = lhs[i] * trans_mat.T.dot(alphas[i-1])
        consts[i] = res.sum()
        alphas[i] = res / consts[i]

    return np.log(alphas), consts


def backward(trans_mat, final_states, consts, lhs):
    betas = np.zeros_like(lhs)
    betas[-1, final_states] = 1.
    for i in reversed(range(lhs.shape[0]-1)):
        res = trans_mat.dot(lhs[i+1] * betas[i+1])
        betas[i] = res / consts[i+1]

    return np.log(betas)


def forward_backward(init_prob, trans_mat, init_states,
                     final_states, llhs):
    # Scale the log-likelihoods before to exponentiate.
    log_scaling = llhs.max(axis=0)
    scaled_llhs = llhs - log_scaling
    lhs = np.exp(scaled_llhs)

    # Scaled forward-backward algorithm.
    log_alphas, consts = forward(init_prob, trans_mat, init_states, lhs.T)
    log_betas = backward(trans_mat, final_states, consts, lhs.T)


    # Remove the scaling constants.
    acc_lconsts = np.cumsum(log_scaling + np.log(consts))
    acc_reversed_lconsts = np.zeros_like(acc_lconsts)
    acc_reversed_lconsts[0:-1] = np.cumsum((log_scaling + \
        np.log(consts))[::-1])[::-1][1:]
    log_alphas += (acc_lconsts)[:, np.newaxis]
    log_betas += acc_reversed_lconsts[:, np.newaxis]

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

