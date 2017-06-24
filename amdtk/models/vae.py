
"""
Variational Auto-Encoder.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

import abc
import pickle
import numpy as np
import theano
import theano.tensor as T
from ..io import PersistentModel
from .mlp_utils import GaussianNeuralNetwork
from .mlp_utils import NeuralNetwork


class SVAE(object):

    def __init__(self, encoder_struct, decoder_struct, prior_latent,
                n_samples=10):

        self.encoder = GaussianNeuralNetwork(encoder_struct, [],
                                             n_samples=n_samples)
        self.decoder = GaussianNeuralNetwork(decoder_struct, [],
                                             self.encoder.sample)
        self.params = self.encoder.params + self.decoder.params
        self.prior_latent = prior_latent
        self._build()

    def _build(self):
        # Mean and variance of the decoder.
        mean = T.reshape(
            self.decoder.mean,
            (self.encoder.n_samples, self.encoder.mean.shape[0], -1)
        )
        var = T.reshape(
            self.decoder.var,
            (self.encoder.n_samples, self.encoder.mean.shape[0], -1)
        )

        # Log-likelihood.
        targets = self.encoder.inputs
        llh = -.5 * T.sum(T.log(var).mean(axis=0), axis=1)
        llh += -.5 * T.sum((((targets - mean) ** 2) / var).mean(axis=0),
                           axis=1)
        llh = T.sum(llh)

        # Mean and variance of the encoder (variational distribution).
        mean = self.encoder.mean
        var = self.encoder.var

        # KL divergence posterior/prior.
        prior_mean = T.matrix(dtype=theano.config.floatX)
        prior_var = T.matrix(dtype=theano.config.floatX)
        kl_div = .5 * T.log(prior_var / var) - .5
        kl_div += ((prior_mean - mean)**2 + var) / (2 * prior_var)
        kl_div = T.sum(kl_div)

        # Variational objective function.
        objective = llh - kl_div

        # Gradient function of the neural network.
        self._get_gradients = theano.function(
            inputs=[self.encoder.inputs, prior_mean, prior_var],
            outputs=[objective] + \
                [T.grad(objective, param) for param in self.params],
        )

        # Forward and input to the encoder network.
        self.forward = theano.function(
            inputs=[self.encoder.inputs],
            outputs=[mean, var]
        )

    def generate_features(self, data):
        mean, var, predictions = self.forward(data)

        # Expected value of the sufficient statistics.
        s_stats = np.c_[mean**2 + var, mean,
                        np.ones((len(mean), 2 * mean.shape[1]))]

        return s_stats

    def decode(self, data):
        mean, var = self.forward(data)

        # Expected value of the sufficient statistics.
        s_stats = np.c_[mean**2 + var, mean,
                        np.ones((len(mean), 2 * mean.shape[1]))]

        # Clustering.
        return self.prior_latent.decode(s_stats, is_s_stats=True)
        #resps, _, _ = self.prior_latent.get_posteriors(s_stats, True)

        #return np.argmax(resps, axis=1)

    def get_gradients(self, data, alignments=None):
        mean, var = self.forward(data)

        # Expected value of the sufficient statistics.
        s_stats = np.c_[mean**2 + var, mean,
                        np.ones((len(mean), 2 * mean.shape[1]))]

        # Clustering.
        posts, _, acc_stats = \
            self.prior_latent.get_posteriors(s_stats, accumulate=True,
                                             alignments=alignments,
                                             gauss_posteriors=True)
        print(posts.shape)

        # Expected value of the prior's components parameters.
        dim_latent = self.encoder.layers[-1].dim_out
        p_np1 = [comp.posterior.grad_log_partition[:dim_latent]
                 for comp in self.prior_latent.components]
        p_np2 = [comp.posterior.grad_log_partition[dim_latent:2 * dim_latent]
                 for comp in self.prior_latent.components]
        q_np1 = posts.T.dot(p_np1)
        q_np2 = posts.T.dot(p_np2)

        # Convert the natural parameters to the standard parameters.
        prior_var = -1 / (2 * q_np1)
        prior_mean = q_np2 * prior_var

        # Gradients of the objective function w.r.t. the parameters of
        # the neural network (encoder + decoder).
        val_and_grads = self._get_gradients(data, prior_mean, prior_var)
        objective, grads = val_and_grads[0], val_and_grads[1:]

        return objective, acc_stats, grads

class MLPClassifier(object):

    def __init__(self, structure):
        self.nnet = NeuralNetwork(structure, [])
        self.params = self.nnet.params
        self._build()

    def _build(self):

        # Evidence Lower-Bound.
        resps = T.matrix()
        prediction = self.nnet.outputs
        llh = T.sum(resps * T.log(prediction))

        self._get_gradients = theano.function(
            inputs=[self.nnet.inputs, resps],
            outputs=[llh] + \
                [T.grad(llh, param) for param in self.params],
        )

        self.classify = theano.function(
            inputs=[self.nnet.inputs],
            outputs=T.argmax(prediction, axis=1)
        )

    def get_gradients(self, data, log_resps):
        val_and_grads = self._get_gradients(data, np.exp(log_resps))
        return val_and_grads[0], val_and_grads[1:]


