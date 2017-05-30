
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
from .model import PersistentModel
from .mlp_utils import GaussianNeuralNetwork
from .mlp_utils import NeuralNetwork


class SVAE(object):

    def __init__(self, encoder_structure, decoder_structure,
                 prior, var_reg=-1, no_llh=False, kl_weights=1.,
                 classifier_structure=None):
        self.encoder = GaussianNeuralNetwork(encoder_structure, [])
        self.params = self.encoder.params
        if not no_llh:
            self.decoder = GaussianNeuralNetwork(decoder_structure, [],
                                                 self.encoder.sample)
            self.params += self.decoder.params

        self.var_reg = var_reg
        self.no_llh = no_llh
        self.kl_weights = kl_weights
        self.prior = prior
        if classifier_structure is not None:
            self.classifier = classifier_structure
            #self.classifier = NeuralNetwork(classifier_structure, [],
            #                                self.encoder.sample)
            #self.params += self.classifier.params
        else:
            self.classifier = None

        self._build()

    def _build(self):
        post_mean = self.encoder.mean
        post_var = self.encoder.var

        # KL divergence posterior/prior.
        prior_mean = T.matrix(dtype=theano.config.floatX)
        prior_var = T.matrix(dtype=theano.config.floatX)
        kl_div = (.5 * (prior_mean - post_mean)**2) / prior_var
        ratio = post_var / prior_var
        kl_div += .5 * (ratio - 1 - T.log(ratio))
        kl_div = T.sum(kl_div, axis=1)

        # Log-likelihood of the data (up to a constant).
        if not self.no_llh:
            targets = self.encoder.inputs
            mean = self.decoder.outputs
            var = self.decoder.var
            llh = -.5 * T.sum(T.log(var), axis=1)
            llh += -.5 * T.sum(((targets - mean) ** 2) / var, axis=1)
            if self.var_reg > 0:
                llh += T.sum(T.log(self.var_reg) - self.var_reg * var, axis=1)
        else:
            llh = 0.

        # Evidence Lower-Bound.
        resps = T.matrix()
        #if self.classifier is not None:
            #classifier_llh = T.sum(resps * T.log(self.classifier.outputs), axis=1)
            #llh += classifier_llh

            #self.classify = theano.function(
            #    inputs=[self.encoder.inputs],
            #    outputs=T.argmax(self.classifier.outputs, axis=1)
            #)

        #s_stats = T.concatenate([post_var + post_mean**2, post_mean], axis=1)
        s_stats = T.concatenate([self.encoder.sample**2, self.encoder.sample], axis=1)
        pad_s_stats = T.concatenate([s_stats, T.ones_like(s_stats)], axis=1)
        prediction = self.prior.sym_classify(pad_s_stats)
        classifier_llh = T.sum(resps * T.log(prediction), axis=1)
        if self.classifier is not None:
            llh += classifier_llh

        elbo = T.sum(llh - self.kl_weights * kl_div)

        self._get_gradients = theano.function(
            inputs=[self.encoder.inputs, prior_mean, prior_var, resps],
            outputs=[T.sum(llh), self.kl_weights * T.sum(kl_div)] + \
                [T.grad(elbo, param) for param in self.params],
            on_unused_input='ignore'
        )

        self.classify = theano.function(
            inputs=[self.encoder.inputs],
            outputs=T.argmax(prediction, axis=1)
        )

    def decode(self, prior, data, state_path=False):
        mean, var = self.encoder.forward(data)
        return prior.decode(mean, state_path)

    def classify(self, prior, data):
        mean, var = self.encoder.forward(data)

        # Expected value of the sufficient statistics.
        s_stats = np.c_[mean**2 + var, mean,
                        np.ones((len(mean), 2 * mean.shape[1]))]

        # Clustering.
        log_norm, resps, acc_stats = prior.get_resps(s_stats)

        return resps[0].T.argmax(axis=1)

    def get_gradients(self, prior, data, log_resps=None):
        mean, var = self.encoder.forward(data)


        # Expected value of the sufficient statistics.
        s_stats = np.c_[mean**2 + var, mean,
                        np.ones((len(mean), 2 * mean.shape[1]))]

        # Clustering.
        log_norm, resps, acc_stats = prior.get_resps(s_stats, log_resps)
        #log_norm, resps, acc_stats = prior.get_resps(s_stats)

        # Expected value of the prior's components parameters.
        dim_latent = self.encoder.layers[-1].dim_out
        p_np1 = [comp.posterior.grad_log_partition[:dim_latent]
                 for comp in prior.components]
        p_np2 = [comp.posterior.grad_log_partition[dim_latent:2 * dim_latent]
                 for comp in prior.components]
        q_np1 = resps[0].T.dot(p_np1)
        q_np2 = resps[0].T.dot(p_np2)

        # Convert the natural parameters to the standard parameters.
        prior_var = -1 / (2 * q_np1)
        prior_mean = q_np2 * prior_var

        val_and_grads = self._get_gradients(data, prior_mean, prior_var,
                                            resps[0].T)
        #val_and_grads = self._get_gradients(data, prior_mean, prior_var,
        #                                    np.exp(log_resps))
        return val_and_grads[0] - val_and_grads[1], val_and_grads[2:], \
               acc_stats


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

