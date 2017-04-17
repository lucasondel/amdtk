
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
from .mlp_utils import MLPGaussian
from .sga_training import AdamSGATheano


class MLPEncoder(MLPGaussian):
    """Encoding MLP for the VAE."""

    def __init__(self, dim_fea, dim_latent, n_layers, n_units, activation,
                 log_var_prior, non_informative_prior=False):
        """Initialize a MLP.

        Parameters
        ----------
        dim_fea : int
            Dimension of the input.
        dim_latent : int
            Dimension of the latent variable.
        n_layers : int
            Number of hidden layers.
        n_units : int
            Number of units per layer.
        activation : function
            Non-linear activation.

        """
        self.dim_fea = dim_fea
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.log_var_prior = log_var_prior
        self.non_informative_prior = non_informative_prior

        self._build()

    def _build(self):
        # Input to the encoder.
        self.input = T.matrix('x', dtype=theano.config.floatX)

        # Create the MLP Gaussian structure.
        MLPGaussian.__init__(self, self.input, self.dim_fea, self.dim_latent,
                             self.n_layers, self.n_units, self.activation)

        # Noise variable for the reparameterization trick.
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams()
        else:
            srng = T.shared_randomstreams.RandomStreams()
        self.eps = srng.normal(self.mean.shape)

        # Latent variable.
        self.output = self.mean + T.exp(.5 * self.log_var) * self.eps

        # KL divergence between the posterior and the prior distribution over
        # the latent variable.
        if not self.non_informative_prior:
            #self.kl_div = .5 * T.sum(-1 - self.log_var + T.exp(self.log_var) +
            #                         self.mean**2, axis=1)
            var_prior = np.exp(self.log_var_prior)
            var = T.exp(self.log_var)
            self.kl_div = .5 * (self.log_var_prior - self.log_var)
            self.kl_div += .5 * (self.mean**2) / var_prior
            self.kl_div += .5 * ((var / var_prior) - 1)
            self.kl_div = T.sum(self.kl_div, axis=1)
        else:
            # In this case this is just the negative entropy of the
            # posterior.
            self.kl_div = -.5 * T.sum(np.log(2 * np.pi + self.log_var + 1),
                                     axis=1)



class MLPDecoder(MLPGaussian):
    """Decoding MLP for the VAE."""

    def __init__(self, encoder, dim_fea, dim_latent, n_layers, n_units,
                 activation):
        """Initialize a MLP.

        Parameters
        ----------
        encoder : :class:`MLPEncoder`
            Encoding MLP.
        dim_fea : int
            Dimension of the features feeded as input to the encoder.
        dim_latent : int
            Dimension of the latent variable.
        n_layers : int
            Number of hidden layers.
        n_units : int
            Number of units per layer.
        activation : function
            Non-linear activation.

        """
        self.encoder = encoder
        self.dim_fea = dim_fea
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation

        self._build()

    def _build(self):
        # Create the MLP Gaussian structure.
        MLPGaussian.__init__(self, self.encoder.output, self.dim_latent,
                             self.dim_fea, self.n_layers, self.n_units,
                             self.activation)

        # Log-likelihood (up to a constant).
        self.llh = T.sum(-.5 * (self.log_var +
                         ((self.encoder.input - self.mean)**2) /
                         T.exp(self.log_var)),
                         axis=1)

        # Noise variable for the decoder.
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams()
        else:
            srng = T.shared_randomstreams.RandomStreams()
        self.eps = srng.normal(self.mean.shape)

        # Decoded value.
        self.output = self.mean + 0 * T.exp(.5 * self.log_var) * self.eps


class VAE(PersistentModel, AdamSGATheano):
    """Variational Auto-Encoder."""

    def __init__(self, mean, precision, dim_fea, dim_latent, n_layers, n_units,
                 activation, log_var_prior, non_informative_prior=False):
        """Initialize a VAE.

        Parameters
        ----------
        mean : numpy.ndarray
            Mean of the data set.
        precision : numpy.ndarray
            Precision of the data set.
        dim_fea : int
            Dimension of the input features.
        dim_latent : int
            Dimension of the latent variable.
        n_layers : int
            Number of hidden layers.
        n_units : int
            Number of units per layer.
        activation : function
            Non-linear activation.

        """
        self.dim_fea = dim_fea
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.log_var_prior = log_var_prior
        self.non_informative_prior = non_informative_prior

        self._build()

        # Magic initialization ! This initialization seems to work
        # well when the featurea are whitened.
        values = np.random.normal(0., 1, (n_units, dim_latent))
        values = np.asarray(values, dtype=theano.config.floatX)
        self.encoder.mean_layer.params[0].set_value(values)
        values = np.asarray(np.zeros(dim_latent) -5., dtype=theano.config.floatX)
        self.encoder.log_var_layer.params[1].set_value(values)

        # Specific initialization of the bias (log-variance).
        #values = np.random.normal(0., .1, (n_units, dim_fea))
        #values = np.asarray(values, dtype=theano.config.floatX)
        #self.decoder.mean_layer.params[0].set_value(values)
        #values = np.asarray(mean, dtype=theano.config.floatX)
        #self.decoder.mean_layer.params[1].set_value(values)
        values = np.asarray(-np.ones_like(precision)*3, dtype=theano.config.floatX)
        self.decoder.log_var_layer.params[1].set_value(values)

    def _build(self):
        # Encoding network.
        self.encoder = MLPEncoder(
            self.dim_fea,
            self.dim_latent,
            self.n_layers,
            self.n_units,
            self.activation,
            self.log_var_prior,
            self.non_informative_prior
        )

        # Decoding network.
        self.decoder = MLPDecoder(
            self.encoder,
            self.dim_fea,
            self.dim_latent,
            self.n_layers,
            self.n_units,
            self.activation
        )

        # Utilities..
        self.encode = theano.function([
            self.encoder.input],
            outputs=self.encoder.output
        )
        self.decode = theano.function([
            self.encoder.input],
            outputs=self.decoder.output
        )
        self.kl = theano.function([
            self.encoder.input],
            outputs=self.encoder.kl_div
        )
        self.llh = theano.function([
            self.encoder.input],
            outputs=self.decoder.llh
        )

        # Lower bound. Note that because we use the accumulated
        # log-likelihod rather than the mean the magnitude of the
        # gradients may be very large. We rescale it during the
        # inference.
        llh = self.decoder.llh - self.encoder.kl_div
        objective = T.sum(llh)
        self.log_likelihood = theano.function(
            inputs=[self.encoder.input],
            outputs=llh
        )

        # Parameters to update.
        self.params = self.encoder.params + self.decoder.params

        # Gradient of cost function with respect to the parameters.
        gradients = [T.grad(objective, param) for param in self.params]
        self.get_gradients = theano.function(
            inputs=[self.encoder.input],
            outputs=[self.decoder.llh] + gradients
        )


        # ADAM SGA training.
        AdamSGATheano.__init__(
            self,
            [],
            self.params,
            gradients
        )


    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'dim_fea': self.dim_fea,
            'dim_latent': self.dim_latent,
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'activation': self.activation,
            'log_var_prior': self.log_var_prior,
            'non_informative_prior': self.non_informative_prior,
            'params': [param.get_value() for param in self.params]
        }

    @staticmethod
    def load_from_dict(model_data):
        model = VAE.__new__(VAE)

        model.dim_fea = model_data['dim_fea']
        model.dim_latent = model_data['dim_latent']
        model.n_layers = model_data['n_layers']
        model.n_units = model_data['n_units']
        model.activation = model_data['activation']
        model.log_var_prior = model_data['log_var_prior']
        model.non_informative_prior = model_data['non_informative_prior']

        model._build()
        for model_param, param in zip(model.params, model_data['params']):
            model_param.set_value(param)

        return model

    # -----------------------------------------------------------------


class MLPEncoderGMM(MLPEncoder):
    """Encoding MLP for the VAE GMM prior over the latent variable."""

    def __init__(self, dim_fea, dim_latent, n_layers, n_units, activation,
                 log_var_prior, non_informative_prior):
        """Initialize a MLP.

        Parameters
        ----------
        dim_fea : int
            Dimension of the input.
        dim_latent : int
            Dimension of the latent variable.
        n_layers : int
            Number of hidden layers.
        n_units : int
            Number of units per layer.
        activation : function
            Non-linear activation.

        """
        MLPEncoder.__init__(
            self,
            dim_fea,
            dim_latent,
            n_layers,
            n_units,
            activation,
            log_var_prior,
            non_informative_prior
        )

    def _build(self):
        MLPEncoder._build(self)

        # Create a function to output the natural parameters of the
        # Gaussian posteriors.
        np1 = - 1 / (2 * T.exp(self.log_var))
        np2 = self.mean / (T.exp(self.log_var))
        nparams = T.concatenate([np1, np2], axis=1)
        self.natural_params = theano.function(
            inputs=[self.input],
            outputs=[np1, np2]
        )
        self.np1 = np1
        self.np2 = np2

        # Re-parameterization trick. The reparameterization is slightly
        # more complex in this case as we have to first convert the
        # natural parameters into the standard parameters.
        self.exp_np1 = T.matrix(dtype=theano.config.floatX)
        self.exp_np2 = T.matrix(dtype=theano.config.floatX)
        q_np1 = np1 + self.exp_np1
        q_np2 = np2 + self.exp_np2
        var = -1. / (2 * q_np1)
        mean = var * q_np2
        self.output = mean + T.sqrt(var) * self.eps

        # KL divergence (up to a constant) between the posterior and
        # the prior over the latent features.
        exp_x1 = (q_np2 ** 2) / (4 * (q_np1 ** 2)) - 1. / (2 * q_np1)
        exp_x2 = -q_np2 / (2 * q_np1)
        a_q = T.sum(-.5 * T.log(-2 * q_np1) - (q_np2 ** 2) / (
                              4 * q_np1), axis=1)
        a_p = T.sum(-.5 * T.log(-2 * self.exp_np1) - (self.exp_np2 ** 2) / (
                                4 * self.exp_np1), axis=1)
        self.kl_div = T.sum(np1 * exp_x1, axis=1)
        self.kl_div += T.sum(np2 * exp_x2, axis=1)
        self.kl_div += a_p - a_q


class SVAE(PersistentModel, AdamSGATheano):
    """Variational Auto-Encoder with GMM."""

    def __init__(self, dim_fea, dim_latent, n_layers, n_units,
                 activation='relu'):
        """Initialize a VAE.

        Parameters
        ----------
        dim_fea : int
            Dimension of the input features.
        dim_latent : int
            Dimension of the latent variable.
        n_layers : int
            Number of hidden layers.
        n_units : int
            Number of units per layer.
        activation : function
            Non-linear activation.

        """
        self.dim_fea = dim_fea
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation

        # The constructor does not include the prior model as it gives
        # more flexibility to do operation on the VAE structure or on
        # the prior separately.
        self.prior = None

        self._build()

    def _build(self):
        # Encoding network.
        self.encoder = MLPEncoderGMM(
            self.dim_fea,
            self.dim_latent,
            self.n_layers,
            self.n_units,
            self.activation,
            self.log_var_prior,
            self.non_informative_prior
        )

        # Decoding network.
        self.decoder = MLPDecoder(
            self.encoder,
            self.dim_fea,
            self.dim_latent,
            self.n_layers,
            self.n_units,
            self.activation
        )

        # Lower bound of the log-likelihood.
        llh = self.decoder.llh - self.encoder.kl_div
        objective = T.sum(llh)
        self.log_likelihood = theano.function(
            inputs=[
                self.encoder.input,
                self.encoder.exp_np1,
                self.encoder.exp_np2
            ],
            outputs=llh,
        )

        # Parameters to update.
        self.params = self.encoder.params + self.decoder.params

        # Gradient of cost function with respect to the parameters.
        gradients = [T.grad(objective, param) for param in self.params]

        # Specific gradients for the prior.
        grad_np1 = T.grad(objective, self.encoder.np1)
        grad_np2 = T.grad(objective, self.encoder.np2)

        inputs = [
            self.encoder.input,
            self.encoder.exp_np1,
            self.encoder.exp_np2
        ]
        outputs = [llh, T.concatenate([grad_np1, grad_np2], axis=1)]
        outputs += gradients

        self.get_gradients = theano.function(inputs=inputs, outputs=outputs)

        # ADAM SGA training.
        AdamSGATheano.__init__(
            self,
            [],
            self.params,
            gradients
        )

    def optimize_local_factors(self, data, n_iter=1):
        """Optimize the local factors q(x) and q(z).

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.
        n_iter : int
            Number of iterations for the optimization.

        Returns
        -------
        resps : numpy.ndarray
            Responsibilities for each component of the mixture.
        exp_np1 : numpy.ndarray
            First natural parameters of the optimal q(x).
        exp_np2 : numpy.ndarray
            Second natural parameters of the optimal q(x).
        s_stats : numpy.ndarray
            Sufficient statistics of the expected value of latent
            features: E[phi(x)].

        """
        dim_latent = self.dim_latent

        # Expected value of the prior's components parameters.
        p_np1 = [comp.posterior.grad_log_partition[:dim_latent]
                 for comp in self.prior.components]
        p_np2 = [comp.posterior.grad_log_partition[dim_latent:2 * dim_latent]
                 for comp in self.prior.components]

        # Get the output of the Gaussian encoder.
        np1, np2 = self.encoder.natural_params(data)
        nparams = np.hstack([np1, np2])

        # Initialization of the assignments.
        resps = self.prior.init_resps(len(data))

        # Padding value for the sufficient statistics.
        padding = np.ones((len(resps), dim_latent * 2))

        for i in range(n_iter):
            # Estimate the optimal parameters of q(X) given
            # the responsibilities.
            exp_np1 = resps.dot(p_np1)
            exp_np2 = resps.dot(p_np2)
            q_np1 = np1 + exp_np1
            q_np2 = np2 + exp_np2

            # Get the expected value sufficient stats: E_q(x)[phi(x)].
            exp_x1 = (q_np2 ** 2) / (4 * (q_np1 ** 2)) - 1. / (2 * q_np1)
            exp_x2 = -q_np2 / (2 * q_np1)

            # Re-estimate the responsibilities.
            s_stats = np.c_[exp_x1, exp_x2, padding]
            _, resps, model_data = self.prior.get_resps(s_stats)

        # Estimate the optimal parameters of q(X) given
        # the responsibilities.
        exp_np1 = resps.dot(p_np1)
        exp_np2 = resps.dot(p_np2)
        q_np1 = np1 + exp_np1
        q_np2 = np2 + exp_np2

        # Get the expected value sufficient stats: E_q(x)[phi(x)].
        exp_x1 = (q_np2 ** 2) / (4 * (q_np1 ** 2)) - 1. / (2 * q_np1)
        exp_x2 = -q_np2 / (2 * q_np1)
        s_stats = np.c_[exp_x1, exp_x2, padding]

        return resps, exp_np1, exp_np2, s_stats, model_data

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'dim_fea': self.dim_fea,
            'dim_latent': self.dim_latent,
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'activation': self.activation,
            'log_var_prior': self.log_var_prior,
            'non_informative_prior': self.non_informative_prior,
            'params': [param.get_value() for param in self.params]
        }

    @staticmethod
    def load_from_dict(model_data):
        model = SVAE.__new__(SVAE)

        model.dim_fea = model_data['dim_fea']
        model.dim_latent = model_data['dim_latent']
        model.n_layers = model_data['n_layers']
        model.n_units = model_data['n_units']
        model.activation = model_data['activation']
        model.log_var_prior = model_data['log_var_prior']
        model.non_informative_prior = model_data['non_informative_prior']

        model._build()
        for model_param, param in zip(model.params, model_data['params']):
            model_param.set_value(param)

        return model

    @staticmethod
    def load(file_obj):
        model_data = pickle.load(file_obj)
        return SVAE.load_from_dict(model_data)

    # -----------------------------------------------------------------

