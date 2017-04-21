
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
import autograd.numpy as np
from autograd import grad
from .model import PersistentModel
from .mlp_utils import init_weights_matrix
from .mlp_utils import init_bias
from .mlp_utils import relu


def _vae_forward(params, data):
    # Forward the input through the hidden layers.
    inputs = data
    for idx in range(0, len(params[:-4]), 2):
        weights = params[idx]
        bias = params[idx + 1]
        outputs = np.dot(inputs, weights) + bias
        inputs = relu(outputs)

    # Get the Gaussian parameters.
    mean = np.dot(inputs, params[-4]) + params[-3]
    logvar = np.dot(inputs, params[-2]) + params[-1]

    return mean, logvar


def _vae_encode(enc_params, noninf_prior, data):
    mean, logvar = _vae_forward(enc_params, data)

    # KL divergence between the posterior and the prior distribution
    # over the latent variable.
    if not noninf_prior:
        dim = len(enc_params[-1])
        #kl_div = .5 * np.sum(-1 - logvar + np.exp(logvar) + mean**2, axis=1)
        var_prior = np.ones(dim) * 1
        logvar_prior = np.log(var_prior)
        var = np.exp(logvar)
        kl_div = .5 * (logvar_prior - logvar)
        kl_div += .5 * (mean**2) / var_prior
        kl_div += .5 * ((var / var_prior) - 1)
        kl_div = np.sum(kl_div, axis=1)
    else:
        # In this case this is just the negative entropy of the
        # posterior.
        kl_div = -.5 * np.sum(logvar, axis=1)
        ###kl_div = 0.

    # Sample using the reparameterization trick.
    eps = np.random.randn(mean.shape[0], mean.shape[1])
    latent =  mean + np.exp(.5 * logvar) * eps

    return latent, kl_div


def _vae_decode(dec_params, latent, data):
    mean, logvar = _vae_forward(dec_params, latent)
    llh = np.sum(-.5 * (logvar + ((mean - data)**2) / np.exp(logvar)),
                 axis=1)
    return llh


def _vae_sample(dec_params, latent):
    mean, logvar = _vae_forward(dec_params, latent)
    eps = np.random.randn(mean.shape[0], mean.shape[1])
    return mean + np.exp(.5 * logvar) * eps


def _vae_elbo(params, noninf_prior, data, mse=False):
    # Separate encoder/decoder params.
    idx = len(params) // 2
    enc_params = params[:idx]
    dec_params = params[idx:]

    latent, kl_div = _vae_encode(enc_params, noninf_prior, data)
    llh = _vae_decode(dec_params, latent, data)

    if mse:
        return llh.sum()
    else:
        return np.sum(llh - kl_div)


def _svae_get_nparams(enc_params, data):
    mean, logvar = _vae_forward(enc_params, data)

    # Convert the Gaussian parameters to the natural parameters.
    np1 = - 1 / (2 * np.exp(logvar))
    np2 = mean / (np.exp(logvar))
    return np1, np2


def _svae_encode(enc_params, exp_np1, exp_np2, data):
    # Re-parameterization trick. The reparameterization is slightly
    # more complex in this case as we have to first convert the
    # natural parameters into the standard parameters.
    np1, np2 = _svae_get_nparams(enc_params, data)
    q_np1 = np1 + exp_np1
    q_np2 = np2 + exp_np2
    var = -1. / (2 * q_np1)
    mean = var * q_np2

    # Sample using the reparameterization trick.
    eps = np.random.randn(mean.shape[0], mean.shape[1])
    latent = mean + np.sqrt(var) * eps

    # KL divergence between the posterior and the prior over the latent
    # features.
    exp_x1 = (q_np2 ** 2) / (4 * (q_np1 ** 2)) - 1. / (2 * q_np1)
    exp_x2 = -q_np2 / (2 * q_np1)
    a_q = np.sum(-.5 * np.log(-2 * q_np1) - (q_np2 ** 2) / (
                          4 * q_np1), axis=1)
    a_p = np.sum(-.5 * np.log(-2 * exp_np1) - (exp_np2 ** 2) / (
                            4 * exp_np1), axis=1)
    kl_div = np.sum(np1 * exp_x1, axis=1)
    kl_div += np.sum(np2 * exp_x2, axis=1)
    kl_div += a_p - a_q

    return latent, kl_div


def _svae_elbo(params, exp_np1, exp_np2, data, mse=False):
    # Separate encoder/decoder params.
    idx = len(params) // 2
    enc_params = params[:idx]
    dec_params = params[idx:]

    latent, kl_div = _svae_encode(enc_params, exp_np1, exp_np2, data)
    llh = _vae_decode(dec_params, latent, data)

    if mse:
        return np.sum(llh)
    else:
        return np.sum(llh - kl_div)


def _svae_sample(dec_params, latent):
    mean, logvar = _vae_forward(dec_params, latent)
    eps = np.random.randn(mean.shape[0], mean.shape[1])
    return mean + np.exp(.5 * logvar) * eps


# Gradient of the cost function with respect to the parameters.
_vae_elbo_gradients = grad(_vae_elbo)
_svae_elbo_gradients = grad(_svae_elbo)


class VAE(PersistentModel):
    """Variational Auto-Encoder."""

    def __init__(self, mean, precision, dim_fea, dim_latent, n_layers, n_units,
                 non_informative_prior=False):
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

        """
        self.dim_fea = dim_fea
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_units = n_units
        self.non_informative_prior = non_informative_prior

        self._build()

    def _build(self):
        # Encoder.
        # Build the hidden layers.
        enc_params = [init_weights_matrix(self.dim_fea, self.n_units),
                       init_bias(self.n_units)]
        for n in range(self.n_layers - 1):
            weights = init_weights_matrix(self.n_units, self.n_units)
            bias = self.n_units
            enc_params += [weights, bias]

        # Create the Gaussian layer.
        mean_w = init_weights_matrix(self.n_units, self.dim_latent,
                                     scale=100.)
        mean_b = init_bias(self.dim_latent)
        logvar_w = init_weights_matrix(self.n_units, self.dim_latent)
        logvar_b = init_bias(self.dim_latent, shift=-3.)
        enc_params += [mean_w, mean_b]
        enc_params += [logvar_w, logvar_b]

        # Decoder.
        # Build the hidden layers.
        dec_params = [init_weights_matrix(self.dim_latent, self.n_units),
                       init_bias(self.n_units)]
        for n in range(self.n_layers - 1):
            weights = init_weights_matrix(self.n_units, self.n_units)
            bias = self.n_units
            dec_params += [weights, bias]

        # Create the Gaussian layer.
        mean_w = init_weights_matrix(self.n_units, self.dim_fea)
        mean_b = init_bias(self.dim_fea)
        logvar_w = init_weights_matrix(self.n_units, self.dim_fea)
        logvar_b = init_bias(self.dim_fea, shift=-5.)
        dec_params += [mean_w, mean_b]
        dec_params += [logvar_w, logvar_b]

        self.params = enc_params + dec_params

    def encode(self, data):
        idx = len(self.params) // 2
        enc_params = self.params[:idx]
        latent, _ = _vae_encode(enc_params, self.non_informative_prior, data)
        return latent

    def sample(self, data):
        idx = len(self.params) // 2
        enc_params = self.params[:idx]
        dec_params = self.params[idx:]
        latent, _ = _vae_encode(enc_params, self.non_informative_prior, data)
        rec_data = _vae_sample(dec_params, latent)
        return rec_data

    def log_likelihood(self, data):
        llh = _vae_elbo(self.params, self.non_informative_prior, data,
                         mse=False)
        return llh

    def get_gradients(self, data):
        return _vae_elbo_gradients(self.params, self.non_informative_prior,
                                   data)


    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):
        return {
            'dim_fea': self.dim_fea,
            'dim_latent': self.dim_latent,
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'non_informative_prior': self.non_informative_prior,
            'params': self.params
        }

    @staticmethod
    def load_from_dict(model_data):
        model = VAE.__new__(VAE)

        model.dim_fea = model_data['dim_fea']
        model.dim_latent = model_data['dim_latent']
        model.n_layers = model_data['n_layers']
        model.n_units = model_data['n_units']
        model.non_informative_prior = model_data['non_informative_prior']

        model._build()
        model.params = model_data['params']

        return model

    # -----------------------------------------------------------------


class SVAE(PersistentModel):
    """Variational Auto-Encoder with GMM."""

    def __init__(self, dim_fea, dim_latent, n_layers, n_units):
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

        """
        self.dim_fea = dim_fea
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_units = n_units

        # The constructor does not include the prior model as it gives
        # more flexibility to do operation on the VAE structure or on
        # the prior separately.
        self.prior = None

        self._build()

    def _build(self):
        # Encoder.
        # Build the hidden layers.
        enc_params = [init_weights_matrix(self.dim_fea, self.n_units),
                      init_bias(self.n_units)]
        for n in range(self.n_layers - 1):
            weights = init_weights_matrix(self.n_units, self.n_units)
            bias = self.n_units
            enc_params += [weights, bias]

        # Create the Gaussian layer.
        mean_w = init_weights_matrix(self.n_units, self.dim_latent,
                                     scale=100.)
        mean_b = init_bias(self.dim_latent)
        logvar_w = init_weights_matrix(self.n_units, self.dim_latent)
        logvar_b = init_bias(self.dim_latent, shift=-3)
        enc_params += [mean_w, mean_b]
        enc_params += [logvar_w, logvar_b]

        # Decoder.
        # Build the hidden layers.
        dec_params = [init_weights_matrix(self.dim_latent, self.n_units),
                      init_bias(self.n_units)]
        for n in range(self.n_layers - 1):
            weights = init_weights_matrix(self.n_units, self.n_units)
            bias = self.n_units
            dec_params += [weights, bias]

        # Create the Gaussian layer.
        mean_w = init_weights_matrix(self.n_units, self.dim_fea, scale=100)
        mean_b = init_bias(self.dim_fea)
        logvar_w = init_weights_matrix(self.n_units, self.dim_fea)
        logvar_b = init_bias(self.dim_fea, shift=-5.)
        dec_params += [mean_w, mean_b]
        dec_params += [logvar_w, logvar_b]

        self.params = enc_params + dec_params

    def encode(self, data, exp_np1, exp_np2):
        idx = len(self.params) // 2
        enc_params = self.params[:idx]
        latent, _ = _svae_encode(enc_params, exp_np1, exp_np2, data)
        return latent

    def sample(self, data, exp_np1, exp_np2):
        idx = len(self.params) // 2
        enc_params = self.params[:idx]
        dec_params = self.params[idx:]
        latent, _ = _svae_encode(enc_params, exp_np1, exp_np2, data)
        rec_data = _svae_sample(dec_params, latent)
        return rec_data

    def log_likelihood(self, data, exp_np1, exp_np2):
        return _svae_elbo(self.params, exp_np1, exp_np2, data)

    def get_gradients(self, data, exp_np1, exp_np2):
        return _svae_elbo_gradients(self.params, exp_np1, exp_np2, data)

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
        # Separate encoder/decoder params.
        idx = len(self.params) // 2
        enc_params = self.params[:idx]

        # Expected value of the prior's components parameters.
        p_np1 = [comp.posterior.grad_log_partition[:dim_latent]
                 for comp in self.prior.components]
        p_np2 = [comp.posterior.grad_log_partition[dim_latent:2 * dim_latent]
                 for comp in self.prior.components]

        # Get the output of the Gaussian encoder.
        np1, np2 = _svae_get_nparams(enc_params, data)
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
            'non_informative_prior': self.non_informative_prior,
            'params': [param for param in self.params]
        }

    @staticmethod
    def load_from_dict(model_data):
        model = SVAE.__new__(SVAE)

        model.dim_fea = model_data['dim_fea']
        model.dim_latent = model_data['dim_latent']
        model.n_layers = model_data['n_layers']
        model.n_units = model_data['n_units']
        model.non_informative_prior = model_data['non_informative_prior']

        model._build()
        model.params = model_data['params']

        return model

    @staticmethod
    def load(file_obj):
        model_data = pickle.load(file_obj)
        return SVAE.load_from_dict(model_data)

    # -----------------------------------------------------------------

