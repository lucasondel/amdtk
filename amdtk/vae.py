
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
from autograd import value_and_grad
from autograd.core import getval
from autograd.util import quick_grad_check
from .model import PersistentModel
from .mlp_utils import GaussianResidualMLP


def _svae_elbo(params, activation, q_np1, q_np2, data):
    idx = len(params) // 2
    enc_params, dec_params = params[:idx], params[idx:]
    prior_mean, prior_var = GaussianResidualMLP.std_params(q_np1, q_np2)
    latent, kl_div = GaussianResidualMLP.sample(enc_params, activation, data,
                                                (prior_mean, prior_var))
    llh = GaussianResidualMLP.llh(dec_params, activation, latent, data)
    return np.sum(llh - kl_div)


_svae_elbo_gradients = value_and_grad(_svae_elbo)


class VAE(PersistentModel):
    """Variational Auto-Encoder."""

    def __init__(self, args, precision):
        self.dim_fea = int(args['dim_fea'])
        self.dim_latent = int(args['dim_latent'])
        self.dim_h = int(args['dim_h'])
        self.n_layers = int(args.get('n_layers', 1))
        self.scale = float(args.get('scale', 1e-2))
        self.mean_prior = float(args.get('mean_prior', 0.))
        self.var_prior = float(args.get('var_prior', 1.))
        self.precision = precision
        self.params = None

        self._build()

    def _build(self):
        self.mean_prior = self.mean_prior * np.ones(self.dim_latent)
        self.var_prior = self.var_prior * np.ones(self.dim_latent)

        enc_params = GaussianResidualMLP.create(
            self.dim_fea,
            self.dim_latent,
            self.dim_h,
            self.n_layers,
            self.scale,
            self.precision
        )

        dec_params = GaussianResidualMLP.create(
            self.dim_latent,
            self.dim_fea,
            self.dim_h,
            self.n_layers,
            self.scale,
            self.precision
        )

        self.params = enc_params + dec_params

    def sample_latent(self, data):
        idx = len(self.params) // 2
        enc_params = self.params[:idx]
        return GaussianResidualMLP.sample(enc_params, np.tanh, data)

    def sample(self, data):
        idx = len(self.params) // 2
        dec_params = self.params[idx:]
        latent = self.sample_latent(data)
        return GaussianResidualMLP.sample(dec_params, np.tanh, latent)

    def get_gradients(self, data):
        prior_params = (self.mean_prior, self.var_prior)
        return _vae_elbo_gradients(self.params, np.tanh, data, prior_params)


    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    def to_dict(self):

        return {
            'dim_fea': self.dim_fea,
            'dim_latent': self.dim_latent,
            'dim_h': self.dim_h,
            'n_layers': self.n_layers,
            'scale': self.scale,
            'mean_prior': self.mean_prior[0],
            'var_prior': self.var_prior[0],
            'self.precision': self.precision,
            'params': self.params,
            'model_class': self.__class__
        }

    @staticmethod
    def load_from_dict(model_data):
        model_cls = model_data['model_class']
        model = model_cls.__new__(model_cls)

        model.dim_fea = int(model_data['dim_fea'])
        model.dim_latent = int(model_data['dim_latent'])
        model.dim_h = int(model_data['dim_h'])
        model.n_layers = int(model_data['n_layers'])
        model.scale = float(model_data['scale'])
        model.mean_prior = float(model_data['mean_prior'])
        model.var_prior = float(model_data['var_prior'])
        model.precision = model_data['precision']

        model._build()
        model.params = model_data['params']

        return model

    # -----------------------------------------------------------------


class SVAE(VAE):
    """Variational Auto-Encoder with structured prior."""

    def __init__(self, args, precision):
        VAE.__init__(self, args, precision)

        self._build()


    def sample_latent(self, data):
        idx = len(self.params) // 2
        enc_params = self.params[:idx]
        return GaussianResidualMLP.sample(enc_params, np.tanh, data)

    def sample(self, data):
        idx = len(self.params) // 2
        dec_params = self.params[idx:]
        latent = self.sample_latent(prior, data)
        return GaussianResidualMLP.sample(dec_params, np.tanh, latent)

    def decode(self, prior, data, state_path=False):
        params = self.params
        idx = len(params) // 2
        enc_params, dec_params = params[:idx], params[idx:]
        mean, var = GaussianResidualMLP.forward(enc_params, np.tanh, data)
        return prior.decode(mean, state_path)

    def get_gradients(self, prior, data, log_resps=None):
        params = self.params
        idx = len(params) // 2
        enc_params, dec_params = params[:idx], params[idx:]
        mean, var = GaussianResidualMLP.forward(enc_params, np.tanh, data)

        # Clustering.
        s_stats = prior.get_sufficient_stats(mean)
        log_norm, resps, acc_stats = prior.get_resps(s_stats, log_resps)

        # Expected value of the prior's components parameters.
        dim_latent = self.dim_latent
        p_np1 = [comp.posterior.grad_log_partition[:dim_latent]
                 for comp in prior.components]
        p_np2 = [comp.posterior.grad_log_partition[dim_latent:2 * dim_latent]
                 for comp in prior.components]
        q_np1 = resps[0].T.dot(p_np1)
        q_np2 = resps[0].T.dot(p_np2)

        val, grads = _svae_elbo_gradients(params, np.tanh, q_np1, q_np2, data)
        return val, grads, acc_stats

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    @staticmethod
    def load(file_obj):
        model_data = pickle.load(file_obj)
        model_data['model_class'] = SVAE
        return VAE.load_from_dict(model_data)

    # -----------------------------------------------------------------

