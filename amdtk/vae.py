
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


def _vae_elbo(params, activation, data, prior_params):
    idx = len(params) // 2
    enc_params, dec_params = params[:idx], params[idx:]
    latent, kl_div = GaussianResidualMLP.sample(enc_params, activation, data,
                                                prior_params)
    llh = GaussianResidualMLP.llh(dec_params, activation, latent, data)
    return np.sum(llh - kl_div)


_vae_elbo_gradients = value_and_grad(_vae_elbo)


def _svae_elbo(params, activation, prior, exp_np1, exp_np2, num_points,
               num_batches, data):
    idx = len(params) // 2
    enc_params, dec_params = params[:idx], params[idx:]
    latent, kl_div = GaussianResidualMLP.sample_np(enc_params, activation,
                                                   data, exp_np1, exp_np2)
    llh = GaussianResidualMLP.llh(dec_params, activation, latent, data)
    kl_div_glob = prior.kl_div_posterior_prior()
    return (num_batches * np.sum(llh - kl_div) - kl_div_glob) / num_points


_svae_elbo_gradients = value_and_grad(_svae_elbo)

def _pvae_elbo(params, activation, prior, q_np1, q_np2, num_points, num_batches, data):
    idx = len(params) // 2
    enc_params, dec_params = params[:idx], params[idx:]

    prior_mean, prior_var = GaussianResidualMLP.std_params(q_np1, q_np2)
    latent, kl_div = GaussianResidualMLP.sample(enc_params, activation, data,
                                                (prior_mean, prior_var))

    kl_div_glob = prior.kl_div_posterior_prior()

    #latent, kl_div = GaussianResidualMLP.sample(enc_params, activation, data,
    #                                            prior_params)
    llh = GaussianResidualMLP.llh(dec_params, activation, latent, data)
    return (num_batches * np.sum(llh - kl_div) - kl_div_glob) / num_points


_pvae_elbo_gradients = value_and_grad(_pvae_elbo)


class VAE(PersistentModel):
    """Variational Auto-Encoder."""

    def __init__(self, args):
        self.dim_fea = int(args['dim_fea'])
        self.dim_latent = int(args['dim_latent'])
        self.dim_h = int(args['dim_h'])
        self.n_layers = int(args.get('n_layers', 1))
        self.scale = float(args.get('scale', 1e-2))
        self.mean_prior = float(args.get('mean_prior', 0.))
        self.var_prior = float(args.get('var_prior', 1.))
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
        )

        dec_params = GaussianResidualMLP.create(
            self.dim_latent,
            self.dim_fea,
            self.dim_h,
            self.n_layers,
            self.scale,
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

    def check_grad(self, data):
        prior_params = (self.mean_prior, self.var_prior)
        return quick_grad_check(_vae_elbo, self.params, (np.tanh, data, prior_params))

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

        model._build()
        model.params = model_data['params']

        return model

    # -----------------------------------------------------------------


class SVAE(VAE):
    """Variational Auto-Encoder with structured prior."""

    @staticmethod
    def _exp_stats(p_np1, p_np2, np1, np2, resps, padding):
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

        return exp_np1, exp_np2, s_stats

    def __init__(self, args):
        VAE.__init__(self, args)

        self._build()

    def _estimate_prior_np(self, prior, data):
        idx = len(self.params) // 2
        enc_params, dec_params = self.params[:idx], self.params[idx:]
        mean, var = GaussianResidualMLP.forward(enc_params, np.tanh,
                                                data)
        np1, np2 = GaussianResidualMLP.natural_params(mean, var)
        resps, exp_np1, exp_np2, s_stats, model_data = \
            self.optimize_local_factors(prior, np1, np2)
        q_np1 = np1 + exp_np1
        q_np2 = np2 + exp_np2

        return exp_np1, exp_np2, q_np1, q_np2, s_stats, resps, model_data

    def sample_latent(self, prior, data):
        #idx = len(self.params) // 2
        #enc_params = self.params[:idx]
        #exp_np1, exp_np2, _, _, _, _, _ = self._estimate_prior_np(prior, data)
        #samples, _ = GaussianResidualMLP.sample_np(enc_params, np.tanh, data,
        #                                           exp_np1, exp_np2)
        #return samples
        idx = len(self.params) // 2
        enc_params = self.params[:idx]
        return GaussianResidualMLP.sample(enc_params, np.tanh, data)

    def sample(self, prior, data):
        #idx = len(self.params) // 2
        #dec_params = self.params[idx:]
        #latent = self.sample_latent(prior, data)
        #return GaussianResidualMLP.sample(dec_params, np.tanh, latent)
        idx = len(self.params) // 2
        dec_params = self.params[idx:]
        latent = self.sample_latent(prior, data)
        return GaussianResidualMLP.sample(dec_params, np.tanh, latent)

    def sample2(self, data):
        idx = len(self.params) // 2
        dec_params = self.params[idx:]
        latent = self.sample_latent(data)
        return GaussianResidualMLP.sample(dec_params, np.tanh, latent)

    def get_gradients2(self, prior, data, num_points, num_batches, resps=None):
        params = self.params
        idx = len(params) // 2
        enc_params, dec_params = params[:idx], params[idx:]
        mean, var = GaussianResidualMLP.forward(enc_params, np.tanh, data)

        # Clustering.
        s_stats = prior.get_sufficient_stats(mean)
        if resps is None:
            log_norm, resps, model_data = prior.get_resps(getval(s_stats))
        else:
            log_norm, _, model_data = prior.get_resps(getval(s_stats))


        # Expected value of the prior's components parameters.
        dim_latent = self.dim_latent
        p_np1 = [comp.posterior.grad_log_partition[:dim_latent]
                 for comp in prior.components]
        p_np2 = [comp.posterior.grad_log_partition[dim_latent:2 * dim_latent]
                 for comp in prior.components]
        q_np1 = resps.dot(p_np1)
        q_np2 = resps.dot(p_np2)

        val, grads = _pvae_elbo_gradients(params, np.tanh, prior, q_np1, q_np2, num_points,
                                          num_batches, data)
        return val, grads, prior.accumulate_stats(s_stats, resps, model_data)

    def get_gradients(self, prior, data, num_points, num_batches):
        exp_np1, exp_np2, q_np1, q_np2, s_stats, resps, model_data = \
            self._estimate_prior_np(prior, data)
        params = self.params
        val, grads = _svae_elbo_gradients(params, np.tanh, prior, exp_np1,
                                          exp_np2, num_points, num_batches, data)
        #corr_grad = np.c_[grads[-2], grads[-1]]

        #padding = s_stats.shape[1] - corr_grad.shape[1]
        #s_stats += np.c_[corr_grad, np.zeros((len(data), padding))]
        acc_stats = prior.accumulate_stats(s_stats, resps, model_data)

        return val, grads, acc_stats

    def optimize_local_factors(self, prior, np1, np2, n_iter=100):
        dim_latent = self.dim_latent

        # Expected value of the prior's components parameters.
        p_np1 = [comp.posterior.grad_log_partition[:dim_latent]
                 for comp in prior.components]
        p_np2 = [comp.posterior.grad_log_partition[dim_latent:2 * dim_latent]
                 for comp in prior.components]

        # Initialization of the assignments.
        resps = prior.init_resps(len(np1))

        # Padding value for the sufficient statistics.
        padding = np.ones((len(resps), dim_latent * 2))

        tol = 1e-3
        old_stats = None
        for i in range(n_iter):
            exp_np1, exp_np2, s_stats = \
                SVAE._exp_stats(p_np1, p_np2, np1, np2, resps, padding)
            _, resps, model_data = prior.get_resps(s_stats)
            #_, _, model_data = prior.get_resps(s_stats)

            if old_stats is None:
                old_stats = s_stats
            else:
                if np.linalg.norm(old_stats - s_stats) <= tol:
                    break
                old_stats = s_stats

        if i == n_iter - 1:
            print('WARNING: reached maximum number of iterations.')

        exp_np1, exp_np2, s_stats = \
            SVAE._exp_stats(p_np1, p_np2, np1, np2, resps, padding)

        return resps, exp_np1, exp_np2, s_stats, model_data

    # PersistentModel interface implementation.
    # -----------------------------------------------------------------

    @staticmethod
    def load(file_obj):
        model_data = pickle.load(file_obj)
        model_data['model_class'] = SVAE
        return VAE.load_from_dict(model_data)

    # -----------------------------------------------------------------

