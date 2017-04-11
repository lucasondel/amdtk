
"""
Training algorithms vor various models.

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

import abc
import time
import numpy as np
import theano
import theano.tensor as T
from ipyparallel.util import interactive


class Inference(metaclass=abc.ABCMeta):
    """Base class for the training algorithms."""

    def __init__(self, dview, params, model):
        """Initialize the training algorithm.

        Parameters
        ----------
        dview : object
            Remote client objects to parallelize the training.
        params : dictionary
            Settings of the training. The settings are training
            dependent.
        model : object
            Model to train.

        """
        self.dview = dview
        self._epochs = int(params.get('epochs', 1))
        self._batch_size = int(params.get('batch_size', 1))
        
        with self.dview.sync_imports():
            import numpy 
            from amdtk import read_htk

    def run(self, data, callback):
        """Run the training.

        Parameters
        ----------
        model : object
            Model to train.
        data : numpy.ndarray
            Data to train on.
        callback : function
            Callback function to monitor the convergence of the
            training. The function takes a dictionary as input that
            contains the data about the current state of the training.
            Keys of the dictionary are:
              * iteration : current iteration (int)
              * batch : current mini-bacth index (int)
              * n_batch : total number of mini-batch (int)
              * time : elapsed time since the beginning of the training.
                       (float)
              * objective : current value of the objective function.

        """
        start_time = time.time()
        time_step = 0.

        for epoch in range(self._epochs):
            # Shuffle the data to avoid cycle in the training.
            np_data = np.array(data)
            idxs = np.arange(0, len(data))
            np.random.shuffle(idxs)
            shuffled_data = np_data[idxs]

            if self._batch_size < 0:
                batch_size = len(data)
            else:
                batch_size = self._batch_size

            for mini_batch in range(0, len(data), batch_size):
                # Index of the data mini-batch.
                start = mini_batch
                end = mini_batch + batch_size

                # Update the model.
                objective = self.train(shuffled_data[start:end], epoch + 1,
                                       mini_batch)

                # Monitor the convergence.
                callback({
                    'epoch': epoch + 1,
                    'batch': int(mini_batch / batch_size) + 1,
                    'n_batch': int(np.ceil(len(data) / batch_size)),
                    'time': time.time() - start_time,
                    'objective': objective
                })

    @abc.abstractmethod
    def train(self, data, epoch, mini_batch):
        """One update of the training process.

        Parameters
        ----------
        data : numpy.ndarray
            Data to train on.
        epoch : int
            Epoch of the training.
        mini_batch : int
            Mini-batch index for the current epoch.

        """
        pass


@interactive
def std_vb_e_step(fea_list):    
    """Jod for the standard E-step.
    
    Parameters
    ----------
    fea_list : list
        List of features file.

    Returns
    -------
    exp_llh : float
        Accumulated log-likelihood.
    acc_stats : object
        Accumulated sufficient statistics.

    """
    exp_llh = 0.
    acc_stats = None
    n_frames = 0
    for fea_file in fea_list:
        data = read_htk(fea_file)
        log_norm, new_acc_stats = model.vb_e_step(data)
        n_frames += len(data)
        exp_llh += numpy.sum(log_norm)
        if acc_stats is None:
            acc_stats = new_acc_stats
        else:
            acc_stats += new_acc_stats

    return (exp_llh, acc_stats, n_frames)


class StdVBInference(Inference):
    """Standard mean-field Variational Bayes training."""

    def __init__(self, dview, params, model):
        """Initialize the standard Variational Bayes training."""
        Inference.__init__(self, dview, params, model)
        self.model = model
        model.n_frames = int(params.get('n_frames', model.n_frames))

        # The standard VB training cannot be done on minibatches.
        self._batch_size = -1

    def train(self, fea_list, epoch, mini_batch):
        """One update of the training process.

        Parameters
        ----------
        model : model
            Model to train.
        fea_list : str
            List of features files.
        epoch : int
            Epoch of the training.
        mini_batch : int
            Mini-batch index for the current epoch.

        """
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Accumulate statistics.
        new_fea_list = [fea_list[i:i + len(self.dview)] 
                        for i in range(0, len(fea_list), len(self.dview))]
        stats_list = \
            self.dview.map_sync(std_vb_e_step, new_fea_list)

        exp_llh = stats_list[0][0]
        acc_stats = stats_list[0][1]
        for new_exp_llh, new_acc_stats, new_batch_n_frames in stats_list[1:]:
            exp_llh += new_exp_llh
            acc_stats += new_acc_stats 

        # Compute the lower-bound of the data given the model. Needs
        # to be done before we update the parameters.
        kl_div = self.model.kl_div_posterior_prior()
        lower_bound = exp_llh - kl_div

        # Update the parameters of the model.
        self.model.vb_m_step(acc_stats)

        return lower_bound / self.model.n_frames


class SGAVBInference(Inference):
    """Standard SGA training."""

    def __init__(self, dview, params, model):
        """Initialize the training."""
        Inference.__init__(self, dview, params, model)

        model.forgetting_rate = float(params.get('forgetting_rate',
                                                 model.forgetting_rate))
        model.delay = float(params.get('delay', model.delay))
        model.scale = float(params.get('scale', model.scale))
        model.n_frames = int(params.get('n_frames', model.n_frames))
        self._time_step = 0
        self.model = model

    def train(self, fea_list, epoch, mini_batch):
        """One update of the training process.

        Parameters
        ----------
        model : model
            Model to train.
        data : numpy.ndarray
            Data to train on.
        epoch : int
            Epoch of the training.
        mini_batch : int
            Mini-batch index for the current epoch.

        """
        self._time_step += 1

        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })
        
        # Compute the learning rate.
        lrate = self.model.scale * \
            ((self.model.delay + self._time_step)**(-self.model.forgetting_rate))

        # Accumulate statistics.
        new_fea_list = [fea_list[i:i + len(self.dview)] 
                        for i in range(0, len(fea_list), len(self.dview))]
        stats_list = \
            self.dview.map_sync(std_vb_e_step, new_fea_list)

        # Compute the gradients.
        print(len(stats_list))
        exp_llh = stats_list[0][0]
        acc_stats = stats_list[0][1]
        batch_n_frames = stats_list[0][2]
        for new_exp_llh, new_acc_stats, new_batch_n_frames in stats_list[1:]:
            exp_llh += new_exp_llh
            acc_stats += new_acc_stats 
            batch_n_frames += new_batch_n_frames

        # Scale the statistics.
        scale = self.model.n_frames / batch_n_frames 
        acc_stats *= scale
        
        # Estimate the lower bound.
        kl_div = self.model.kl_div_posterior_prior()
        elbo = (scale * exp_llh - kl_div) / self.model.n_frames

        # Update the parameters.
        grads = self.model.grads_from_acc_stats(acc_stats)
        for param, grad in zip(self.model.params, grads):
            param += lrate * grad

        # Post-processing.
        self.model.after_grad_update()

        return elbo


class AdamSGAInference(Inference):
    """Variational Auto-Encoder ADAM training."""

    def __init__(self, params, model):
        """Initialize the VAE Adam SGA training."""
        Inference.__init__(self, params, model)
        model.b1 = float(params.get('b1', model.b1))
        model.b2 = float(params.get('b2', model.b2))
        model.lrate = float(params.get('lrate', model.lrate))
        self.model = model

    def train(self, data, epoch, mini_batch):
        """One update of the training process.

        Parameters
        ----------
        model : model
            Model to train.
        data : numpy.ndarray
            Data to train on.
        epoch : int
            Epoch of the training.
        mini_batch : int
            Mini-batch index for the current epoch.

        """
        # Update the parameters of the VAE.
        lower_bound = self.model.adam_sga_update(data, epoch)

        return lower_bound


class SVAEStdSGAInference(Inference):
    """Structured Variational Auto-Encoder training."""

    def __init__(self, params, model):
        """Initialize the VAE SGA training."""
        Inference.__init__(self, params, model)
        model.prior.forgetting_rate = \
            float(params.get('prior_forgetting_rate',
                             model.prior.forgetting_rate))
        model.prior.delay = float(params.get('prior_delay', model.prior.delay))
        model.prior.scale = float(params.get('prior_scale', model.prior.scale))
        model.prior.n_frames = int(params.get('prior_n_frames',
                                              model.prior.n_frames))
        model.forgetting_rate = float(params.get('forgetting_rate',
                                                 model.forgetting_rate))
        model.delay = float(params.get('delay', model.delay))
        model.scale = float(params.get('scale', model.scale))
        model.n_frames = int(params.get('n_frames', model.n_frames))
        self._n_iter = int(params.get('n_iter', 1))
        self._time_step = 0
        self.model = model

    def train(self, data, epoch, mini_batch):
        """One update of the training process.

        Parameters
        ----------
        model : model
            Model to train.
        data : numpy.ndarray
            Data to train on.
        epoch : int
            Epoch of the training.
        mini_batch : int
            Mini-batch index for the current epoch.

        """
        self._time_step += 1

        # Optimize the log factors q(Z) and q(X)
        resps, exp_np1, exp_np2, s_stats = \
            self.model.optimize_local_factors(data, self._n_iter)

        # Update the VAE.
        lower_bound, corr_grad = self.model.std_sga_update(
            data,
            exp_np1,
            exp_np2,
            self._time_step
        )

        # Correct the expected value of the sufficient statistics.
        padding = s_stats.shape[1] - corr_grad.shape[1]
        s_stats = s_stats + np.c_[corr_grad, np.zeros((len(data), padding))]

        # Accumulate the statistics for the latent model.
        acc_s_stats = self.model.prior.accumulate_stats(s_stats, resps)

        # Scale the sufficient statistics.
        n_frames = self.model.prior.n_frames
        scale = self.model.prior.n_frames / len(resps)
        acc_s_stats *= scale

        vae_prior_kl = self.model.prior.kl_div_posterior_prior()

        # Compute the gradients.
        grads = self.model.prior.grads_from_acc_stats(acc_s_stats)

        lower_bound -= vae_prior_kl / n_frames

        # Update the VAE's prior.
        self.model.prior.std_sga_update_from_grads(grads, self._time_step)

        return lower_bound


class SVAEAdamSGAInference(Inference):
    """Structured Variational Auto-Encoder training."""

    def __init__(self, params, model):
        """Initialize the VAE SGA training."""
        Inference.__init__(self, params, model)
        model.prior.forgetting_rate = \
            float(params.get('forgetting_rate', model.prior.forgetting_rate))
        model.prior.delay = float(params.get('delay', model.prior.delay))
        model.prior.scale = float(params.get('scale', model.prior.scale))
        model.prior.n_frames = int(params.get('n_frames', model.prior.n_frames))
        model.b1 = float(params.get('b1', model.b1))
        model.b2 = float(params.get('b2', model.b2))
        model.lrate = float(params.get('lrate', model.lrate))
        self._n_iter = int(params.get('n_iter', 1))
        self._time_step = 0
        self.model = model

    def train(self, data, epoch, mini_batch):
        """One update of the training process.

        Parameters
        ----------
        model : model
            Model to train.
        data : numpy.ndarray
            Data to train on.
        epoch : int
            Epoch of the training.
        mini_batch : int
            Mini-batch index for the current epoch.

        """
        self._time_step += 1

        # Optimize the log factors q(Z) and q(X)
        resps, exp_np1, exp_np2, s_stats, model_data = \
            self.model.optimize_local_factors(data, self._n_iter)

        # Update the VAE.
        lower_bound, corr_grad = self.model.adam_sga_update(
            data,
            exp_np1,
            exp_np2,
            epoch
        )

        # Correct the expected value of the sufficient statistics.
        padding = s_stats.shape[1] - corr_grad.shape[1]
        s_stats = s_stats + np.c_[corr_grad, np.zeros((len(data), padding))]

        # Accumulate the statistics for the latent model.
        acc_s_stats = self.model.prior.accumulate_stats(s_stats, resps,
                                                        model_data)

        # Scale the sufficient statistics.
        n_frames = self.model.prior.n_frames
        scale = self.model.prior.n_frames / len(resps)
        acc_s_stats *= scale

        vae_prior_kl = self.model.prior.kl_div_posterior_prior()

        # Compute the gradients.
        grads = self.model.prior.grads_from_acc_stats(acc_s_stats)

        lower_bound -= vae_prior_kl / n_frames

        # Update the VAE's prior.
        self.model.prior.std_sga_update_from_grads(grads, self._time_step)

        return lower_bound

