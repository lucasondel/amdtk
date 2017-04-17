
"""
Training algorithms vor various models.

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
import time
import numpy as np
from ipyparallel.util import interactive


class Inference(metaclass=abc.ABCMeta):
    """Base class for the training algorithms."""

    def __init__(self, dview, data_stats, params, model):
        """Initialize the training algorithm.

        Parameters
        ----------
        dview : object
            Remote client objects to parallelize the training.
        data_stats : object
            Statistics of the training data.
        params : dictionary
            Settings of the training. The settings are training
            dependent.
        model : object
            Model to train.

        """
        self.dview = dview
        self.epochs = int(params.get('epochs', 1))
        self.batch_size = int(params.get('batch_size', 1))
        self.model = model
        self.time_step = 0
        self.data_stats = data_stats

        with self.dview.sync_imports():
            import numpy
            from amdtk import read_htk

        self.dview.push({
            'data_stats': self.data_stats,
        })

    def run(self, data, callback):
        """Train the model.

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
              * time : elapsed time since the beginning of the
                       training (float).
              * objective : current value of the objective function.

        """
        start_time = time.time()

        for epoch in range(self.epochs):
            # Shuffle the data to avoid cycle in the training.
            np_data = np.array(data)
            idxs = np.arange(0, len(data))
            np.random.shuffle(idxs)
            shuffled_data = np_data[idxs]

            if self.batch_size < 0:
                batch_size = len(data)
            else:
                batch_size = self.batch_size

            for mini_batch in range(0, len(data), batch_size):
                self.time_step += 1

                # Index of the data mini-batch.
                start = mini_batch
                end = mini_batch + batch_size

                # Reshaped the list of features.
                fea_list = shuffled_data[start:end]
                new_fea_list = [fea_list[i:i + len(self.dview)]  for i in
                                range(0, len(fea_list), len(self.dview))]

                # Update the model.
                objective, kl_div = self.train(new_fea_list, epoch + 1,
                                               self.time_step)

                # Monitor the convergence.
                callback({
                    'epoch': epoch + 1,
                    'batch': int(mini_batch / batch_size) + 1,
                    'n_batch': int(np.ceil(len(data) / batch_size)),
                    'time': time.time() - start_time,
                    'objective': objective,
                    'kl_div': kl_div
                })

    @abc.abstractmethod
    def train(self, data, epoch, time_step):
        """One update of the training process.

        Parameters
        ----------
        data : numpy.ndarray
            Data to train on.
        epoch : int
            Epoch of the training.
        time_step : int
            Number of the training update.

        """
        pass


class StdVBInference(Inference):
    """Standard mean-field Variational Bayes training."""

    @staticmethod
    @interactive
    def e_step(fea_list):
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
        n_frames : int
            Number of frames in the batch.

        """
        # Initialize the returned values.
        exp_llh = 0.
        acc_stats = None
        n_frames = 0

        # For each features file...
        for fea_file in fea_list:
            # Load the features.
            data = read_htk(fea_file)

            # Mean/Variance normalization.
            var = 1. / data_stats['precision']
            data -= data_stats['mean']
            data /= numpy.sqrt(var)

            # Get the accumulated sufficient statistics for the
            # given set of features.
            log_norm, new_acc_stats = model.vb_e_step(data)

            # Global accumulators.
            n_frames += len(data)
            exp_llh += numpy.sum(log_norm)
            if acc_stats is None:
                acc_stats = new_acc_stats
            else:
                acc_stats += new_acc_stats

        return (exp_llh, acc_stats, n_frames)

    def __init__(self, dview, data_stats, params, model):
        Inference.__init__(self, dview, data_stats, params, model)
        self.model = model

        # The standard VB training cannot be done on minibatches.
        self.batch_size = -1

    def train(self, fea_list, epoch, time_step):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel accumulation of the sufficient statistics.
        stats_list = self.dview.map_sync(StdVBInference.e_step, fea_list)

        # Accumulate the results from all the jobs.
        exp_llh = stats_list[0][0]
        acc_stats = stats_list[0][1]
        for new_exp_llh, new_acc_stats, new_batch_n_frames in stats_list[1:]:
            exp_llh += new_exp_llh
            acc_stats += new_acc_stats

        # Compute the lower-bound of the data given the model. Needs
        # to be done before we update the parameters.
        kl_div = self.model.kl_div_posterior_prior()
        lower_bound = exp_llh

        # Update the parameters of the model.
        self.model.vb_m_step(acc_stats)

        return lower_bound / self.data_stats['count'], kl_div


class StochasticVBInference(Inference):
    """Stochastic VB training."""

    def __init__(self, dview, data_stats, params, model):
        Inference.__init__(self, dview, data_stats, params, model)
        self.forgetting_rate = float(params.get('forgetting_rate', .51))
        self.delay = float(params.get('delay', 0.))
        self.scale = float(params.get('scale', 1.))

    def train(self, fea_list, epoch, time_step):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel accumulation of the sufficient statistics.
        stats_list = self.dview.map_sync(StdVBInference.e_step, fea_list)

        # Accumulate the results from all the jobs.
        exp_llh = stats_list[0][0]
        acc_stats = stats_list[0][1]
        batch_n_frames = stats_list[0][2]
        for new_exp_llh, new_acc_stats, new_batch_n_frames in stats_list[1:]:
            exp_llh += new_exp_llh
            acc_stats += new_acc_stats
            batch_n_frames += new_batch_n_frames

        # Scale the statistics.
        scale = self.data_stats['count'] / batch_n_frames
        acc_stats *= scale

        # Compute the learning rate.
        lrate = self.scale * \
            ((self.delay + time_step)**(-self.forgetting_rate))

        # Estimate the lower bound.
        kl_div = self.model.kl_div_posterior_prior()
        elbo = (scale * exp_llh) / self.data_stats['count']

        # Update the parameters.
        self.model.natural_grad_update(acc_stats, lrate)

        return elbo, kl_div


class AdamSGAInference(Inference):
    """Variational Auto-Encoder ADAM training."""

    @staticmethod
    @interactive
    def gradients(fea_list):
        """Jod to compute the gradients.

        Parameters
        ----------
        fea_list : list
            List of features file.

        Returns
        -------
        exp_llh : float
            Accumulated log-likelihood.
        gradients : object
            Gradients of the objective function.
        n_frames : int
            Number of frames in the batch.

        """
        # Initialize the returned values.
        exp_llh = 0.
        gradients = None
        n_frames = 0

        # For each features file...
        for fea_file in fea_list:
            # Load the features.
            data = read_htk(fea_file)

            # Mean/Variance normalization.
            var = 1. / data_stats['precision']
            data -= data_stats['mean']
            data /= numpy.sqrt(var)

            # Get the accumulated sufficient statistics for the
            # given set of features.
            res = model.get_gradients(data)
            llh = res[0]
            new_gradients = res[1:]

            # Global accumulators.
            n_frames += len(data)
            exp_llh += numpy.sum(llh)
            if gradients is None:
                gradients = new_gradients
            else:
                for grad1, grad2 in zip(gradients, new_gradients):
                    grad1[0] += grad2[0]

        return (exp_llh, gradients, n_frames)

    def __init__(self, dview, data_stats, params, model):
        Inference.__init__(self, dview, data_stats, params, model)
        self.forgetting_rate = float(params.get('forgetting_rate', .51))
        self.delay = float(params.get('delay', 0.))
        self.scale = float(params.get('scale', 1.))
        self.b1 = float(params.get('b1', .95))
        self.b2 = float(params.get('b2', .999))
        self.lrate = float(params.get('lrate', .01))
        self.model = model

    def train(self, fea_list, epoch, time_step):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel computation of the gradients.
        res_list = self.dview.map_sync(AdamSGAInference.gradients, fea_list)

        # Total number of frame of the mini-batch.
        exp_llh = 0.
        total_n_frames = 0.
        grads = None
        for new_exp_llh, new_grads, n_frames in res_list:
            exp_llh += new_exp_llh
            total_n_frames += n_frames
            if grads is None:
                grads = new_grads
            else:
                for grad1, grad2 in zip(grads, new_grads):
                    grad1[0] += grad2[0]

        # Rescale the gradients.
        for grad in grads:
            grad /= total_n_frames

        # Compute the learning rate.
        lrate = self.scale * \
            ((self.delay + time_step)**(-self.forgetting_rate))

        # Update the parameters of the VAE.
        self.model.adam_sga_update(
            *grads,
            self.b1,
            self.b2,
            self.lrate,
            time_step
        )

        return exp_llh / total_n_frames, 0.


class SVAEAdamSGAInference(Inference):
    """Structured Variational Auto-Encoder training."""

    @staticmethod
    @interactive
    def gradients(fea_list):
        """Jod to compute the gradients.

        Parameters
        ----------
        fea_list : list
            List of features file.

        Returns
        -------
        exp_llh : float
            Accumulated log-likelihood.
        gradients : object
            Gradients of the objective function.
        n_frames : int
            Number of frames in the batch.

        """
        # Initialize the returned values.
        exp_llh = 0.
        acc_stats = None
        gradients = None
        corr_grad = None
        n_frames = 0

        # For each features file...
        for fea_file in fea_list:
            # Load the features.
            data = read_htk(fea_file)

            # Mean/Variance normalization.
            var = 1. / data_stats['precision']
            data -= data_stats['mean']
            data /= numpy.sqrt(var)

            # Optimize the log factors q(Z) and q(X)
            resps, exp_np1, exp_np2, s_stats, model_data = \
                model.optimize_local_factors(data, 10)

            # Get the gradients.
            res = model.get_gradients(data, exp_np1, exp_np2)
            llh = res[0]
            new_corr_grad = res[1]
            new_gradients = res[2:]

            # Correct the expected value of the sufficient statistics.
            #padding = s_stats.shape[1] - corr_grad.shape[1]
            #s_stats = s_stats + numpy.c_[corr_grad, numpy.zeros((len(data), padding))]

            # Accumulate the statistics for the latent model.
            new_acc_stats = model.prior.accumulate_stats(s_stats, resps,
                                                         model_data)

            # Global accumulators.
            n_frames += len(data)
            exp_llh += numpy.sum(llh)
            if gradients is None:
                acc_stats = new_acc_stats
                gradients = new_gradients
                corr_grad = new_corr_grad.sum(axis=0)
            else:
                acc_stats += new_acc_stats
                for grad1, grad2 in zip(gradients, new_gradients):
                    grad1[0] += grad2[0]
                corr_grad += new_corr_grad.sum(axis=0)

        return (exp_llh, gradients, acc_stats, corr_grad, n_frames)

    def __init__(self, dview, data_stats, params, model):
        Inference.__init__(self, dview, data_stats, params, model)
        self.forgetting_rate = float(params.get('forgetting_rate', .51))
        self.delay = float(params.get('delay', 0.))
        self.scale = float(params.get('scale', 1.))
        self.b1 = float(params.get('b1', .95))
        self.b2 = float(params.get('b2', .999))
        self.lrate = float(params.get('lrate', .01))
        self.model = model

        print(self.forgetting_rate, self.delay, self.scale)

    def train(self, fea_list, epoch, time_step):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel computation of the gradients.
        res_list = self.dview.map_sync(SVAEAdamSGAInference.gradients, fea_list)

        # Total number of frame of the mini-batch.
        exp_llh = 0.
        total_n_frames = 0.
        acc_stats = None
        grads = None
        corr_grad = None
        for new_exp_llh, new_grads, new_acc_stats, new_corr_grad, n_frames in res_list:
            exp_llh += new_exp_llh
            total_n_frames += n_frames
            if grads is None:
                acc_stats = new_acc_stats
                grads = new_grads
                corr_grad = new_corr_grad
            else:
                acc_stats += new_acc_stats
                for grad1, grad2 in zip(grads, new_grads):
                    grad1[0] += grad2[0]
                corr_grad += new_corr_grad


        # Correct the expected value of the sufficient statistics.
        #padding = acc_stats[1].shape[1] - corr_grad.shape[0]
        #corr_grad = corr_grad
        #acc_stats[1] += np.r_[corr_grad, np.zeros(padding)]
        #print(np.linalg.norm(corr_grad))

        # Scale the statistics.
        scale = self.data_stats['count'] / total_n_frames
        acc_stats *= scale

        # Compute the learning rate.
        lrate = self.scale * \
            ((self.delay + time_step)**(-self.forgetting_rate))

        # Estimate the lower-bound.
        kl_div = self.model.prior.kl_div_posterior_prior()
        elbo = exp_llh - kl_div
        elbo /= total_n_frames


        # Update the parameters.
        self.model.prior.natural_grad_update(acc_stats, lrate)

        # Rescale the gradients.
        for grad in grads:
            grad /= total_n_frames

        # Update the parameters of the VAE.
        self.model.adam_sga_update(
            *grads,
            self.b1,
            self.b2,
            self.lrate,
            time_step
        )

        return elbo, kl_div

