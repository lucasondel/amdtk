
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
        self.valid_freq = int(params.get('valid_freq', 1))
        self.model = model
        self.time_step = 0
        self.data_stats = data_stats

        with self.dview.sync_imports():
            import numpy
            from amdtk import read_htk

        self.dview.push({
            'data_stats': self.data_stats,
        })

    def run(self, data, valid_data, callback):
        """Train the model.

        Parameters
        ----------
        model : object
            Model to train.
        data : list
            Data to train on.
        valid_data : list
            Validation data.
        valid_freq : int
            Frequency of the validation.
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
                n_utts = batch_size // len(self.dview)
                new_fea_list = [fea_list[i:i + n_utts]  for i in
                                range(0, len(fea_list), n_utts)]

                # Update the model.
                self.train(new_fea_list, epoch + 1, self.time_step)

                if self.time_step % self.valid_freq == 0:
                    fea_list = valid_data
                    new_fea_list = [fea_list[i:i + len(self.dview)]  for i in
                                    range(0, len(fea_list), len(self.dview))]
                    objective = self.validate(new_fea_list)

                    # Monitor the convergence.
                    callback({
                        'epoch': epoch + 1,
                        'batch': int(mini_batch / batch_size) + 1,
                        'n_batch': int(np.ceil(len(data) / batch_size)),
                        'time': time.time() - start_time,
                        'objective': objective,
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

    @abc.abstractmethod
    def validate(self, data):
        """Compute the objective function on a validation set.

        Parameters
        ----------
        data : numpy.ndarray
            Validation data.

        """
        pass


class StochasticVBInference(Inference):
    """Stochastic VB training."""

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
        acc_stats : object
            Accumulated sufficient statistics.
        n_frames : int
            Number of frames in the batch.

        """
        # Initialize the returned values.
        acc_stats = None
        n_frames = 0

        # For each features file...
        for fea_file in fea_list:
            # Load the features.
            data = read_htk(fea_file)

            # Mean/Variance normalization.
            #var = 1. / data_stats['precision']
            #data -= data_stats['mean']
            #data /= numpy.sqrt(var)

            # Get the accumulated sufficient statistics for the
            # given set of features.
            _, new_acc_stats = model.vb_e_step(data)

            # Global accumulators.
            n_frames += len(data)
            if acc_stats is None:
                acc_stats = new_acc_stats
            else:
                acc_stats += new_acc_stats

        return (acc_stats, n_frames)

    @staticmethod
    @interactive
    def objective(fea_list):
        """Jod compute the objective function.

        Parameters
        ----------
        fea_list : list
            List of features file.

        Returns
        -------
        exp_llh : float
            Accumulated log-likelihood.

        """
        exp_llh = 0.
        n_frames = 0

        # For each features file...
        for fea_file in fea_list:
            # Load the features.
            data = read_htk(fea_file)

            # Mean/Variance normalization.
            #var = 1. / data_stats['precision']
            #data -= data_stats['mean']
            #data /= numpy.sqrt(var)

            # Get the accumulated sufficient statistics for the
            # given set of features.
            log_norm, _ = model.vb_e_step(data)

            exp_llh += numpy.sum(log_norm)
            n_frames += len(data)

        return exp_llh, n_frames

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
        stats_list = self.dview.map_sync(StochasticVBInference.e_step,
                                         fea_list)

        # Accumulate the results from all the jobs.
        acc_stats = stats_list[0][0]
        batch_n_frames = stats_list[0][1]
        for new_acc_stats, new_batch_n_frames in stats_list[1:]:
            acc_stats += new_acc_stats
            batch_n_frames += new_batch_n_frames

        # Scale the statistics.
        scale = self.data_stats['count'] / batch_n_frames
        acc_stats *= scale

        # Compute the learning rate.
        lrate = self.scale * \
            ((self.delay + time_step)**(-self.forgetting_rate))

        # Update the parameters.
        self.model.natural_grad_update(acc_stats, lrate)

    def validate(self, fea_list):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel computation of the gradients.
        res = self.dview.map_sync(StochasticVBInference.objective, fea_list)

        exp_llh = res[0][0]
        n_frames = res[0][1]
        for val1, val2 in res[1:]:
            exp_llh += val1
            n_frames += val2

        kl_div = self.model.kl_div_posterior_prior()

        return (exp_llh - kl_div) / n_frames


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
            new_gradients = model.get_gradients(data)

            # Global accumulators.
            n_frames += len(data)
            if gradients is None:
                gradients = new_gradients
            else:
                for grad1, grad2 in zip(gradients, new_gradients):
                    grad1 += grad2

        return (gradients, n_frames)

    @staticmethod
    @interactive
    def objective(fea_list):
        """Jod to compute the objective function.

        Parameters
        ----------
        fea_list : list
            List of features file.

        Returns
        -------
        exp_llh : float
            Accumulated log-likelihood.

        """
        objective = 0.
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
            objective += model.log_likelihood(data)
            n_frames += len(data)

        return objective, n_frames

    def __init__(self, dview, data_stats, params, model):
        Inference.__init__(self, dview, data_stats, params, model)
        self.forgetting_rate = float(params.get('forgetting_rate', .51))
        self.delay = float(params.get('delay', 0.))
        self.scale = float(params.get('scale', 1.))
        self.b1 = float(params.get('b1', .95))
        self.b2 = float(params.get('b2', .999))
        self.lrate = float(params.get('lrate', .01))
        self.model = model

        self.pmean = []
        self.pvar = []
        for param in self.model.params:
            self.pmean.append(np.zeros_like(param))
            self.pvar.append(np.zeros_like(param))

    def adam_update(self, params, gradients, time_step):
        gamma = np.sqrt(1 - self.b2**time_step) / (1 - self.b1**time_step)

        for idx in range(len(params)):
            grad = gradients[idx]
            pmean = self.pmean[idx]
            pvar = self.pvar[idx]

            # Biased moments estimate.
            new_m = self.b1 * pmean + (1. - self.b1) * grad
            new_v = self.b2 * pvar + (1. - self.b2) * (grad**2)

            # Biased corrected moments estimate.
            c_m = new_m / (1 - self.b1**time_step)
            c_v = new_v / (1 - self.b2**time_step)

            params[idx] += self.lrate * c_m / (np.sqrt(c_v) + 1e-8)

            self.pmean[idx] = new_m
            self.pvar[idx] = new_v

    def train(self, fea_list, epoch, time_step):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel computation of the gradients.
        res_list = self.dview.map_sync(AdamSGAInference.gradients, fea_list)

        # Total number of frame of the mini-batch.
        total_n_frames = 0.
        grads = None
        for new_grads, n_frames in res_list:
            total_n_frames += n_frames
            if grads is None:
                grads = new_grads
            else:
                for grad1, grad2 in zip(grads, new_grads):
                    grad1 += grad2

        # Rescale the gradients.
        for grad in grads:
            grad /= total_n_frames

        # Update the parameters of the VAE.
        self.adam_update(self.model.params, grads, epoch)

    def validate(self, fea_list):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel computation of the gradients.
        res = self.dview.map_sync(AdamSGAInference.objective, fea_list)

        exp_llh = res[0][0]
        n_frames = res[0][1]
        for val1, val2 in res[1:]:
            exp_llh += val1
            n_frames += val2

        return exp_llh / n_frames


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
        from amdtk import read_htk
        import numpy

        # Initialize the returned values.
        acc_stats = None
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

            # Optimize the log factors q(Z) and q(X)
            resps, exp_np1, exp_np2, s_stats, model_data = \
                model.optimize_local_factors(data, 1)

            # Get the gradients.
            new_gradients = model.get_gradients(data, exp_np1, exp_np2)

            # Accumulate the statistics for the latent model.
            new_acc_stats = model.prior.accumulate_stats(s_stats, resps,
                                                         model_data)

            # Global accumulators.
            n_frames += len(data)
            if gradients is None:
                acc_stats = new_acc_stats
                gradients = new_gradients
            else:
                acc_stats += new_acc_stats
                for grad1, grad2 in zip(gradients, new_gradients):
                    grad1[0] += grad2[0]
        return (gradients, acc_stats, n_frames)

    @staticmethod
    @interactive
    def objective(fea_list):
        # Initialize the returned values.
        exp_llh = 0.
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
                model.optimize_local_factors(data, 1)

            new_llh = model.log_likelihood(data, exp_np1, exp_np2)

            # Global accumulators.
            n_frames += len(data)
            exp_llh += numpy.sum(new_llh)

        return (exp_llh, n_frames)

    def __init__(self, dview, data_stats, params, model):
        Inference.__init__(self, dview, data_stats, params, model)
        self.forgetting_rate = float(params.get('forgetting_rate', .51))
        self.delay = float(params.get('delay', 0.))
        self.scale = float(params.get('scale', 1.))
        self.b1 = float(params.get('b1', .95))
        self.b2 = float(params.get('b2', .999))
        self.lrate = float(params.get('lrate', .01))
        self.update_rate = float(params.get('update_rate', 1))
        self.model = model
        self.time_step2 = 0

        self.pmean = []
        self.pvar = []
        for param in self.model.params:
            self.pmean.append(np.zeros_like(param))
            self.pvar.append(np.zeros_like(param))

        self.pmean_p = []
        self.pvar_p = []
        for param in self.model.prior.get_params():
            self.pmean_p.append(np.zeros_like(param))
            self.pvar_p.append(np.zeros_like(param))

    def adam_update(self, pmean, pvar, params, gradients, time_step, lrate):
        for idx in range(len(params)):
            grad = gradients[idx]
            m = pmean[idx]
            v = pvar[idx]

            # Biased moments estimate.
            new_m = self.b1 * m + (1. - self.b1) * grad
            new_v = self.b2 * v + (1. - self.b2) * (grad**2)

            # Biased corrected moments estimate.
            c_m = new_m / (1 - self.b1**time_step)
            c_v = new_v / (1 - self.b2**time_step)

            params[idx] += lrate * c_m / (np.sqrt(c_v) + 1e-8)

            pmean[idx] = new_m
            pvar[idx] = new_v

    def train(self, fea_list, epoch, time_step):
        self.time_step2 += 1

        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel computation of the gradients.
        res_list = self.dview.map_sync(SVAEAdamSGAInference.gradients, fea_list)

        # Total number of frame of the mini-batch.
        total_n_frames = 0.
        acc_stats = None
        grads = None
        for new_grads, new_acc_stats, n_frames in res_list:
            total_n_frames += n_frames
            if grads is None:
                acc_stats = new_acc_stats
                grads = new_grads
            else:
                acc_stats += new_acc_stats
                for grad1, grad2 in zip(grads, new_grads):
                    grad1[0] += grad2[0]

        # Scale the statistics.
        scale = self.data_stats['count'] / total_n_frames
        acc_stats *= scale

        # Compute the learning rate.
        if self.time_step2 % self.update_rate == 0:
            lrate = self.scale * \
                ((self.delay + self.time_step2)**(-self.forgetting_rate))

            # Update the parameters.
            self.model.prior.natural_grad_update(acc_stats, lrate)

        # Rescale the gradients.
        for grad in grads:
            grad /= total_n_frames

        # Update the parameters of the VAE.
        self.adam_update(
            self.pmean,
            self.pvar,
            self.model.params,
            grads,
            time_step,
            self.lrate
        )

        #params_p = self.model.prior.get_params()
        #self.adam_update(
        #    self.pmean_p,
        #    self.pvar_p,
        #    params_p,
        #    grads_p,
        #    time_step,
        #    self.lrate_prior,
        #)
        #self.model.prior.set_params(params_p)

    def validate(self, fea_list):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel computation of the gradients.
        res = self.dview.map_sync(SVAEAdamSGAInference.objective, fea_list)

        exp_llh = res[0][0]
        n_frames = res[0][1]
        for val1, val2 in res[1:]:
            exp_llh += val1
            n_frames += val2

        kl_div = self.model.prior.kl_div_posterior_prior()

        return (exp_llh - kl_div) / n_frames

