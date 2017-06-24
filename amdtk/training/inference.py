
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

class Optimizer(metaclass=abc.ABCMeta):

    def __init__(self, dview, data_stats, args, model):
        self.dview = dview
        self.epochs = int(args.get('epochs', 1))
        self.batch_size = int(args.get('batch_size', 2))
        self.model = model
        self.time_step = 0
        self.data_stats = data_stats

        with self.dview.sync_imports():
            import numpy
            from amdtk import read_htk

        self.dview.push({
            'data_stats': data_stats
        })

    def run(self, data, callback, alignments=None):
        start_time = time.time()

        for epoch in range(self.epochs):
            self.time_step += 1

            # Shuffle the data to avoid cycle in the training.
            np_data = np.array(data, dtype=object)
            idxs = np.arange(0, len(data))
            np.random.shuffle(idxs)
            shuffled_data = np_data[idxs]

            if self.batch_size < 0:
                batch_size = len(data)
            else:
                batch_size = self.batch_size

            for mini_batch in range(0, len(data), batch_size):
                # Index of the data mini-batch.
                start = mini_batch
                end = mini_batch + batch_size

                # Reshaped the list of features.
                fea_list = shuffled_data[start:end]
                n_utts = batch_size // len(self.dview)
                new_fea_list = [fea_list[i:i + n_utts]  for i in
                                range(0, len(fea_list), n_utts)]

                # Update the model.
                objective = \
                    self.train(new_fea_list, epoch + 1, self.time_step,
                               alignments)

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
        pass


class StochasticVBOptimizer(Optimizer):

    @staticmethod
    @interactive
    def e_step(args_list):
        import os

        exp_llh = 0.
        acc_stats = None
        n_frames = 0

        for arg in args_list:
            fea_file = arg

            # Mean / Variance normalization.
            data = read_htk(fea_file)
            data -= data_stats['mean']
            data /= numpy.sqrt(data_stats['var'])

            # Check if the alignments is provided for this utterance.
            if alignments is not None:
                try:
                    bname = os.path.basename(fea_file)
                    key, ext = os.path.splitext(bname)
                    ali = alignments[key]
                except KeyError:
                    ali = None
            else:
                ali = None

            # Get the accumulated sufficient statistics for the
            # given set of features.
            s_stats = model.get_sufficient_stats(data)
            posts, llh, new_acc_stats = model.get_posteriors(s_stats,
                                                             accumulate=True,
                                                             alignments=ali)

            exp_llh += numpy.sum(llh)
            n_frames += len(data)
            if acc_stats is None:
                acc_stats = new_acc_stats
            else:
                acc_stats += new_acc_stats

        return (exp_llh, acc_stats, n_frames)


    def __init__(self, dview, data_stats, args, model):
        Optimizer.__init__(self, dview, data_stats, args, model)
        self.lrate = float(args.get('lrate', 1))

    def train(self, fea_list, epoch, time_step, alignments=None):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
            'alignments': alignments
        })

        # Parallel accumulation of the sufficient statistics.
        stats_list = self.dview.map_sync(StochasticVBOptimizer.e_step,
                                         fea_list)

        # Accumulate the results from all the jobs.
        exp_llh = stats_list[0][0]
        acc_stats = stats_list[0][1]
        n_frames = stats_list[0][2]
        for val1, val2, val3 in stats_list[1:]:
            exp_llh += val1
            acc_stats += val2
            n_frames += val3

        kl_div = self.model.kl_div_posterior_prior()

        # Scale the statistics.
        scale = self.data_stats['count'] / n_frames
        acc_stats *= scale
        self.model.natural_grad_update(acc_stats, self.lrate)

        return (scale * exp_llh - kl_div) / self.data_stats['count']


class SVAEStochasticVBOptimizer(Optimizer):

    @staticmethod
    @interactive
    def e_step(args_list):
        import os

        exp_llh = 0.
        acc_stats = None
        acc_grads = None
        n_frames = 0

        for arg in args_list:
            fea_file = arg

            # Mean / Variance normalization.
            data = read_htk(fea_file)
            data -= data_stats['mean']
            data /= numpy.sqrt(data_stats['var'])

            # Check if the alignments is provided for this utterance.
            if alignments is not None:
                try:
                    bname = os.path.basename(fea_file)
                    key, ext = os.path.splitext(bname)
                    ali = alignments[key]
                except KeyError:
                    ali = None
            else:
                ali = None

            # Gradient of the model for the given mini-batch.
            llh, new_acc_stats, grads = model.get_gradients(data,
                                                            alignments=ali)

            # Accumulate.
            exp_llh += numpy.sum(llh)
            n_frames += len(data)
            if acc_stats is None:
                acc_stats = new_acc_stats
                acc_grads = grads
            else:
                acc_stats += new_acc_stats
                for i, grad in enumerate(grads):
                    acc_grads[i] += grad

        return (exp_llh, acc_grads, acc_stats, n_frames)


    def __init__(self, dview, data_stats, args, model):
        Optimizer.__init__(self, dview, data_stats, args, model)
        self.lrate1 = float(args.get('lrate1', 1e-3))
        self.lrate2 = float(args.get('lrate2', 1e-3))

        # Initialize the mean / variance of the gradient for the
        # ADAM updates.
        self.pmean = []
        self.pvar = []
        for param in model.params:
            self.pmean.append(np.zeros_like(param.get_value()))
            self.pvar.append(np.zeros_like(param.get_value()))

    def adam_update(self, pmean, pvar, params, gradients, time_step, b1, b2,
                    lrate):
        for idx in range(len(params)):
            grad = gradients[idx]
            p_m = pmean[idx]
            p_v = pvar[idx]

            # Biased moments estimate.
            new_m = b1 * p_m + (1. - b1) * grad
            new_v = b2 * p_v + (1. - b2) * (grad**2)

            # Biased corrected moments estimate.
            c_m = new_m / (1 - b1**time_step)
            c_v = new_v / (1 - b2**time_step)

            p_value = params[idx].get_value()
            params[idx].set_value(p_value + lrate * c_m / (np.sqrt(c_v) + 1e-8))

            pmean[idx] = new_m
            pvar[idx] = new_v

    def train(self, fea_list, epoch, time_step, alignments=None):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
            'alignments': alignments
        })

        # Parallel accumulation of the sufficient statistics.
        stats_list = self.dview.map_sync(SVAEStochasticVBOptimizer.e_step,
                                         fea_list)

        # Accumulate the results from all the jobs.
        exp_llh = stats_list[0][0]
        acc_grads = stats_list[0][1]
        acc_stats = stats_list[0][2]
        n_frames = stats_list[0][3]
        for val1, val2, val3, val4 in stats_list[1:]:
            exp_llh += val1
            acc_grads += val2
            acc_stats += val3
            n_frames += val4

        # Update the neural-network parameters.
        self.adam_update(
            self.pmean,
            self.pvar,
            self.model.params,
            acc_grads,
            time_step,
            .95,
            .999,
            self.lrate1
        )

        kl_div = self.model.prior_latent.kl_div_posterior_prior()

        # Scale the statistics.
        scale = self.data_stats['count'] / n_frames
        acc_stats *= scale
        self.model.prior_latent.natural_grad_update(acc_stats, self.lrate2)

        return (scale * exp_llh - kl_div) / self.data_stats['count']

