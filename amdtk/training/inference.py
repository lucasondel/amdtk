
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


def _create_ali_log_resps(phone_list, labels, extra_frames=1):
    import numpy
    n_comp = len(phone_list)
    log_resps = numpy.zeros((len(labels), n_comp)) - np.inf
    previous_phone = labels[0]
    for i in range(len(labels)):
        phone = labels[i]
        phone_idx = phone_list.index(phone)
        log_resps[i, phone_idx] = 0.
        if previous_phone != phone and extra_frames > 0:
            previous_phone_idx = phone_list.index(previous_phone)
            log_resps[i: i + extra_frames, previous_phone_idx] = 0.
            log_resps[i - extra_frames: i, phone_idx] = 0.
        previous_phone = phone

    return log_resps


class Optimizer(metaclass=abc.ABCMeta):

    def __init__(self, dview, data_stats, args, model, phone_list=None):
        self.dview = dview
        self.epochs = int(args.get('epochs', 1))
        self.batch_size = int(args.get('batch_size', 2))
        self.model = model
        self.time_step = 0
        self.data_stats = data_stats
        self.phone_list = phone_list

        with self.dview.sync_imports():
            import numpy
            from amdtk import read_htk

        self.dview.push({
            'phone_list': phone_list,
            '_create_ali_log_resps': _create_ali_log_resps,
            'data_stats': data_stats
        })

    def run(self, data, callback):
        start_time = time.time()

        for epoch in range(self.epochs):

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
                objective = \
                    self.train(new_fea_list, epoch + 1, self.time_step)

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
        exp_llh = 0.
        acc_stats = None
        n_frames = 0

        for arg in args_list:
            if isinstance(arg, numpy.ndarray):
                fea_file = arg[0]
                labels = arg[1]
                log_resps = _create_ali_log_resps(phone_list, labels)
            else:
                fea_file = arg
                log_resps = None

            data = read_htk(fea_file)
            data -= data_stats['mean']
            data *= numpy.sqrt(data_stats['precision'])

            if log_resps is not None:
                min_len = min(len(data), len(log_resps))
                data = data[:min_len]
                log_resps = log_resps[:min_len]


            # Get the accumulated sufficient statistics for the
            # given set of features.
            llh, new_acc_stats = model.vb_e_step(data, log_resps)

            exp_llh += numpy.sum(llh)
            n_frames += len(data)
            if acc_stats is None:
                acc_stats = new_acc_stats
            else:
                acc_stats += new_acc_stats

        return (exp_llh, acc_stats, n_frames)


    def __init__(self, dview, data_stats, args, model, phone_list=None):
        Optimizer.__init__(self, dview, data_stats, args, model, phone_list)
        self.lrate = float(args.get('lrate', 1))

    def train(self, fea_list, epoch, time_step):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
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


class SVAEAdamSGAOptimizer(Optimizer):

    @staticmethod
    @interactive
    def gradients(args_list):
        exp_llh = 0.
        acc_stats = None
        gradients = None
        n_frames = 0

        for arg in args_list:
            if isinstance(arg, numpy.ndarray):
                fea_file = arg[0]
                labels = arg[1]
                log_resps = _create_ali_log_resps(phone_list, labels)
            else:
                fea_file = arg
                log_resps = None

            data = read_htk(fea_file)
            data -= data_stats['mean']
            data *= numpy.sqrt(data_stats['precision'])

            if log_resps is not None:
                min_len = min(len(data), len(log_resps))
                data = data[:min_len]
                log_resps = log_resps[:min_len]

            # Get the gradients.
            llh, new_gradients, new_acc_stats = \
                model.get_gradients(prior, data, log_resps)

            # Global accumulators.
            exp_llh += llh
            n_frames += len(data)
            if gradients is None:
                acc_stats = new_acc_stats
                gradients = new_gradients
            else:
                acc_stats += new_acc_stats
                for idx in range(len(new_gradients)):
                    gradients[idx] += new_gradients[idx]

        return (exp_llh, gradients, acc_stats, n_frames)


    def __init__(self, dview, data_stats, args, model, phone_list=None):
        Optimizer.__init__(self, dview, data_stats, args, model[0],
                                  phone_list)
        self.b1 = float(args.get('b1', .95))
        self.b2 = float(args.get('b2', .999))
        self.lrate = float(args.get('lrate', .01))
        self.prior_lrate = float(args.get('prior_lrate'))
        self.delta = float(args.get('delta', 1.))
        self.model = model[0]
        self.prior = model[1]

        self.pmean = []
        self.pvar = []
        for param in self.model.params:
            self.pmean.append(np.zeros_like(param.get_value()))
            self.pvar.append(np.zeros_like(param.get_value()))

    def adam_update(self, params, gradients, time_step, lrate):
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

            p_value = params[idx].get_value()
            #params[idx].set_value(p_value + lrate * grad)
            params[idx].set_value(p_value + lrate * c_m / (np.sqrt(c_v) + 1e-8))

            self.pmean[idx] = new_m
            self.pvar[idx] = new_v

    def train(self, fea_list, epoch, time_step):
        # Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
            'prior': self.prior,
        })

        # Parallel computation of the gradients.
        res_list = self.dview.map_sync(SVAEAdamSGAOptimizer.gradients,
                                       fea_list)

        exp_llh, grads, acc_stats, n_frames = res_list[0]
        for val1, val2, val3, val4 in res_list[1:]:
            exp_llh += val1
            grads += val2
            acc_stats += val3
            n_frames += val4

        # Scale the statistics.
        scale = self.data_stats['count'] / n_frames

        # Update the parameters.
        self.prior.natural_grad_update(acc_stats, self.prior_lrate)

        # Update the parameters of the VAE.
        self.adam_update(
            self.model.params,
            grads,
            time_step * self.delta,
            scale * self.lrate
        )

        kl_div = self.prior.kl_div_posterior_prior()

        return (scale * exp_llh - kl_div) / self.data_stats['count']

