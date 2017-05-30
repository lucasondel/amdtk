
"""Implementation of the training agorithms for the phone loop."""

import os
import time
import pickle
import shutil
import random
from ipyparallel.util import interactive
import numpy as np


def acc_stats(stats_list):
    """Accumulate a list of sufficient statistics.

    Parameters
    ----------
    stats_list : list
        List of sufficient statistics.

    Returns
    -------
    stats : dict
        Accumulated sufficient statistics.

    """
    new_stats = {}

    for stats in stats_list:
        for key1, model_stats in stats.items():
            if key1 not in new_stats:
                new_stats[key1] = {}
            for key2, value in model_stats.items():
                try:
                    new_stats[key1][key2] += value
                except KeyError:
                    new_stats[key1][key2] = value

    return new_stats


def add_stats(stats, a_stats):
    """Add a subset of the statistics.

    Parameters
    ----------
    stats : dictionary
        Accumulated statistics.
    a_stats : dictionary
        Statistics to add from the total accumulated statistics.

    """
    for key1, model_stats in a_stats.items():
        if key1 not in stats:
            stats[key1] = model_stats
        else:
            for key2, value in model_stats.items():
                stats[key1][key2] += value


def remove_stats(stats, rm_stats):
    """Remove a subset of the statistics.

    Parameters
    ----------
    stats : dictionary
        Accumulated statistics.
    rm_stats : dictionary
        Statistics to remove from the total accumulated statistics.

    """
    for key1, model_stats in rm_stats.items():
        for key2, value in model_stats.items():
            stats[key1][key2] -= value

@interactive
def log_predictive(fea_file):
    """Lower-bound of the predictive distribution.

    Parameters
    ----------
    fea_file : str
        Path to a features file (HTK format).

    Returns
    -------
    stats : dict
        Dictionary of with the log-predictive probability.

    """
    # pylint: disable=too-many-locals
    # Because this function is meant to be executed in a
    # separated thread, we prefer to reduce the number of
    # function call to simplify the dependencies.

    # pylint: disable=global-variable-not-assigned
    # pylint: disable=undefined-variable
    # The value of these global variable will be pushed
    # to the workers dynamically.
    global MODEL, TEMP_DIR

    # pylint: disable=redefined-outer-name
    # pylint: disable=reimported
    # These imports are done on the remote workers.
    import os
    import pickle

    # Extract the key of the utterance.
    basename = os.path.basename(fea_file)
    key, ext = os.path.splitext(basename)
    if '[' in ext:
        idx = ext.index('[')
        key += ext[idx:]

    # Load the features.
    data = read_htk(fea_file)

    expected_llh, unit_stats, state_resps, comp_resps = \
        MODEL.expected_log_likelihood(data)

    # Add the normalizer to the stats to compute the
    # lower bound.
    stats = {}
    stats[-1] = {
        'E_log_X': expected_llh,
        'N': data.shape[0]
    }

    # Store the stats.
    out_path = os.path.join(TEMP_DIR, key)
    with open(out_path, 'wb') as file_obj:
        pickle.dump(stats, file_obj)

    return out_path


@interactive
def std_exp(fea_file):
    """E-Step of the standard Variational Bayes.

    Parameters
    ----------
    fea_file : str
        Path to a features file (HTK format).

    Returns
    -------
    stats : dict
        Dictionary of statistics.

    """
    # pylint: disable=too-many-locals
    # Because this function is meant to be executed in a
    # separated thread, we prefer to reduce the number of
    # function call to simplify the dependencies.

    # pylint: disable=global-variable-not-assigned
    # pylint: disable=undefined-variable
    # The value of these global variable will be pushed
    # to the workers dynamically.
    global MODEL, TEMP_DIR, ALIGNMENTS

    # pylint: disable=redefined-outer-name
    # pylint: disable=reimported
    # These imports are done on the remote workers.
    import os
    import pickle

    # Extract the key of the utterance.
    basename = os.path.basename(fea_file)
    key, ext = os.path.splitext(basename)
    if '[' in ext:
        idx = ext.index('[')
        key += ext[idx:]

    # Load the features.
    data = read_htk(fea_file)

    # Compute the responsibilities per component.
    if ALIGNMENTS is not None:
        ali = ALIGNMENTS[fea_file]
    else:
        ali = None
    expected_llh, unit_stats, state_resps, comp_resps = \
            MODEL.expected_log_likelihood(data)

    # Get the sufficient statistics of the model given the
    # responsibilities.
    stats = MODEL.get_stats(data, unit_stats, state_resps,
                            comp_resps)

    # Add the normalizer to the stats to compute the
    # lower bound.
    stats[-1] = {
        'E_log_X': expected_llh,
        'N': data.shape[0]
    }

    # Store the stats.
    out_path = os.path.join(TEMP_DIR, key)
    with open(out_path, 'wb') as file_obj:
        pickle.dump(stats, file_obj)

    return out_path


@interactive
def count_frames(fea_file):
    """Count the number of frames in the features file.

    Parameters
    ----------
    fea_file : str
        Path to the features file.

    Returns
    -------
    count : int
        Number of frames.

    """
    return read_htk(fea_file).shape[0]


class StandardVariationalBayes(object):
    """Standard mean-field Variational Bayes training of the
    phone loop model.

    """

    def __init__(self, fealist, dview, train_args, tmpdir, alignments=None,
                 callback=None):
        """

        Parameters
        ----------
        fealist : list
            List of features file.
        dview : object
            Remote client objects to parallelize the training.
        train_args : dict
            Training specific arguments.
        tmpdir : str
            Path to the directory where to store temporary results.
        alignments : MLF data
            Unit level alignments (optional).
        callback : function
            Function called after each batch/epoch.

        """
        self.fealist = fealist
        self.dview = dview
        self.alignments = alignments
        self.callback = callback
        self.dir_path = tmpdir

        self.epochs = int(train_args.get('epochs', 1))
        self.initial_pruning = int(train_args.get('initial_pruning_threshold',
                                                  500))
        self.pruning = int(train_args.get('pruning_threshold', 100))

        with self.dview.sync_imports():
            from amdtk import read_htk

    def run(self, model):
        """Run the Standard Variational Bayes training.

        Parameters
        ----------
        model : :class:`PhoneLoop`
            Phone Loop model to train.

        """
        start_time = time.time()
        for epoch in range(self.epochs):
            # Create a temporary directory.
            self.temp_dir = os.path.join(self.dir_path, 'epoch' +
                                         str(epoch + 1))
            os.makedirs(self.temp_dir, exist_ok=True)

            # Set the pruning thresold.
            if epoch == 0:
                model.pruning_threshold = self.initial_pruning
            else:
                model.pruning_threshold = self.pruning

            # Perform one epoch of the training.
            lower_bound = self.epoch(model)

            # Monitor the convergence after each epoch.
            if self.callback is not None:
                args = {
                    'model': model,
                    'lower_bound': lower_bound,
                    'epoch': epoch + 1,
                    'tmpdir': self.dir_path,
                    'time': time.time() - start_time
                }
                self.callback(args)

    def epoch(self, model):
        """Perform one epoch (i.e. processing the whole data set)
        of the training.

        Parameters
        ----------
        model : :class:`PhoneLoop`
            Phone Loop model to train.

        """

        # Propagate the model to all the remote nodes.
        self.dview.push({
            'MODEL': model,
            'TEMP_DIR': self.temp_dir,
            'ALIGNMENTS': self.alignments
        })

        # Optimize the latent variables given the current values of the
        # parameters of the posteriors for each feature file.
        paths = self.dview.map_sync(std_exp, self.fealist)

        # Accumulate the statistics into a single statistics object.
        total_stats = None
        for path in paths:
            if total_stats is None:
                with open(path, 'rb') as file_obj:
                    total_stats = pickle.load(file_obj)
            else:
                with open(path, 'rb') as file_obj:
                    add_stats(total_stats, pickle.load(file_obj))

        # Compute the lower bound before the update.
        lower_bound = (total_stats[-1]['E_log_X'] -
                       model.kl_divergence()) / total_stats[-1]['N']

        # Second step of the coordinate ascent: optimize the posteriors
        # parameters given the values of the latent variables.
        model.update(total_stats)

        return lower_bound


class StochasticVariationalBayes(object):
    """Stochastic (mean-field) Variational Bayes training of the
    phone loop model.

    """

    def __init__(self, fealist, dview, train_args, tmpdir, alignments=None,
                 callback=None):
        """

        Parameters
        ----------
        fealist : list
            List of features file.
        dview : object
            Remote client objects to parallelize the training.
        train_args : dict
            Training specific arguments.
        tmpdir : str
            Path to the directory where to store temporary results.
        alignments : MLF data
            Unit level alignments (optional).
        callback : function
            Function called after each batch/epoch.

        """
        self.fealist = fealist
        self.dview = dview
        self.alignments = alignments
        self.callback = callback
        self.dir_path = tmpdir

        self.epochs = int(train_args.get('epochs', 1))
        self.batch_size = int(train_args.get('batch_size', 1))
        self.initial_pruning = int(train_args.get('initial_pruning_threshold',
                                                  500))
        self.pruning = int(train_args.get('pruning_threshold', 100))
        self.forgetting_rate = float(train_args.get('forgetting_rate', .51))
        self.delay = float(train_args.get('delay', 0))
        self.scale = float(train_args.get('scale', 1))

        with self.dview.sync_imports():
            from amdtk import read_htk

        # Count the total number of frames in the DB.
        counts = self.dview.map_sync(count_frames, self.fealist)
        self.n_frames = np.sum(counts)

    def run(self, model):
        """Run the Stochastic Variational Bayes training.

        Parameters
        ----------
        model : :class:`PhoneLoop`
            Phone Loop model to train.

        """
        t = 0
        start_time = time.time()
        for epoch in range(self.epochs):
            # Create a temporary directory.
            self.temp_dir = os.path.join(self.dir_path, 'epoch' +
                                         str(epoch + 1))
            os.makedirs(self.temp_dir, exist_ok=True)

            # Set the pruning thresold.
            if epoch == 0:
                model.pruning_threshold = self.initial_pruning
            else:
                model.pruning_threshold = self.pruning

            # Perform one epoch of the training.
            fealist = random.sample(self.fealist, len(self.fealist))
            for i in range(0, len(self.fealist), self.batch_size):
                # Time step of the gradient descent.
                t += 1

                # Compute the learning rate for the given time step.
                lrate = self.scale * \
                    ((self.delay + t)**(-self.forgetting_rate))

                start = i
                end = i + self.batch_size
                lower_bound = self.batch(model,
                                         fealist[start:end],
                                         lrate)

                # Monitor the convergence after each epoch.
                if self.callback is not None:
                    args = {
                        'model': model,
                        'lower_bound': lower_bound,
                        'epoch': epoch + 1,
                        'batch': int(i / self.batch_size) + 1,
                        'n_batch': int(np.ceil(len(self.fealist) /
                                       self.batch_size)),
                        'lrate': lrate,
                        'tmpdir': self.dir_path,
                        'time': time.time() - start_time
                    }
                    self.callback(args)

        # Cleanup the resources allocated during the training.
        self.cleanup()

    def batch(self, model, batch_fea_list, lrate):
        """Perform one batch update.

        Parameters
        ----------
        model : :class:`PhoneLoop`
            Phone Loop model to train.
        batch_fea_list : list
            List of features file to evaluate the gradient.
        lrate : float
            Learning rate.

        """
        # Propagate the model to all the remote nodes.
        self.dview.push({
            'MODEL': model,
            'TEMP_DIR': self.temp_dir,
            'ALIGNMENTS': self.alignments
        })

        # Optimize the latent variables given the current values of the
        # parameters of the posteriors for each feature file.
        paths = self.dview.map_sync(std_exp, batch_fea_list)

        # Accumulate the statistics into a single statistics object.
        total_stats = None
        for path in paths:
            if total_stats is None:
                with open(path, 'rb') as file_obj:
                    total_stats = pickle.load(file_obj)
            else:
                with open(path, 'rb') as file_obj:
                    add_stats(total_stats, pickle.load(file_obj))

        # Ratio between total number of frames and number of frames
        # of current batch
        scale = self.n_frames / total_stats[-1]['N']

        # Compute the lower bound before the update.
        lower_bound = (scale * total_stats[-1]['E_log_X'] -
                       model.kl_divergence()) / self.n_frames

        # Second step of the coordinate ascent: optimize the posteriors
        # parameters given the values of the latent variables.
        model.natural_grad_update(total_stats, scale, lrate)

        return lower_bound

    def cleanup(self):
        """Cleanup the resources allocated for the training."""
        shutil.rmtree(self.temp_dir)
