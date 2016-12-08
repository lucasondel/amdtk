
"""Implementation of the training agorithms for the phone loop."""

import tempfile
import os
import pickle
import shutil
from ipyparallel.util import interactive


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

    print('processing:', fea_file)

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
        MODEL.expected_log_likelihood(data, ali)

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


class StandardVariationalBayes(object):
    """Standard mean-field Variational Bayes training of the
    phone loop model.

    """

    def __init__(self, fealist, dview, alignments=None):
        """

        Parameters
        ----------
        fealist : list
            List of features file.
        remote_clister : object
            Remote client object to parallelize the training.
        alignments : MLF data
            Unit level alignments (optional).

        """
        self.fealist = fealist
        self.dview = dview
        cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(dir=cwd)
        self.alignments = alignments
        self.dview.block = True

        with self.dview.sync_imports():
            from amdtk import read_htk

    def epoch(self, model):
        """Perform one epoch (i.e. processing the whole data set)
        of the training.

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

        # Second step of the coordinate ascent: optimize the posteriors
        # parameters given the values of the latent variables.
        model.update(total_stats)

        return (total_stats[-1]['E_log_X'] - model.kl_divergence()) / total_stats[-1]['N']

    def cleanup(self):
        """Cleanup the resources allocated for the training."""
        shutil.rmtree(self.temp_dir)
