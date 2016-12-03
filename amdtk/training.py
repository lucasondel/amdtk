
"""Implementation of the training agorithms for the phone loop."""

import tempfile
import os 
import pickle
import shutil
import logging
from random import shuffle
import numpy as np
from scipy.misc import logsumexp
from scipy.special import gammaln
from ipyparallel.util import interactive
from . import utils
from . import nb_utils

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
    global model, temp_dir, alignments
    
    # Make the import here to avoids conflicts with local imports.
    import os
    import pickle
    from bayesian_models import utils
    
    print('processing:', fea_file)
    
    # Extract the key of the utterance.
    basename = os.path.basename(fea_file)
    key, ext = os.path.splitext(basename)
    if '[' in ext:
        idx = ext.index('[')
        key += ext[idx:]
    
    # Load the features.
    X = utils.read_htk(fea_file)
    
    # Compute the responsibilities per component.
    if alignments is not None:
        ali = alignments[fea_file]
    else:
        ali = None
    E_log_p_X, unit_resps, state_resps = model.expected_log_likelihood(X, ali)
    
    # Get the sufficient statistics of the model given the 
    # responsibilities.
    stats = model.get_stats(X, unit_resps, state_resps)
    
    # Add the normalizer to the stats to compute the 
    # lower bound.
    stats[-1] = {'E_log_X': E_log_p_X.sum()}
    
    # Store the stats.
    out_path = os.path.join(temp_dir, key)
    with open(out_path, 'wb') as f:
        pickle.dump(stats, f)
    
    return out_path

@interactive
def fe_exp(fea_file):
    """E-Step of the Free-Energy training.
    
    Parameters
    ----------
    fea_file : str
        Path to a features file (HTK format).
    
    Returns
    -------
    stats : dict
        Dictionary of statistics.
    
    """
    # Variable theoretically received from the root envorinment. 
    #-----------------------------------------------------------------------------
    global system, temp_dir, alignments
    
    # Make the import here to avoids conflicts with local imports.
    #-----------------------------------------------------------------------------
    import os
    import pickle
    from bayesian_models import utils
    
    # The phone loop model.
    #-----------------------------------------------------------------------------
    model = system.ploop
    
    # Extract the key of the utterance.
    #-----------------------------------------------------------------------------
    basename = os.path.basename(fea_file)
    key, ext = os.path.splitext(basename)
    if '[' in ext:
        idx = ext.index('[')
        key += ext[idx:]
    
    # Load the system's state corresponding to this utterance.
    #-----------------------------------------------------------------------------
    state_path = os.path.join(temp_dir, key + '_system_state.bin') 
    if os.path.exists(state_path):
        system.load_state(state_path)
    
    # Load the features.
    #-----------------------------------------------------------------------------
    X = utils.read_htk(fea_file)
    
    # Transform the features using the learning system. This corresponds to the
    # 'action' of the system on its environment.
    #-----------------------------------------------------------------------------
    Y = system.encode(X) 
    
    # Compute the responsibilities per component.
    #-----------------------------------------------------------------------------
    if alignments is not None:
        ali = alignments[fea_file]
    else:
        ali = None
    E_log_p_Y, unit_resps, state_resps = model.expected_log_likelihood(Y, ali)
    
    # Get the sufficient statistics of the model given the 
    # responsibilities.
    #-----------------------------------------------------------------------------
    stats = model.get_stats(Y, unit_resps, state_resps)
    
    # Add the normalizer to the stats to compute the 
    # lower bound.
    #-----------------------------------------------------------------------------
    stats[-1] = {'E_log_Y': E_log_p_Y.sum()}
    
    # Store the stats.
    #-----------------------------------------------------------------------------
    out_path = os.path.join(temp_dir, key)
    with open(out_path, 'wb') as f:
        pickle.dump(stats, f)
    
    return out_path

@interactive
def fe_interact(fea_file):
    """Interacting step of the Free-Energy training.
    
    Parameters
    ----------
    fea_file : str
        Path to a features file (HTK format).
    
    """
    # Variable theoretically received from the root envorinment. 
    #-----------------------------------------------------------------------------
    global system, temp_dir, alignments
    
    # Make the import here to avoids conflicts with local imports.
    #-----------------------------------------------------------------------------
    import os
    import pickle
    from bayesian_models import utils
    
    # The phone loop model.
    #-----------------------------------------------------------------------------
    model = system.ploop
    
    # Extract the key of the utterance.
    #-----------------------------------------------------------------------------
    basename = os.path.basename(fea_file)
    key, ext = os.path.splitext(basename)
    if '[' in ext:
        idx = ext.index('[')
        key += ext[idx:]
        
    # Load the system's state corresponding to this utterance.
    #-----------------------------------------------------------------------------
    state_path = os.path.join(temp_dir, key + '_system_state.bin') 
    if os.path.exists(state_path):
        system.load_state(state_path)
    
    # Load the features.
    #-----------------------------------------------------------------------------
    X = utils.read_htk(fea_file)
    
    # Update the actio of the model.
    #-----------------------------------------------------------------------------
    system.update_encoder(X)
    
    # Update the decoding (for the MI regularization).
    #-----------------------------------------------------------------------------
    system.update_decoder(X)
    
    # Store the new system's state on disk.
    #-----------------------------------------------------------------------------
    system.save_state(state_path)
    
def std_vb_evaluate(model, fea_file):
    """Evaluate the model using the predicitive density.
    
    Parameters
    ----------
    model : :class:`Mixture`
        Bayesian GMM model.
    fea_file : str
        Path to a features file (HTK format).
    
    Returns
    -------
    log_evidence : dict
        Log evidence.
    
    """
    # Load the features.
    X = utils.read_htk(fea_file)

    return model.log_predictive(X).sum()


class StandardVariationalBayes(object):

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
        self.dview.block =True
    
    def epoch(self, model):
        # Propagate the model to all the remote nodes.
        self.dview.push({
                'model': model,
                'temp_dir': self.temp_dir,
                'alignments': self.alignments
            })
        
        # Optimize the latent variables given the current values of the
        # parameters of the posteriors for each feature file. 
        paths = self.dview.map_sync(std_exp, self.fealist)

        # Accumulate the statistics into a single statistics object.
        total_stats = None
        for path in paths:
            if total_stats is None:
                with open(path, 'rb') as f:
                    total_stats = pickle.load(f)
            else:
                with open(path, 'rb') as f:
                    add_stats(total_stats, pickle.load(f))
                    
        # Second step of the coordinate ascent: optimize the posteriors
        # parameters given the values of the latent variables.
        model.update(total_stats)
        
        return total_stats[-1]['E_log_X'] - model.KL()

    def cleanup(self):
        shutil.rmtree(self.temp_dir)
        

class FreeEnergyTraining(object):

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
        
        print('tmp dir:', self.temp_dir)
    
    def epoch(self, system):
        # Propagate the model to all the remote nodes.
        #-------------------------------------------------------------------------
        self.dview.push({
                'system': system,
                'temp_dir': self.temp_dir,
                'alignments': self.alignments
            })
        
        # Optimize the latent variables given the current values of the
        # parameters of the posteriors for each feature file. 
        #-------------------------------------------------------------------------
        paths = self.dview.map_sync(fe_exp, self.fealist)

        # Accumulate the statistics into a single statistics object.
        #-------------------------------------------------------------------------
        total_stats = None
        for path in paths:
            if total_stats is None:
                with open(path, 'rb') as f:
                    total_stats = pickle.load(f)
            else:
                with open(path, 'rb') as f:
                    add_stats(total_stats, pickle.load(f))
                    
        # Second step of the coordinate ascent: optimize the posteriors
        # parameters given the values of the latent variables.
        #-------------------------------------------------------------------------
        system.ploop.update(total_stats)
        
        # We propagate the system to the nodes again as we have updated it's 
        # values.
        #-------------------------------------------------------------------------
        self.dview.push({
                'system': system,
                'temp_dir': self.temp_dir,
                'alignments': self.alignments
            })
        
        # The system has optimized its internal representation, now it will
        # optimized it's "action" on the environment.
        #-------------------------------------------------------------------------
        self.dview.map_sync(fe_interact, self.fealist)
        
        return total_stats[-1]['E_log_Y'] - system.ploop.KL()

    def cleanup(self):
        shutil.rmtree(self.temp_dir)
        