
"""Create a Phone-Loop model with the Gaussian components of the state
initialized randomly from the mean and variance estimated from the
provided features.

"""

import argparse
import pickle
from ipyparallel import Client
import amdtk
import numpy as np


def data_stats(filename):
    """Thread to collect the sufficients statistics for the mean and the
    covariance matrix from a set of features.

    """
    # We  re-import this module here because this code will run
    # remotely.
    import amdtk
    data = amdtk.read_htk(filename)
    stats_0 = data.shape[0]
    stats_1 = data.sum(axis=0)
    stats_2 = (data**2).sum(axis=0)
    stats = {}
    stats[0] = {
        'N': stats_0,
        'x': stats_1,
        'x2': stats_2
    }
    return stats


def main():
    # Command line argument parsing.
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, default=-1, help='seeding value '
                        '(default: -1)')

    # Parallel environment options.
    group = parser.add_argument_group('parallel environment')
    group.add_argument('--profile', help='ipyparallel profile name '
                       '(default: None)')

    # Model settings.
    group = parser.add_argument_group('model')
    group.add_argument('--pruning', type=float, default=float('inf'),
                       help='Pruning threshold for the Baum-Welch algorithm')
    group.add_argument('--concentration', type=float, default=1.,
                       help='Concentration of the Dirichlet Process '
                       '(default: 1)')
    group.add_argument('--n_units', type=int, default=1,
                       help='maximum number of units, i.e. truncation of the '
                       'Dirichlet Process (default: 1)')
    group.add_argument('--n_states', type=int, default=1,
                       help='number of states per unit (default: 1)')
    group.add_argument('--n_gaussian', type=int, default=1,
                       help='maximum number of Gaussian in the model '
                       '(default: 1)')
    group.add_argument('--state-weights-prior-count', type=float, default=1.,
                       help='prior (uniform) weight for the mixture '
                       '(default: 1.)')
    group.add_argument('--mean-prior-count', type=float, default=1.,
                       help='prior count for the prior mean (default: 1)')
    group.add_argument('--precision-prior-count', type=float, default=1.,
                       help='prior count for the prior precision '
                       '(default: 1)')
    group.add_argument('--ins_penalty', type=float, default=1.,
                       help='instertion penalty, i.e. fudge factor '
                       '(default: 1)')

    # Compulsory arguments.
    parser.add_argument('fealist', type=argparse.FileType('r'),
                        help='list of features files')
    parser.add_argument('model', type=argparse.FileType('wb'),
                        help='path to the output model')

    # Parse the command line.
    args = parser.parse_args()

    # Connect to the ipyparallel cluster.
    rc = Client(profile=args.profile)
    dview = rc[:]

    # Seed Numpy's random number generator if requested.
    if args.seed >= 0:
        np.random.seed(args.seed)

    # Load the list of features.
    segments = []
    segments_key = {}
    for line in args.fealist:
            key, segment = line.strip().split()
            segments.append(segment)
            segments_key[segment] = key

    # Estimate the mean and variance from the given features list.
    stats = dview.map_sync(data_stats, segments)
    stats = amdtk.acc_stats(stats)
    N = stats[0]['N']
    db_mean = stats[0]['x'] / N
    db_prec = N / stats[0]['x2']

    # Create the states' emission.
    emissions = []
    st_prior_count = np.zeros(args.n_gaussian) + args.state_weights_prior_count
    for i in range(args.n_units * args.n_states):
        gaussian = [amdtk.GaussianDiagCov(db_mean,
                                          args.mean_prior_count,
                                          db_prec,
                                          args.precision_prior_count)
                    for i in range(args.n_gaussian)]
        emissions.append(amdtk.Mixture(gaussian,
                                       st_prior_count))

    # Create the phone loop model.
    init_ploop = amdtk.PhoneLoop(args.n_units,
                                 emissions,
                                 args.concentration,
                                 args.ins_penalty,
                                 args.pruning)

    # Initialize the Gaussian mean's posterior.
    cov_mat = np.diag(1 / db_prec)
    for units in init_ploop.components:
        for g in units.components:
            g.posterior_mean = np.random.multivariate_normal(db_mean,
                                                             cov_mat)

    # Write the model on the disk.
    pickle.dump(init_ploop, args.model)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
