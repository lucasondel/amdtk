
"""Create a Phone-Loop model with the Gaussian components of the state
initialized randomly from the mean and variance estimated from the
provided features.

"""

# Needed modules 
# ----------------------------------------------------------------------------
import argparse
import pickle
from ipyparallel import Client
import amdtk
import numpy as np

 
def build_phone_loop(n_units, n_states, data_mean, data_precision):
    prior = amdtk.NormalGamma(
        data_mean,
        np.ones_like(data_mean),
        data_precision,
        np.ones_like(data_mean)
    )

    dirichlet_prior = amdtk.Dirichlet(
        np.ones(n_units)
    )
    dirichlet_posterior = amdtk.Dirichlet(
        np.ones(n_units)
    )

    components = []
    cov = np.diag(1. / data_precision)
    for i in range(n_units * n_states):
        posterior = amdtk.NormalGamma(
            np.random.multivariate_normal(data_mean, cov),
            np.ones_like(data_mean),
            data_precision,
            np.ones_like(data_mean)
        )
        components.append(amdtk.NormalDiag(prior, posterior))
    
    return amdtk.PhoneLoop(
        dirichlet_prior, 
        dirichlet_posterior,
        components
    )

def main():
    # Command line argument parsing.
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)

    # Model settings.
    # ------------------------------------------------------------------------
    group = parser.add_argument_group('model')
    group.add_argument('--n_units', type=int, default=1,
                       help='maximum number of units, i.e. truncation of the '
                       'Dirichlet Process (default: 1)')
    group.add_argument('--n_states', type=int, default=1,
                       help='number of states per unit (default: 1)')

    # Mandatory arguments.
    # ------------------------------------------------------------------------
    parser.add_argument('stats', type=argparse.FileType('rb'),
                        help='data statistics')
    parser.add_argument('model', type=argparse.FileType('wb'),
                        help='path to the output model')

    # Parse the command line.
    # ------------------------------------------------------------------------
    args = parser.parse_args()

    # Load the data statistics.
    # ------------------------------------------------------------------------
    n_frames, data_mean, data_precision = pickle.load(args.stats)

    # Create the phone loop.
    # ------------------------------------------------------------------------
    ploop = build_phone_loop(
        args.n_units, 
        args.n_states, 
        data_mean, 
        data_precision
    )

    # Write the model on the disk.
    # ------------------------------------------------------------------------
    pickle.dump(ploop, args.model)

# Makes sure this script cannot be imported.
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
