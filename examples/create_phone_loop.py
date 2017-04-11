
"""
Build a Bayesian phone-loop model.

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

import argparse
import pickle
from ipyparallel import Client
import amdtk
import numpy as np


DOC = "Build Bayesian phone loop model."


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
    parser = argparse.ArgumentParser(description=DOC)

    # Model settings.
    group = parser.add_argument_group('model')
    group.add_argument('--n_units', type=int, default=1,
                       help='maximum number of units, i.e. truncation of the '
                       'Dirichlet Process (default: 1)')
    group.add_argument('--n_states', type=int, default=1,
                       help='number of states per unit (default: 1)')

    # Mandatory arguments.
    parser.add_argument('stats', type=argparse.FileType('rb'),
                        help='data statistics')
    parser.add_argument('model', type=argparse.FileType('wb'),
                        help='path to the output model')

    # Parse the command line.
    args = parser.parse_args()

    # Load the data statistics.
    data_stats = pickle.load(args.stats)

    # Create the phone loop.
    ploop = build_phone_loop(
        args.n_units,
        args.n_states,
        data_stats['mean'],
        data_stats['precision']
    )

    # Write the model on the disk.
    ploop.save(args.model)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
