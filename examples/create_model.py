
"""
Create a model.

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
import ast
from ipyparallel import Client
import amdtk
import numpy as np


DOC = "Create and randomly initialze a model."


def build_gmm(args, data_stats):
    eps = 1
    n_components = int(args.get('n_components', 1))
    mean = data_stats['mean']
    precision = data_stats['precision']

    prior = amdtk.NormalGamma(
        mean,
        np.ones_like(mean),
        eps * np.ones_like(precision),
        eps / precision,
    )

    dirichlet_prior = amdtk.Dirichlet(np.ones(n_components))
    dirichlet_posterior = amdtk.Dirichlet(np.ones(n_components))

    components = []
    cov = np.diag(1. / precision)
    for i in range(n_components):
        s_mean = np.random.multivariate_normal(mean, cov)
        posterior = amdtk.NormalGamma(
            s_mean,
            np.ones_like(mean),
            eps * np.ones_like(precision),
            eps / precision)
        components.append(amdtk.NormalDiag(prior, posterior))

    return amdtk.Mixture(dirichlet_prior, dirichlet_posterior,  components)


def build_phone_loop(args, data_stats):
    n_units = int(args.get('n_units', 1))
    n_states = int(args.get('n_states', 1))
    precision_scale = float(args.get('precision_scale', 1.))
    mean = data_stats['mean']
    precision = data_stats['precision']

    prior = amdtk.NormalGamma(mean, np.ones_like(mean), precision,
                              np.ones_like(mean))

    dirichlet_prior = amdtk.Dirichlet(np.ones(n_units))
    dirichlet_posterior = amdtk.Dirichlet(np.ones(n_units))

    components = []
    scaled_precision = precision_scale * precision
    cov = np.diag(1. / scaled_precision)
    for i in range(n_units * n_states):
        s_mean = np.random.multivariate_normal(mean, cov)
        posterior = amdtk.NormalGamma(s_mean, np.ones_like(mean), scaled_precision,
                                      np.ones_like(mean))
        components.append(amdtk.NormalDiag(prior, posterior))

    return amdtk.PhoneLoop(dirichlet_prior, dirichlet_posterior, components)


def build_vae(args, data_stats):
    n_layers = int(args.get('n_layers', 1))
    n_units = int(args.get('n_units', 1))
    activation = args.get('activation', 'relu')
    dim_latent = int(args.get('dim_latent', 1))
    dim_fea = len(data_stats['mean'])

    return amdtk.VAE(
        data_stats['mean'],
        data_stats['precision'],
        dim_fea,
        dim_latent,
        n_layers,
        n_units,
        activation,
        0, # Unused.
        non_informative_prior=True
    )


build_model = {
    'gmm': build_gmm,
    'hmm': build_phone_loop,
    'vae': build_vae,
}


def main():
    # Command line argument parsing.
    parser = argparse.ArgumentParser(description=DOC)

    # Model settings.
    group = parser.add_argument_group('model')
    group.add_argument('--whitened', action='store_true',
                       help='indicate that the data will be whitened.')
    group.add_argument('--model_type', choices=build_model.keys(),
                       default='gmm', help='type of the model to build')
    group.add_argument('--model_args', type=str, default='{}',
                       help='model specific arguments')

    # Mandatory arguments.
    parser.add_argument('stats', type=argparse.FileType('rb'),
                        help='data statistics')
    parser.add_argument('model', type=argparse.FileType('wb'),
                        help='path to the output model')

    # Parse the command line.
    args = parser.parse_args()

    # Load the data statistics.
    data_stats = pickle.load(args.stats)
    if args.whitened:
        data_stats['mean'] = np.zeros_like(data_stats['mean'])
        data_stats['precision'] = np.ones_like(data_stats['precision'])

    # Load the model arguments.
    model_args = ast.literal_eval(args.model_args)

    # Create the phone loop.
    model = build_model[args.model_type](model_args, data_stats)

    # Write the model on the disk.
    model.save(args.model)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
