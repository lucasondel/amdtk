
"""
Train a SVAE model.

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
import ast
import os
import pickle
from ipyparallel import Client
import amdtk

DOC = "Train a SVAE model."


# Callback to monitor the convergence of the training.
def callback(args):
    epoch = args['epoch']
    lower_bound = args['objective']
    kl = args['kl_div']
    time = args['time']
    print('epoch=' + str(epoch), 'time=' + str(time),
          'lower-bound=' + str(lower_bound), 'kl=' + str(kl))


def main():
    # Command line argument parsing.
    parser = argparse.ArgumentParser(description=DOC)

    # Parallel environment options.
    group = parser.add_argument_group('parallel environment')
    group.add_argument('--profile', help='ipyparallel profile name '
                       '(default: None)')

    # Training.
    group = parser.add_argument_group('training')
    group.add_argument('--train_args', default='{}',
                       help='training specific argument "{key1: val1, '
                            'key2:val2,...}"')

    # Compulsory arguments.
    parser.add_argument('stats', type=argparse.FileType('rb'),
                        help='data statistics')
    parser.add_argument('fealist', type=argparse.FileType('r'),
                        help='list of features files')
    parser.add_argument('model', type=argparse.FileType('rb'),
                        help='model to train')
    parser.add_argument('prior', type=argparse.FileType('rb'),
                        help='SVAE prior')
    parser.add_argument('out_model', type=argparse.FileType('wb'),
                        help='output trained model')
    parser.add_argument('out_prior', type=argparse.FileType('wb'),
                        help='output trained prior')

    # Parse the command line.
    args = parser.parse_args()

    # Load the data statistics.
    data_stats = pickle.load(args.stats)

    # Connect to the ipyparallel cluster.
    rc = Client(profile=args.profile)
    dview = rc[:]
    print('# jobs:', len(dview))

    # Load the list of features.
    segments = [fname.strip() for fname in args.fealist]

    # Load the model to train.
    prior = amdtk.PersistentModel.load(args.prior)
    model = amdtk.SVAE.load(args.model)
    print(model)
    model.prior = prior

    # Parse the training arguments.
    train_args = ast.literal_eval(args.train_args)

    # Train the model.
    training = amdtk.SVAEAdamSGAInference(dview, data_stats, train_args,
                                          model)
    training.run(segments, callback)

    # Write the updated model on the disk.
    model.save(args.out_model)
    model.prior.save(args.out_prior)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
