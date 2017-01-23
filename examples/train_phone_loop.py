
"""Train a Phone-Loop model."""

import argparse
import ast
import os
import pickle
from ipyparallel import Client
import amdtk


def vb_callback(args):
    epoch = args['epoch']
    lower_bound = args['lower_bound']
    time = args['time']
    print('epoch=' + str(epoch), 'time=' + str(time),
          'lower-bound=' + str(lower_bound))

    tmpdir = args['tmpdir']
    model_path = os.path.join(tmpdir, 'model_epoch' + str(epoch) + '.bin')
    with open(model_path, 'wb') as f:
        pickle.dump(args['model'], f)


def svb_callback(args):
    epoch = args['epoch']
    lower_bound = args['lower_bound']
    batch = args['batch']
    n_batch = args['n_batch']
    time = args['time']
    print('epoch=' + str(epoch), 'batch='+str(batch)+'/'+str(n_batch),
          'time=' + str(time), 'lower-bound=' + str(lower_bound))

    # If this is the end of an epoch, store the model.
    if batch == n_batch:
        tmpdir = args['tmpdir']
        model_path = os.path.join(tmpdir, 'model_epoch' + str(epoch) + '.bin')
        with open(model_path, 'wb') as f:
            pickle.dump(args['model'], f)


# Callbacks to monitor the convergence.
callbacks = {
    'vb': vb_callback,
    'svb': svb_callback
}


# Possible training strategies.
training_alg = {
    'vb': amdtk.StandardVariationalBayes,
    'svb': amdtk.StochasticVariationalBayes
}


def main():
    # Command line argument parsing.
    parser = argparse.ArgumentParser(description=__doc__)

    # Parallel environment options.
    group = parser.add_argument_group('parallel environment')
    group.add_argument('--profile', help='ipyparallel profile name '
                       '(default: None)')

    # Training.
    group = parser.add_argument_group('training')
    group.add_argument('--training', default='vb',
                       choices=training_alg.keys(),
                       help='training strategy (default: vb)')
    group.add_argument('--tmpdir', default='./',
                       help='temporary directory preferably on a large '
                            'and fast disk')
    group.add_argument('--train_args', default='{}',
                       help='training specific argument "{key1: val1, '
                            'key2:val2,...}"')

    # Compulsory arguments.
    parser.add_argument('fealist', type=argparse.FileType('r'),
                        help='list of features files')
    parser.add_argument('model', type=argparse.FileType('rb'),
                        help='model to train')
    parser.add_argument('out_model', type=argparse.FileType('wb'),
                        help='output trained model')

    # Parse the command line.
    args = parser.parse_args()

    # Connect to the ipyparallel cluster.
    print('Connecting to the cluster')
    rc = Client(profile=args.profile)
    dview = rc[:]

    print('# jobs:', len(dview))

    # Load the list of features.
    segments = []
    segments_key = {}
    for line in args.fealist:
            key, segment = line.strip().split()
            segments.append(segment)
            segments_key[segment] = key
    segments = segments

    # Load the model to train.
    ploop = pickle.load(args.model)

    # Parse the training arguments.
    train_args = ast.literal_eval(args.train_args)

    # Train the model.
    training = training_alg[args.training](segments,
                                           dview,
                                           train_args,
                                           args.tmpdir,
                                           callback=callbacks[args.training])
    training.run(ploop)

    # Write the model on the disk.
    pickle.dump(ploop, args.out_model)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
