
"""Train a Phone-Loop model."""

# Needed modules.
# ----------------------------------------------------------------------------
import argparse
import ast
import os
import pickle
from ipyparallel import Client
import amdtk


# Callback to monitor the convergence of the training.
# ----------------------------------------------------------------------------
def callback(args):
    epoch = args['epoch']
    lower_bound = args['objective']
    time = args['time']
    print('epoch=' + str(epoch), 'time=' + str(time),
          'lower-bound=' + str(lower_bound))


# Possible training strategies.
# ----------------------------------------------------------------------------
training_alg = {
    'vb': amdtk.StdVBInference,
    'svb': amdtk.SGAVBInference
}


def main():
    # Command line argument parsing.
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)

    # Parallel environment options.
    # ------------------------------------------------------------------------
    group = parser.add_argument_group('parallel environment')
    group.add_argument('--profile', help='ipyparallel profile name '
                       '(default: None)')

    # Training.
    # ------------------------------------------------------------------------
    group = parser.add_argument_group('training')
    group.add_argument('--training', default='vb',
                       choices=training_alg.keys(),
                       help='training strategy (default: vb)')
    group.add_argument('--train_args', default='{}',
                       help='training specific argument "{key1: val1, '
                            'key2:val2,...}"')

    # Compulsory arguments.
    # ------------------------------------------------------------------------
    parser.add_argument('stats', type=argparse.FileType('rb'),
                        help='data statistics')
    parser.add_argument('fealist', type=argparse.FileType('r'),
                        help='list of features files')
    parser.add_argument('model', type=argparse.FileType('rb'),
                        help='model to train')
    parser.add_argument('out_model', type=argparse.FileType('wb'),
                        help='output trained model')

    # Parse the command line.
    # ------------------------------------------------------------------------
    args = parser.parse_args()
    
    # Load the data statistics.
    # ------------------------------------------------------------------------
    stats = pickle.load(args.stats)

    # Connect to the ipyparallel cluster.
    # ------------------------------------------------------------------------
    rc = Client(profile=args.profile)
    dview = rc[:]
    print('# jobs:', len(dview))

    # Load the list of features.
    # ------------------------------------------------------------------------
    segments = []
    segments_key = {}
    for line in args.fealist:
            key, segment = line.strip().split()
            segments.append(segment)
            segments_key[segment] = key
    segments = segments

    # Load the model to train.
    # ------------------------------------------------------------------------
    ploop = pickle.load(args.model)

    # Parse the training arguments.
    # ------------------------------------------------------------------------
    train_args = {'n_frames': stats[0]}
    train_args = {**train_args, **ast.literal_eval(args.train_args)}

    # Train the model.
    # ------------------------------------------------------------------------
    print(segments)
    training = training_alg[args.training](dview, train_args, ploop)
    training.run(segments, callback)

    # Write the updated model on the disk.
    # ------------------------------------------------------------------------
    pickle.dump(ploop, args.out_model)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
