
"""
Generate features from a vae.

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

DOC = "Generate features from a VAE."


def transform_features(line):
    """Job to transform the features of 1 utterance."""
    from amdtk import read_htk, write_htk

    # Separate input and output paths.
    fea_file, out_file = line.strip().split()

    # Load the features.
    data = read_htk(fea_file)

    # Transform the features.
    new_data = model.encode(data)

    # Store the new_features.
    write_htk(out_file, new_data)

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
    parser.add_argument('scp', type=argparse.FileType('r'),
                        help='list of input and output features files')
    parser.add_argument('model', type=argparse.FileType('rb'),
                        help='model to train')

    # Parse the command line.
    args = parser.parse_args()

    # Connect to the ipyparallel cluster.
    rc = Client(profile=args.profile)
    dview = rc[:]
    print('# jobs:', len(dview))

    # Load the model to train.
    model = amdtk.VAE.load(args.model)
    dview.push({'model': model })

    dview.map_sync(transform_features, args.scp)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

