
"""
Compute statistics from the data.

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
import numpy as np


DOC = """Compute statistics from the data."""


def data_stats(filename):
    """Job to collect the statistics.

    The statistics are the mean, the precision and the number of
    features vector in the whole data set.

    Parameters
    ----------
    filename : str
        Path to a features file.

    Returns
    -------
    unnorm_stats : tuple
        Tuple of unnormalize statistics.

    """
    # We  re-import this module here because this code will run
    # remotely.
    import amdtk
    data = amdtk.read_htk(filename)
    stats_0 = data.shape[0]
    stats_1 = data.sum(axis=0)
    stats_2 = (data**2).sum(axis=0)
    retval = (
        stats_0,
        stats_1,
        stats_2
    )
    return retval


def main():
    # Command line argument parsing.
    parser = argparse.ArgumentParser(description=DOC)

    # Parallel environment options.
    group = parser.add_argument_group('parallel environment')
    group.add_argument('--profile', help='ipyparallel profile name '
                       '(default: None)')

    # Mandatory arguments.
    parser.add_argument('fealist', type=argparse.FileType('r'),
                        help='list of features files')
    parser.add_argument('stats', type=argparse.FileType('wb'),
                        help='path to the output statistics')

    # Parse the command line.
    args = parser.parse_args()

    # Connect to the ipyparallel cluster.
    rc = Client(profile=args.profile)
    dview = rc[:]
    print('# jobs:', len(dview))

    # Load the list of features.
    segments = [fname.strip() for fname in args.fealist]

    # Estimate the mean and variance from the given features list.
    stats = dview.map_sync(data_stats, segments)

    # Compute the final statistics.
    n_frames = stats[0][0]
    data_mean = stats[0][1]
    data_precision = stats[0][2]
    for stats_0, stats_1, stats_2 in stats[1:]:
        n_frames += stats_0
        data_mean += stats_1
        data_precision += stats_2
    data_mean /= n_frames
    data_precision /= n_frames
    data_precision = 1. / data_precision

    print('# segments:', len(segments))
    print('features dim.:', len(data_mean))
    print('# frames:', n_frames)

    # Store the statistics on the disk.
    retval = {
        'count': n_frames,
        'mean': data_mean,
        'precision': data_precision
    }
    pickle.dump(retval, args.stats)

# Makes sure this script cannot be imported.
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)

