
"""Get statistics from the data."""

# Needed modules 
# ----------------------------------------------------------------------------
import argparse
import pickle
from ipyparallel import Client
import numpy as np


def data_stats(filename):
    """Thread to collect the sufficients statistics for the mean and the
    covariance matrix from a set of features.

    """
    # We  re-import this module here because this code will run
    # remotely.
    # ------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)

    # Parallel environment options.
    # ------------------------------------------------------------------------
    group = parser.add_argument_group('parallel environment')
    group.add_argument('--profile', help='ipyparallel profile name '
                       '(default: None)')

    # Mandatory arguments.
    # ------------------------------------------------------------------------
    parser.add_argument('fealist', type=argparse.FileType('r'),
                        help='list of features files')
    parser.add_argument('stats', type=argparse.FileType('wb'),
                        help='path to the output statistics')

    # Parse the command line.
    # ------------------------------------------------------------------------
    args = parser.parse_args()

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

    # Estimate the mean and variance from the given features list.
    # ------------------------------------------------------------------------
    stats = dview.map_sync(data_stats, segments)

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

    # Write the model on the disk.
    # ------------------------------------------------------------------------
    pickle.dump((n_frames, data_mean, data_precision), args.stats)

# Makes sure this script cannot be imported.
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
else:
    print('This script cannot be imported')
    exit(1)
