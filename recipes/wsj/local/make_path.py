
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add_ext', action='store_true',
                        help='add "wv1" extension')
    parser.add_argument('--split_file', help='Split file containing the speech '
                            'segments without begin and '
                            'end non speech segments.')
    parser.add_argument('path', help='path to wsj database')
    args = parser.parse_args()

    # load segments file
    segments_dict = dict()
    if args.split_file is not None:
        with open(args.split_file) as fid:
            for line in fid:
                data = line.split()
                segments_dict[data[0]] = data[1:3]

    line = sys.stdin.readline()
    while line != '':
        line = line.strip()

        # We use the file's name as the key. 
        basename = os.path.basename(line)
        key, ext = os.path.splitext(basename)

        # Build the path from the entry.
        line = line.replace(' ', '')
        line = line.replace('_', '-', 1)
        line = line.replace('_', '.', 1)
        line = line.replace(':', '/', 1)
        
        if args.add_ext:
            line += '.wv1'
        
        try:
            start, end = segments_dict[key]
            line += '[{}[{}'.format(start, end)
        except KeyError:
            pass
          
        entry = args.path + '/' + line + ':' + key

        print(entry)

        line = sys.stdin.readline()


if __name__ == '__main__':
    main()
else:
    raise ImportError('this script cannot be imported')
