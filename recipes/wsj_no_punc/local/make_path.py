
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add_ext', action='store_true',
                        help='add "wv1" extension')
    parser.add_argument('path', help='path to wsj database')
    args = parser.parse_args()
    line = sys.stdin.readline()

    while line != '':
        line = line.strip()

        # We use the file's name as the key. 
        basename = os.path.basename(line)
        key, ext = os.path.splitext(basename)

        # Build the path from the entry.
        l = list(line)
        l[2] = '-'
        l[4] = '.'
        l[6] = '/'
        line = line.replace(' ', '')
        line = line.replace('_', '-', 1)
        line = line.replace('_', '.', 1)
        line = line.replace(':', '/', 1)

        if args.add_ext: 
            entry = args.path + '/' + line + '.wv1:' + key
        else:
            entry = args.path + '/' + line + ':' + key

        print(entry)

        line = sys.stdin.readline()


if __name__ == '__main__':
    main()
else:
    raise ImportError('this script cannot be imported')
