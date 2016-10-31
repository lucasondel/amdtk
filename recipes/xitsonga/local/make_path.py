#!/usr/bin/env python
import sys
import os


def main():
    segments_dict = dict()
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as fid:
            for line in fid:
                data = line.split()
                segments_dict[data[0]] = data[1:3]

    line = sys.stdin.readline()

    while line != '':
        line = line.strip()

        # We use the file's name as the key. 
        basename = os.path.basename(line)
        key, ext = os.path.splitext(basename)

        try:
            entry = '{}[{}[{}:{}'.format(line, segments_dict[key][0], segments_dict[key][1], key)
        except KeyError:
            entry = line + ':' + key

        print(entry)

        line = sys.stdin.readline()


if __name__ == '__main__':
    main()
else:
    raise ImportError('this script cannot be imported')
