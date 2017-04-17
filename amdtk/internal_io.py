
"""This module provides functions to read and write specific files.
Here is the list of the supported formats:

+----------------+------+-------+----------------+
|     Format     | Read | Write | Compatibility  |
+================+======+=======+================+
| HTK binary     | yes  |  yes  |      HTK       |
+----------------+------+-------+----------------+
| HTK Label      | yes  |  yes  |      HTK       |
+----------------+------+-------+----------------+
| HTK Lattice    | yes  |  no   |      HTK       |
+----------------+------+-------+----------------+
| MLF            | yes  |  yes  |      HTK       |
+----------------+------+-------+----------------+
| TIMIT          | yes  |  no   | TIMIT database |
+----------------+------+-------+----------------+

"""

from struct import unpack, pack
import os
import operator
import numpy as np

# HTK format constants.
LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0o0000100  # has energy
_N = 0o0000200  # absolute energy supressed
_D = 0o0000400  # has delta coefficients
_A = 0o0001000  # has acceleration (delta-delta) coefficients
_C = 0o0002000  # is compressed
_Z = 0o0004000  # has zero mean static coefficients
_K = 0o0010000  # has CRC checksum
_O = 0o0020000  # has 0th cepstral coefficient
_V = 0o0040000  # has VQ data
_T = 0o0100000  # has third differential coefficients

# HTK time unit is 100ns.
TIME_UNIT = 10000000


def read_htk(path, infos=False, ignore_timing=False):
    """ Read binary file according to the specification given
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node58.html>`_.
    Eventually, one can specify a specific portion of the file to load
    by adding a suffix to path as:
    ```
    /path/to/file[start_frame:end_frame]
    ```
    No checking is done about the given frames.


    Parameters
    ----------
    path : str
        Path to the features HTK file.
    infos : boolean
        If "infos" is True, then the header informations are returned
    ignore_timing : boolean
        If True ignore the timing information and load all the
        features.

    Returns
    -------
    data : numpy.ndarray
        The data as a matrix.
    data, info: numpy.ndarray, tuple
        The data as a matrix and a tuple (nSamples, sampPeriod, sampSize)

    """
    # pylint: disable=too-many-locals
    # No need to split this code into multiple functions
    # as it will be only used here.

    start = 0
    end = None
    if '[' in path:
        tokens = path.split('[')
        path = tokens[0]
        timing = tokens[1]
        start, end = timing.strip('[]').split(',')
        start = int(start)
        end = int(end)

    with open(path, 'rb') as file_obj:
        header_size = 12
        header = file_obj.read(header_size)
        n_samples, samp_period, samp_size, param_kind = \
            unpack('>IIHH', header)
        if param_kind & _C:
            size = int(samp_size/2)
            dtype = 'h'
            if param_kind & 0x3F == IREFC:
                denom = 32767.
                bias = 0.
                offset = 0
            else:
                offset = 2*size*4
                file_obj.seek(header_size)
                denom = np.fromfile(file_obj, dtype='>f', count=size)
                bias = np.fromfile(file_obj, dtype='>f', count=size)
        else:
            size = int(samp_size/4)
            dtype = 'f'
            offset = 0

        file_obj.seek(header_size + offset)
        data = np.fromfile(file_obj, dtype='>'+dtype)
        if param_kind & _K:
            data = data[:-1]
        new_shape = (int(len(data)/size), size)
        data = data.reshape(new_shape)

        if param_kind & _C:
            data = (data + bias) / denom

        if ignore_timing:
            ret_data = data
        else:
            ret_data = data[start:end]

        if not infos:
            return ret_data
        return ret_data, (n_samples, samp_period, samp_size)


def write_htk(path, data, samp_period=100000):
    """Write binary file according to the specification given
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node58.html>`_.

    Parameters
    ----------
    path : str
        Path to the features HTK file.
    data : numpy.ndarray
        Data to write.
    samp_period : int
        Sampling periord in 100 nanoseconds (default: 100000).

    """
    if len(data.shape) > 2:
        raise ValueError('Cannot write data with more than 2 dimensions.')
    if len(data.shape) == 1:
        tmp = data[np.newaxis, :]
    else:
        tmp = data
    if tmp.dtype is not np.float32:
        tmp = tmp.astype(np.float32)
    header = pack('>IIHH', tmp.shape[0], samp_period, 4*tmp.shape[1], USER)
    tmp = tmp.flatten(order='C')
    barray = bytearray()
    for value in tmp:
        barray += pack('>f', value)
    with open(path, 'wb') as file_obj:
        file_obj.write(header)
        file_obj.write(barray)


def __is_start_or_end_field(field):
    return field.isdigit()


def __is_score_field(field):
    retval = True
    try:
        float(field)
    except ValueError:
        retval = False
    return retval


def __read_htk_labels(lines, samp_period):
    # pylint: disable=too-many-branches
    # Parsing a file needs many branches.

    # Even though it is not written in the documentation the source code of
    # HTK 3.4.1 assume ';' as comment delimiter.
    comment_delimiter = ';'
    segmentation = []
    for line in lines:
        line = line.strip()
        if line == '///':
            raise NotImplementedError('Unsupported: multiple segmentation in '
                                      'HTK label files.')
        tokens = line.split()
        # Discard empty line
        if len(tokens) == 0:
            continue
        start_end = [-1, -1]
        name = None
        score = None
        aux = []
        auxname = None
        for i, token in enumerate(tokens):
            if i > 2 and name is None:
                raise ValueError('File is badly formatted.')
            if token == comment_delimiter:
                break
            if name is None:
                if i < 2 and __is_start_or_end_field(token):
                    start_end[i] = int(int(token)/samp_period)
                    if i == 1 and start_end[0] == -1:
                        raise ValueError('Invalid start time.')
                else:
                    name = token
            else:
                if __is_score_field(token):
                    if score is None:
                        score = float(token)
                    elif auxname is not None:
                        aux.append((auxname, float(token)))
                        auxname = None
                    else:
                        raise ValueError('No auxiliary name is matching the '
                                         'auxiliary score.')
                else:
                    if auxname is not None:
                        aux.append((auxname, 0.0))
                    auxname = token

        if auxname is not None:
            aux.append((auxname, 0.0))
        if name is None and (start_end[0] != -1 or start_end[1] != -1):
            raise ValueError('Incomplete line.')
        tup = (name, start_end[0], start_end[1], score, aux)
        segmentation.append(tup)
    return segmentation


def __write_htk_labels(file_obj, entries, samp_period=1):
    for entry in entries:
        (name, start, end, _, aux) = entry
        if start is not None:
            print(int(start*samp_period), end=' ', file=file_obj)
            if end is not None:
                print(int(end*samp_period), end=' ', file=file_obj)
        print(name, end=' ', file=file_obj)
        if aux is not None:
            for auxname, auxscore in aux:
                print(auxname, end=' ', file=file_obj)
                print(auxscore, end=' ', file=file_obj)
        print(file=file_obj)


def read_htk_labels(path, samp_period=100000):
    """Read HTK label files.

    Read a HTK label file according to the specification given
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node82.html>`_.

    Parameters
    ----------
    path : str
        Path to the features HTK file.
    sampPeriod : int
        The sampling period in 100ns (default is 1e5).

    Returns
    -------
    r : list
        A list of list of tuple (name, start, end, aux)

    """
    with open(path, 'r') as file_obj:
        return __read_htk_labels(file_obj, samp_period)


def write_htk_labels(out, entries, samp_period=100000):
    """Write the entries as a HTK label file.

    The output file is formatted according to the specification given
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node82.html>`_.

    Parameters
    ----------
    out : str
        Path to the features HTK file or and opened file object.
    entries : list
        A list of tuple (name, start, end, aux). All the values but
        'name' can be set to None.
    samp_period : int
        The sampling period in 100ns (default is 1e5).

    """
    if isinstance(out, str):
        with open(out, 'w') as file_obj:
            __write_htk_labels(file_obj, entries, samp_period)
    else:
        __write_htk_labels(out, entries, samp_period)


def read_timit_labels(path, samp_period=100000):
    """Read TIMIT label files.

    Read a TIMIT label file as defined
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node82.html>`_.
    The sampling rate is assumed to be 16kHz and is fixed.

    Parameters
    ----------
    path : str
        Path to the features HTK file.
    samp_period : int
        The sampling period in 100ns (default is 1e5).

    Returns
    -------
    dicts : list
        A list of tuple (name, start, end, None, None).

    Notes
    -----
    The 'None' values at the end of the tuples is here for compatibility
    with :method:`read_htk_label`

    """

    # We assume 16000 samples/seconds for TIMIT data
    factor = (1/16000.)*(1e7/samp_period)
    segmentation = []
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            tokens = line.split()
            if len(tokens) == 0:
                continue
            if len(tokens) != 3:
                raise ValueError('File is badly formatted.')
            start = int(int(tokens[0])*factor)
            end = int(int(tokens[1])*factor)
            tup = (tokens[2], start, end, None, None)
            segmentation.append(tup)
    return segmentation


def read_mlf(path, samp_period=100000):
    """Read a Master Label File.

    Read a Master Label File (MLF) as defined
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node86.html>`_.
    Only the immediate transcription is supported.

    Parameters
    ----------
    path : str
        Path to the file.
    samp_period : int
        The sampling period in 100ns (default is 1e5).

    """
    header = 0
    mlfdef = 1
    trans = 2
    retval = {}
    with open(path, 'r') as file_obj:
        transcription = []
        state = header
        name = None
        for line in file_obj:
            line = line.strip()

            tokens = line.split()
            if len(tokens) == 0:
                continue

            if state == header:
                if not line == '#!MLF!#':
                    raise ValueError('Invalid MLF header')
                else:
                    state = mlfdef
            elif state == mlfdef:
                tokens = line.split()
                if len(tokens) == 1:
                    name = tokens[0]
                    name = name.replace('"', '')
                    name = os.path.basename(name)
                    name, _ = os.path.splitext(name)
                    state = trans
                    transcription = []
                    continue
                elif len(tokens) == 3:
                    raise ValueError('Search in subdirectory defined in MLF '
                                     'is not supported.')
                else:
                    raise ValueError('Invalid MLF definition.')
            elif state == trans:
                if line != '.':
                    transcription.append(line)
                else:
                    retval[name] = __read_htk_labels(transcription,
                                                     samp_period)
                    state = mlfdef
    return retval


def __write_mlf(file_obj, data, samp_period, header):
    if header:
        print('#!MLF!#', file=file_obj)
    for key in data.keys():
        print('"*/'+str(key)+'"', file=file_obj)
        __write_htk_labels(file_obj, data[key], samp_period)
        print('.', file=file_obj)


def write_mlf(out, data, samp_period=100000, header=True):
    """Write a Master Label File.

    Write a Master Label File (MLF) as defined
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node86.html>`_.
    Only the immediate transcription is supported.

    Parameters
    ----------
    out : str
        Path to the features HTK file or and opened file object.
    data : dict
        Dictionary of list of tuples (name, start, end, aux).
    samp_period : int
        The sampling period in 100ns (default is 1e5).
    header : boolean
        Write the MLF header (default True).

    """
    if isinstance(out, str):
        with open(out, 'w') as file_obj:
            __write_mlf(file_obj, data, samp_period, header)
    else:
        __write_mlf(out, data, samp_period, header)


def read_ctm(fpt, pos=(0, 1, 2, 3), has_duration=False, file_transform=None,
             blacklist=(), add_bool=False, offset_from_filename=None,
             file_segments=None):
    """ read a ctm file

    :param fpt: file pointer to read ctm file to from
    :param pos: 4 element tuple with positions of (filename, word, start, end)
    :param has_duration: ctm uses duration instead of end time
    :param file_transform: transform function to modify filename
    :param blacklist: blacklist of words to skip when reading
    :param add_bool: add boolen to ctm transcription
    :param offset_from_filename: function to derive time offset from filename
    :param file_segments: tuples containing (filename, start, end) of segments
                          ctm will be created with 'filename_start_end' as key
    :return: dict with transcription and timings per file

    Example for ctm with add_bool=True
    reference_ctm['c1lc021h'] = \
        [('MR.', 0.91, 1.26, True),
         ('MAUCHER', 1.26, 1.6, True)]
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # Not sure this code will remain.

    ctm = dict()
    for line in fpt:
        split_line = line.split()
        filename = split_line[pos[0]]
        word = split_line[pos[1]]
        start = float(split_line[pos[2]])
        end = float(split_line[pos[3]])

        if has_duration:
            end += start

        if offset_from_filename is not None:
            offset = offset_from_filename(filename)
            start += offset
            end += offset

        if file_transform is not None:
            filename = file_transform(filename)

        if word not in blacklist:
            if add_bool:
                entry = (word, start, end, False)
            else:
                entry = (word, start, end)

            if filename in ctm:
                ctm[filename].append(entry)
            else:
                ctm[filename] = [entry]

    ctm = {filename: sorted(entry, key=operator.itemgetter(1))
           for filename, entry in ctm.items()}

    if file_segments is not None:
        segments_ctm = dict()
        for file, start, end in file_segments:
            filename = '{}_{:06d}-{:06d}'.format(file,
                                                 int(round(start*1000)),
                                                 int(round(end*1000)))
            segments_ctm[filename] = [entry for entry in ctm[file]
                                      if entry[1] >= start and entry[2] <= end]

        return segments_ctm

    return ctm


def write_eval_to_clusters(ctm, fpt, file_transform=None,
                           write_sequence=False):
    """  Write cluster file for evaluation with https://github.com/bootphon/tde

    :param ctm: ctm file
    :param fpt: output file pointer
    :param file_transform: transform function to modify filename
    :param write_sequence: write out unit sequence for cluster
    """
    clusters = dict()
    for file, sentence in ctm.items():
        for word in sentence:
            if word[0] not in clusters:
                clusters[word[0]] = list()

            if file_transform is not None:
                clusters[word[0]].append((file_transform(file), ) + word[1:])
            else:
                clusters[word[0]].append((file, ) + word[1:])

    for idx, (sequence, words) in enumerate(clusters.items()):
        if write_sequence:
            fpt.write('Class {} [{}]\n'.format(idx, sequence))
        else:
            fpt.write('Class {}\n'.format(idx))
        for word in words:
            fpt.write('{} {} {}\n'.format(*word))

        fpt.write('\n')

