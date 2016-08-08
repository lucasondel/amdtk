
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
import gzip
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


def readHtk(path, infos=False):
    """Read HTK binary file.

    Read binary file according to the specification given
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node58.html>`_.

    Parameters
    ----------
    path : str
        Path to the features HTK file.
    infos : boolean
        If "infos" is True, then the header informations are returned

    Returns
    -------
    data : numpy.ndarray
        The data as a matrix.
    data, info: numpy.ndarray, tuple
        The data as a matrix and a tuple (nSamples, sampPeriod, sampSize)

    """

    with open(path, 'rb') as f:
        header_size = 12
        header = f.read(header_size)
        nSamples, sampPeriod, sampSize, parmKind = \
            unpack('>IIHH', header)
        if parmKind & _C:
            size = int(sampSize/2)
            dtype = 'h'
            if parmKind & 0x3F == IREFC:
                A = 32767.
                B = 0.
                offset = 0
            else:
                offset = 2*size*4
                f.seek(header_size)
                A = np.fromfile(f, dtype='>f', count=size)
                B = np.fromfile(f, dtype='>f', count=size)
        else:
            size = int(sampSize/4)
            dtype = 'f'
            offset = 0

        f.seek(header_size + offset)
        data = np.fromfile(f, dtype='>'+dtype)
        if parmKind & _K:
            data = data[:-1]
        new_shape = (int(len(data)/size), size)
        data = data.reshape(new_shape)
        if parmKind & _C:
            data = (data + B) / A
        if not infos:
            return data
        return data, (nSamples, sampPeriod, sampSize)


def writeHtk(path, data, sampPeriod=100000):
    if len(data.shape) > 2:
        raise ValueError('Cannot write data with more than 2 dimensions.')
    if len(data.shape) == 1:
        tmp = data[np.newaxis, :]
    else:
        tmp = data
    if tmp.dtype is not np.float32:
        tmp = tmp.astype(np.float32)
    header = pack('>IIHH', tmp.shape[0], sampPeriod, 4*tmp.shape[1], USER)
    tmp = tmp.flatten(order='C')
    s = bytearray()
    for value in tmp:
        s += pack('>f', value)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(s)


def __isStartOrEndField(field):
    return field.isdigit()


def __isScoreField(field):
    try:
        float(field)
    except ValueError:
        return False
    return True


def __readHtkLabels(lines, sampPeriod):
    # Even though it is not written in the documentation the source code of
    # HTK 3.4.1 assume ';' as comment delimiter.
    comment_delimiter = ';'
    segmentation = []
    for lineno, line in enumerate(lines):
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
                if i < 2 and __isStartOrEndField(token):
                    start_end[i] = int(int(token)/sampPeriod)
                    if i == 1 and start_end[0] == -1:
                        raise ValueError('Invalid start time.')
                else:
                    name = token
            else:
                if __isScoreField(token):
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
        t = (name, start_end[0], start_end[1], score, aux)
        segmentation.append(t)
    return segmentation


def __writeHtkLabels(f, entries, sampPeriod=1):
    for entry in entries:
        (name, start, end, score, aux) = entry
        if start is not None:
            print(int(start*sampPeriod), end=' ', file=f)
            if end is not None:
                print(int(end*sampPeriod), end=' ', file=f)
        print(name, end=' ', file=f)
        if aux is not None:
            for auxname, auxscore in aux:
                print(auxname, end=' ', file=f)
                print(auxscore, end=' ', file=f)
        print(file=f)


def readHtkLabels(path, sampPeriod=100000):
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
    with open(path, 'r') as f:
        return __readHtkLabels(f, sampPeriod)


def writeHtkLabels(out, entries, sampPeriod=100000):
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
    sampPeriod : int
        The sampling period in 100ns (default is 1e5).

    """
    if isinstance(out, str):
        with open(out, 'w') as f:
            __writeHtkLabels(f, entries, sampPeriod)
    else:
        __writeHtkLabels(out, entries, sampPeriod)


def readTimitLabels(path, sampPeriod=100000):
    """Read TIMIT label files.

    Read a TIMIT label file as defined
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node82.html>`_.
    The sampling rate is assumed to be 16kHz and is fixed.

    Parameters
    ----------
    path : str
        Path to the features HTK file.
    sampPeriod : int
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
    factor = (1/16000.)*(1e7/sampPeriod)
    segmentation = []
    with open(path, 'r') as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            tokens = line.split()
            if len(tokens) == 0:
                continue
            if len(tokens) != 3:
                raise ValueError('File is badly formatted.')
            start = int(int(tokens[0])*factor)
            end = int(int(tokens[1])*factor)
            t = (tokens[2], start, end, None, None)
            segmentation.append(t)
    return segmentation


def __parseTiming(token, lineno):
    if '[' in token:
        if ']' in token:
            i1 = token.index('[')
            i2 = token.index(']')
            timing = token[i1+1:i2]
            tokens = timing.split(',')
            if len(tokens) != 2:
                raise ValueError('Wrong timing format.')
            start = int(tokens[0])
            end = int(tokens[1])
            source = token[:i1]
        else:
            raise ValueError('Missing "]".')
    else:
        start = -1
        end = -1
        source = token
    return source, start, end


def readMlf(path, sampPeriod=100000):
    """Read a Master Label File.

    Read a Master Label File (MLF) as defined
    `here <http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node86.html>`_.
    Only the immediate transcription is supported.

    Parameters
    ----------
    path : str
        Path to the file.
    sampPeriod : int
        The sampling period in 100ns (default is 1e5).

    """
    HEADER = 0
    MLFDEF = 1
    TRANSCRIPTION = 2
    retval = {}
    with open(path, 'r') as f:
        transcription = []
        state = HEADER
        name = None
        for lineno, line in enumerate(f):
            line = line.strip()

            tokens = line.split()
            if len(tokens) == 0:
                continue

            if state == HEADER:
                if not line == '#!MLF!#':
                    raise ValueError('Invalid MLF header')
                else:
                    state = MLFDEF
            elif state == MLFDEF:
                tokens = line.split()
                if len(tokens) == 1:
                    name = tokens[0]
                    name = name.replace('"', '')
                    name = os.path.basename(name)
                    name, ext = os.path.splitext(name)
                    state = TRANSCRIPTION
                    transcription = []
                    continue
                elif len(tokens) == 3:
                    raise ValueError('Search in subdirectory defined in MLF '
                                     'is not supported.')
                else:
                    raise ValueError('Invalid MLF definition.')
            elif state == TRANSCRIPTION:
                if line != '.':
                    transcription.append(line)
                else:
                    retval[name] = __readHtkLabels(transcription, sampPeriod)
                    state = MLFDEF
    return retval


def __write_mlf(f, data, sampPeriod, header):
    if header:
        print('#!MLF!#', file=f)
    for key in data.keys():
        print('"*/'+str(key)+'"', file=f)
        __writeHtkLabels(f, data[key], sampPeriod)
        print('.', file=f)


def writeMlf(out, data, sampPeriod=100000, header=True):
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
    sampPeriod : int
        The sampling period in 100ns (default is 1e5).
    header : boolean
        Write the MLF header (default True).

    """
    if isinstance(out, str):
        with open(out, 'w') as f:
            __write_mlf(f, data, sampPeriod, header)
    else:
        __write_mlf(out, data, sampPeriod, header)

def readHtkLattice(lattfile, ac_weight=1., lm_weight=1., gzipped=True):
    """Read a HTK lattice file and represent it as a OpenFst object.

    Parameters
    ----------
    lattfile : string
        Path to the HTK lattice file.
    ac_weight : float
        Acoustic weight (default: 1).
    lm_weight : float
        Language model weight (default : 1).
    gzipped : boolean
        If the lattice file is compressed with gzip (default: True).

    Returns
    -------
    fst_lattice : pywrapfst.Fst
        The lattice as an OpenFst object.
    
    """
    # Make the import here only as some people may not have openfst installed.
    import pywrapfst as fst

    # Output fst.
    fst_lattice = fst.Fst("log")

    # The mapping id -> label (and the reverse one) are relative to the
    # lattice only. It avoids to have some global mapping.
    label2id = {"!NULL": 0}
    id2label = {0: "!NULL"}
    identifier = 0

    # If the lattice is compressed with gzip choose the correct function.
    if gzipped:
        my_open = gzip.open
        mode = 'rt'
    else:
        my_open = open
        mode = 'r'

    with my_open(lattfile, mode) as f:
        for line in f:
            line = line.strip()

            # Get the number of nodes. If the field is not specified then 
            # the function will fail badly.
            if line[:2] == "N=":
                n_nodes = int(line.split()[0].split("=")[-1])

                # Create all the nodes of the fst lattice.
                for i in range(n_nodes):
                    fst_lattice.add_state()
                fst_lattice.set_start(0)
                fst_lattice.set_final(n_nodes-1)

            elif line[0] == "J":
                # Load the HTK arc definition. 
                fields = line.split()
                start_node = int(fields[1].split("=")[-1])
                end_node = int(fields[2].split("=")[-1])
                label = fields[3].split("=")[-1]

                # Update the mapping id -> label and its reverse.
                try:
                    label_id = label2id[label]
                except KeyError:
                    identifier += 1
                    label_id = identifier
                    label2id[label] = label_id
                    id2label[label_id] = label

                # Add the arc to the FST lattice.
                ac_score = float(fields[5].split("=")[-1])
                lm_score = float(fields[6].split("=")[-1])
                score = ac_weight * ac_score + lm_weight * lm_score
                weight = fst.Weight(fst_lattice.weight_type, -score)
                arc = fst.Arc(label_id, label_id, weight,
                              end_node)
                fst_lattice.add_arc(start_node, arc)
    return fst_lattice, id2label
    
