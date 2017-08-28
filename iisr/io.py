"""
Contain tools to read IISR data files, *.ISE and *.IST.
Read experiment realizations - received quadratures with corresponding parameters.
"""

import contextlib
import gzip
import itertools as it
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
from bitstring import ConstBitStream, ReadError

from iisr.representation import SignalBlock, SignalTimeSeries, Parameters
from iisr.representation import CHANNELS_NUMBER_INFO

__all__ = ['read', 'read_files_by_blocks', 'read_files_by_series']
FILE_EXTENSIONS = ('.ISE', '.ISE.GZ', '.IST', '.IST.GZ')
KEYWORD = b'ORDA'
BYTEORDER = 'little'
HEADER_CODES = {'super': 1, 'data': 2, 'global': 3}
RAW_PARAMETERS_CODES = (
    'reserved',
    'mode',             # mode
    'step',             # decimation step
    'number_all',       # number of samples
    'number_after',     # number of samples after decimation
    'first_delay',      # first sample delay relative to Tk0, us
    'freq_code',        # frequency code
    'channel',
    'data_type',
    'date_year',
    'date_mon_day',     # date: month (2-nd byte) / day (1-st byte)
    'time_h_m',         # time: hour (1-st byte) / minute (2-nd byte)
    'time_sec',         # time: sec
    'time_msec',        # time: msec
    'st1_long_fr_lo',   # STEL1 long pulse frequency, 1-2 bytes
    'st1_long_fr_hi',   # STEL1 long pulse frequency, 3-4 bytes
    'st1_short_fr_lo',  # STEL1 short pulse frequency, 1-2 bytes
    'st1_short_fr_hi',  # STEL1 short pulse frequency, 3-4 bytes
    'st2_long_fr_lo',   # STEL2 long pulse frequency, 1-2 bytes
    'st2_long_fr_hi',   # STEL2 long pulse frequency, 3-4 bytes
    'st2_short_fr_lo',  # STEL2 short pulse frequency, 1-2 bytes
    'st2_short_fr_hi',  # STEL2 short pulse frequency, 3-4 bytes
    'st1_long_len',     # STEL1 long pulse length
    'st1_short_len',    # STEL1 short pulse length
    'st2_long_len',     # STEL2 long pulse length
    'st2_short_len',    # STEL2 short pulse length
    'st1_long_phase',   # STEL1 long pulse phase modulation
    'st1_short_phase',  # STEL1 short pulse phase modulation
    'st2_long_phase',   # STEL2 long pulse phase modulation
    'st2_short_phase',  # STEL2 short pulse phase modulation
    'sample_freq',      # sampling frequency, kHz
    'average',          # steady component mean
    'phase_code',       # code of phase manipulation
    'offset_st1',       # bias
    'timer_lo',         # timer 1-2 bytes (unused here)
    'timer_hi',         # timer 3-4 bytes (unused here)
    'version'           # ISE file version
)


def read_files_by_blocks(paths):
    """
    Read all data files using given paths.

    Parameters
    ----------
    paths: str or list of str
        Paths to data files. May contain directory paths and file paths.
        Function does not consider file in subdirectories.

    Yields
    -------
    data_block: DataBlock
        Block of realizations corresponding to the same time mark.
    """
    file_paths = _collect_valid_file_paths(paths)
    for path in file_paths:
        yield from _read_file_by_signal_blocks(path)


def read_files_by_series(paths):
    """
    Read all data files using given paths.

    Parameters
    ----------
    paths: str or list of str
        Paths to data files. May contain directory paths and file paths.
        Function does not consider file in subdirectories.

    Yields
    -------
    time_series: SignalTimeSeries
        Annotated single realization.
    """
    file_paths = _collect_valid_file_paths(paths)
    for path in file_paths:
        yield from read(path)


def _collect_valid_file_paths(paths):
    """
    Search through given paths to create united file list of IISR data files.

    Parameters
    ----------
    paths: str or list of str
        Paths to data files. May contain directory paths and file paths.
        Function does not consider file in subdirectories.

    Returns
    -------
    files_paths: list of str
    """
    if isinstance(paths, str):
        paths = [paths]

    # Get list of all files paths
    files_paths = []

    def check_and_add_path(new_path):
        extension = os.path.splitext(new_path)[-1].upper()
        if extension in FILE_EXTENSIONS:
            files_paths.append(new_path)

    for path in paths:
        if os.path.isfile(path):
            check_and_add_path(path)
        elif os.path.isdir(path):
            for file_in_dir in sorted(os.listdir(path)):
                check_and_add_path(file_in_dir)
    return files_paths


def _read_file_by_signal_blocks(path, only_headers=False):
    """
    Read IISR data file and return signal blocks - groups of signal series that
    belong to identical time.

    Parameters
    ----------
    path: str
        Path to file.
    only_headers: bool, default False
        If True return only annotation of time series, leaving quadratures field of the
        SignalTimeSeries instance as None.

    Yields
    -------
    signal_block: SignalBlock
        Block of realizations corresponding to the same time mark.
    """
    time_series_generator = read(path, only_headers=only_headers)

    def grouping_condition(series):
        return series.time_mark

    grouped_time_series = it.groupby(time_series_generator, key=grouping_condition)
    for unique_time_mark, time_series_group in grouped_time_series:
        yield SignalBlock(unique_time_mark, list(time_series_group))


@contextlib.contextmanager
def _open_data_file(path):
    """
    Open IISR datafile. If file is compressed, creates a temporal file.

    Parameters
    ----------
    path: str
        Path to file.

    Returns
    -------
    file_steam: stream
    """
    compressed = path.endswith('.gz')
    if compressed:
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(gzip.decompress(open(path, 'rb').read()))
            working_path = file.name
    else:
        working_path = path

    try:
        with open(working_path, 'rb') as file_stream:
            yield file_stream

    finally:
        # Remove temporal file
        if compressed:
            os.remove(working_path)


def read(file_path, only_headers=False):
    """
    Read headers of iisr datafiles.

    Parameters
    ----------
    file_path: str
        Path to file.
    only_headers: bool, default False
        If True return only annotation of time series, leaving quadratures field of the
        SignalTimeSeries instance as None.

    Yields
    -------
    time_series: SignalTimeSeries
    """

    # Read blocks until file ends. Headers of blocks contain header code and block length
    with _open_data_file(file_path) as file_stream:
        # Wait for global block. It must be first
        while True:
            header_code, block_length = _handle_header(file_stream)

            if header_code == HEADER_CODES['global']:
                global_parameters = _handle_raw_parameters_block(file_stream, block_length)
                break
            elif header_code is None:
                raise ReadError('Global header not found in file {}.'.format(file_path))
            else:
                # Skip
                pass

        # Reading remaining data
        while True:
            # Read super block, the data annotation
            header_code, block_length = _handle_header(file_stream)
            if header_code == HEADER_CODES['super']:
                super_parameters = _handle_raw_parameters_block(file_stream, block_length)
            elif header_code is None:
                break
            else:
                raise ReadError('Incorrect code [{}] in file {} (super code {} expected)'
                                ''.format(header_code, file_path, HEADER_CODES['super']))

            # Read data block. It must come after each super header
            header_code, block_length = _handle_header(file_stream)

            if header_code == HEADER_CODES['data']:
                data_address = file_stream.tell()
                data_length = block_length
            elif header_code is None:
                break
            else:
                raise ReadError('Incorrect code [{}] in file {} (data code {} expected)'
                                ''.format(header_code, file_path, HEADER_CODES['data']))

            if not only_headers:
                quadratures = read_quadratures(file_stream, data_address, data_length)
            else:
                quadratures = None

            # Create annotated signal time series (realization)
            raw_parameters = global_parameters.copy()
            raw_parameters.update(super_parameters)
            print(raw_parameters)
            time_mark, parameters = raw2refined_parameters(raw_parameters, data_length)
            time_series = SignalTimeSeries(time_mark, parameters, quadratures)
            yield time_series


def _handle_header(stream):
    """
    Search for keyword in data stream.

    Parameters
    ----------
    stream: opened file stream

    Returns
    -------
    code: int
        Code of header.
    block_length: int
        Length of next block, bytes.
    """
    piece = stream.read(9)
    key_index = piece.find(KEYWORD)

    # If found in first position.
    if key_index == 0:
        code = piece[4]
        block_length = int.from_bytes(piece[5:], byteorder=BYTEORDER)
        return code, block_length

    # If found, but not in first position, change position in stream
    elif key_index > 0:
        stream.seek(key_index + 4)

    # Not found in given piece, launch fast stream search.
    else:
        # bitstring provide fast keyword search (not loading all stream in mem)
        # position drop to 0 in stream when creating ConstBitStream
        bytepos = stream.tell()
        bitstring_stream = ConstBitStream(stream)
        bitstring_stream.bytepos = bytepos
        try:
            bitstring_stream.readto(KEYWORD, bytealigned=True)
        except ReadError:
            return None, None

        stream.seek(bitstring_stream.bytepos)

    # Read after position was found
    code = int.from_bytes(stream.read(1), byteorder=BYTEORDER)
    block_length = int.from_bytes(stream.read(4), byteorder=BYTEORDER)
    return code, block_length


def _handle_raw_parameters_block(stream, block_length):
    """
    Read raw parameters of time series from stream.

    Parameters
    ----------
    stream: opened file stream
    block_length: int
        Length of block in bytes.

    Returns
    -------
    parameters: dict
        Raw parameters of time series.
    """
    parameters = {}

    # Each parameter represented by 2 bytes for code and 2 bytes for value
    n_parameters = block_length // 4
    for _ in range(0, n_parameters):
        bin_code = stream.read(2)
        bin_param = stream.read(2)

        if bin_code == b'' or bin_param == b'':
            raise ReadError('EOF when trying to read parameter')

        code = int.from_bytes(bin_code, byteorder=BYTEORDER)
        value = int.from_bytes(bin_param, byteorder=BYTEORDER)

        # Searching for parameter code in pre-defined tuple
        try:
            parameters[RAW_PARAMETERS_CODES[code]] = value
        except KeyError:
            parameters['undefined_parameter_{}'.format(code)] = value

    return parameters


def read_quadratures(file_stream, data_address, data_byte_length):
    """
    Read quadratures from file_stream given address and length.

    Parameters
    ----------
    file_stream: stream
        File stream. Must implement seek and read.
    data_address: int
        Address of quadratures in the stream.
    data_byte_length: int
        Length of quadratures to be read.

    Returns
    -------
    quadratures: np.ndarray of complex numbers.
    """
    file_stream.seek(data_address)

    quadrature_byte_length = data_byte_length // 2
    quadrature_size = quadrature_byte_length // 2

    # Read consequent I and Q samples. Repeat quadrature_size times.
    dtype = np.dtype([
        ('quad_I', '<i2', quadrature_size),
        ('quad_Q', '<i2', quadrature_size)
    ])

    quadratures = np.fromfile(file_stream, dtype=dtype, count=1)

    return np.array(quadratures['quad_I'][0]) + 1j * np.array(quadratures['quad_Q'][0])


def _check_raw_parameters(raw_parameters):
    """Check raw parameters for validness"""
    with_st1 = False
    with_st2 = False
    for key, value in raw_parameters.items():
        if 'st1' in key and value != 0:
            with_st1 = True
        elif 'st2' in key and value != 0:
            with_st2 = True

    if with_st2 and not with_st1:
        raise NotImplementedError('Processing STEL2 is not implemented')
    elif with_st1 and with_st2:
        raise NotImplementedError('Non-zero st1 and st2 fields')
    elif not with_st1 and not with_st2:
        raise RuntimeError('Raw parameters miss key parameters')


def raw2refined_parameters(raw_parameters, data_byte_length):
    """
    Process raw parameters of IISR data files to get convenient parameters and time.

    Parameters
    ----------
    raw_parameters: dict
        Dictionary of parameters from raw files.
    data_byte_length: int
        Length of corresponding data block in bytes.

    Returns
    -------
    time_mark: datetime
        Time of observation.
    refined_parameters: Parameters
        Parameters of experiment.
    """
    _check_raw_parameters(raw_parameters)

    n_samples = data_byte_length // 4  # two quadratures
    decimation = raw_parameters.pop('number_all') / n_samples
    sampling_frequency = raw_parameters.pop('sample_freq') / decimation

    channel = raw_parameters.pop('channel')
    first_delay = raw_parameters.pop('first_delay')
    offset_st1 = raw_parameters.pop('offset_st1')
    pulse_type = CHANNELS_NUMBER_INFO[channel]['type']

    fr_lo = raw_parameters.pop('st1_{}_fr_lo'.format(pulse_type))
    fr_hi = raw_parameters.pop('st1_{}_fr_hi'.format(pulse_type))
    frequency_kHz = fr_lo + (fr_hi << 16)
    frequency_MHz = frequency_kHz / 1000.

    pulse_length_us = raw_parameters.pop('st1_{}_len'.format(pulse_type))

    # For 900 us pulses, there is no room for short pulse and only channel noise
    # was recorded
    if pulse_type is 'short' and pulse_length_us == 0:
        pulse_type = 'noise'

    if pulse_type is 'long':
        long_pulse_len = pulse_length_us
    else:
        long_pulse_len = 0

    # Magic vague formula to calculate total delay
    total_delay = first_delay - 960 - offset_st1 - 50 - long_pulse_len

    # Time
    month_day = raw_parameters.pop('date_mon_day')
    hour_min = raw_parameters.pop('time_h_m')

    year = raw_parameters.pop('date_year')
    second = raw_parameters.pop('time_sec')
    millisecond = raw_parameters.pop('time_msec')

    month = month_day >> 8
    day = month_day & 0x00FF
    minute = hour_min >> 8
    hour = hour_min & 0x00FF

    # It appears sometimes *.ISE files have millisecond >= 1000
    residual_time = timedelta()
    if millisecond >= 1000:
        delta = int(millisecond - 999)  # delta should be int
        millisecond = 999
        residual_time = timedelta(microseconds=delta * 1000)

    time = datetime(
        year=year, month=month, day=day, hour=hour, minute=minute,
        second=second, microsecond=millisecond * 1000
    ) + residual_time

    # Form output
    parameters = Parameters()
    parameters.sampling_frequency = sampling_frequency
    parameters.pulse_length_us = pulse_length_us
    parameters.pulse_type = pulse_type
    parameters.n_samples = n_samples
    parameters.channel = channel
    parameters.phase_code = raw_parameters.pop('phase_code')
    parameters.frequency_MHz = frequency_MHz
    parameters.total_delay = total_delay

    parameters.rest_raw_parameters = raw_parameters
    return time, parameters


def refined2raw_parameters(time_mark, refined_parameters, default_first_delay=2150,
                           default_offset_st1=80, decimation=2):
    """
    Build raw parameters dictionary from refined parameters.

    Parameters
    ----------
    time_mark: datetime
    refined_parameters: Parameters
    default_first_delay: int
        Default value of first delay. Applied if first delay is not in refined_parameters.
    default_offset_st1: int
        Default value of receiver offset. Applied if offset is not in refined_parameters.
    decimation: int
        Decimation used during observation.

    Returns
    -------
    raw_parameters: dict
    data_byte_length: int
        Length of data block.
    """
    # Reclaim unused raw_parameters
    if refined_parameters.rest_raw_parameters is None:
        raw_parameters = {}
    else:
        raw_parameters = refined_parameters.rest_raw_parameters.copy()

    pulse_type = refined_parameters.pulse_type

    if pulse_type == 'noise':
        pulse_type = 'short'

    raw_parameters['number_all'] = refined_parameters.n_samples * decimation
    data_byte_length = refined_parameters.n_samples * 4
    raw_parameters['sample_freq'] = refined_parameters.sampling_frequency * decimation

    if 'first_delay' not in raw_parameters:
        raw_parameters['first_delay'] = default_first_delay

    if 'offset_st1' not in raw_parameters:
        raw_parameters['offset_st1'] = default_offset_st1

    raw_parameters['channel'] = refined_parameters.channel
    raw_parameters['phase_code'] = refined_parameters.phase_code

    # Time parameters
    raw_parameters['date_year'] = time_mark.year
    month = time_mark.month
    day = time_mark.day
    raw_parameters['date_mon_day'] = (month << 8) + day
    hour = time_mark.hour
    minute = time_mark.minute
    raw_parameters['time_h_m'] = (hour << 8) + minute
    raw_parameters['time_sec'] = time_mark.second
    raw_parameters['time_msec'] = time_mark.microsecond // 1000

    # Frequency
    frequency_kHz = int(refined_parameters.frequency_MHz * 1000)
    fr_lo = frequency_kHz & 0x00FF
    fr_hi = frequency_kHz >> 16
    raw_parameters['st1_{}_fr_lo'.format(pulse_type)] = fr_lo
    raw_parameters['st1_{}_fr_hi'.format(pulse_type)] = fr_hi

    raw_parameters['st1_{}_len'.format(pulse_type)] = refined_parameters.pulse_length_us

    return raw_parameters, data_byte_length


def write(file_path, raw_parameters, data_block_length, quadratures):
    """
    Write IISR data file.

    Parameters
    ----------
    file_path: str
        Path to file.
    raw_parameters: dict
    data_block_length: int
    quadratures: np.ndarray
    """
    raw_parameters = raw_parameters.copy()

    # Global header
    global_parameters_names = ['number_all', 'first_delay', 'offset_st1', 'number_after',
                               'mode', 'step', 'sample_freq', 'version']
    global_parameters = {}
    for name in global_parameters_names:
        if name in raw_parameters:
            global_parameters[name] = raw_parameters.pop(name)

    global_block, global_block_length = _create_raw_parameters_block(global_parameters)
    global_header = _create_header(HEADER_CODES['global'], global_block_length)

    super_block, super_block_length = _create_raw_parameters_block(raw_parameters)
    super_header = _create_header(HEADER_CODES['super'], super_block_length)

    quads_i = (int(number) for number in quadratures.real)
    quads_q = (int(number) for number in quadratures.imag)

    quads_i_bytes = (number.to_bytes(2, BYTEORDER, signed=True) for number in quads_i)
    quads_q_bytes = (number.to_bytes(2, BYTEORDER, signed=True) for number in quads_q)

    # data_block = b''.join(map(lambda x: b''.join(x), zip(quads_i_bytes, quads_q_bytes)))
    data_block = b''.join(it.chain(quads_i_bytes, quads_q_bytes))
    # print(data_block)
    # data_block = b''.join(it.chain(quads_i_bytes, quads_q_bytes))
    data_header = _create_header(HEADER_CODES['data'], data_block_length)

    with open(file_path, 'wb') as file_stream:
        file_stream.write(b''.join([global_header, global_block,
                                    super_header, super_block,
                                    data_header, data_block]))


def _create_header(code, block_length):
    return b''.join([KEYWORD,
                     code.to_bytes(1, byteorder=BYTEORDER),
                     block_length.to_bytes(4, byteorder=BYTEORDER)])


def _create_raw_parameters_block(raw_parameters):
    block = []
    for name, value in raw_parameters.items():
        code = RAW_PARAMETERS_CODES.index(name)
        block.append(code.to_bytes(2, byteorder=BYTEORDER))
        block.append(value.to_bytes(2, byteorder=BYTEORDER))

    n_bytes = len(block) * 2
    return b''.join(block), n_bytes
