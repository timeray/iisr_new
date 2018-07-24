"""
Contain tools to read and write IISR data files, *.ISE and *.IST.
Read and write experiment realizations - received quadratures with parameters.

There are two representations for parameters of realizations. First is raw parameters,
which are the options that originate from binary format of the input files. They contain
all necessary information but are verbose and low level. Second representation is
refined parameters that store all important information and easy to work with.

Raw parameters are represented by a dictionary.
Refined parameters are represented by class Parameters.
"""

import contextlib
import gzip
import itertools as it
import os
import tempfile
import struct
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from iisr.representation import TimeSeriesPackage, SignalTimeSeries, Parameters
from iisr.representation import CHANNELS_INFO
from iisr import units

__all__ = ['DataFileReader', 'DataFileWriter', 'open_data_file',
           'read_files_by_series', 'read_files_by_packages',
           'refined2raw_parameters', 'raw2refined_parameters']
FILE_EXTENSIONS = ('.ISE', '.ISE.GZ', '.IST', '.IST.GZ')
DELAY_FORMULA_CONSTANT = -960 - 50
KEYWORD = b'ORDA'
BYTEORDER = 'little'
HEADER_CODES = {'super': 1, 'data': 2, 'global': 3}
# Index corresponds to a code
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


class ReadError(RuntimeError):
    pass


class DataFileReader:
    """Read binary data stream."""
    def __init__(self, stream, file_path: Optional[str] = '', only_headers: Optional[bool] = False,
                 series_filter: Optional[ParameterFilter] = None):
        """Create reader instance.

        Args:
            stream: Input data stream.
            file_path: Path to file. Defaults to empty string.
            only_headers: If True return only annotation of time series,
                leaving quadratures field of SignalTimeSeries instance as None. Defaults to False.
            series_filter: Filter for options. Defaults to None.
        """
        self.stream = stream
        self.only_headers = only_headers
        self.file_path = file_path
        self._series = self._series_generator()
        self.filter = series_filter

    def read_series(self):
        yield from self._series

    def read_blocks(self):
        """
        Return signal blocks - groups of signal series that belong to identical time.

        Yields
        -------
        signal_block: SignalBlock
            Block of realizations corresponding to the same time mark.
        """
        def grouping_condition(series):
            return series.time_mark

        grouped_time_series = it.groupby(self._series, key=grouping_condition)
        for unique_time_mark, time_series_group in grouped_time_series:
            yield TimeSeriesPackage(unique_time_mark, list(time_series_group))

    def __iter__(self):
        return self._series

    def _series_generator(self):
        """Read headers of iisr datafiles.

        Yields:
            time_series: SignalTimeSeries
        """
        # Read blocks until file ends.
        # Headers of blocks contain header code and block length.
        # Read global header block
        header_code, block_length = self._read_header()
        if header_code == HEADER_CODES['global']:
            global_parameters = self._read_raw_parameters_block(block_length)
        # Empty file does not raise error
        elif header_code is None:
            return
        else:
            raise ReadError(
                'Global header is not at the first position of file {} '
                '(get code {} instead of {})'
                ''.format(self.file_path, header_code, HEADER_CODES['global'])
            )

        # Reading remaining data
        while True:
            # Read super block, annotation of data
            header_code, block_length = self._read_header()
            if header_code == HEADER_CODES['super']:
                super_parameters = self._read_raw_parameters_block(block_length)
            # End of file
            elif header_code is None:
                break
            else:
                raise ReadError(
                    'Incorrect code [{}] in file {} (superheader code {} expected)'
                    ''.format(header_code, self.file_path, HEADER_CODES['super']))

            # Read data block. It must come after each super header
            header_code, data_length = self._read_header()

            if header_code == HEADER_CODES['data']:
                data_address = self.stream.tell()
            # End of file
            elif header_code is None:
                break
            else:
                raise ReadError(
                    'Incorrect code [{}] in file {} (data code {} expected)'
                    ''.format(header_code, self.file_path, HEADER_CODES['data']))

            # Form refined options
            raw_parameters = global_parameters.copy()
            raw_parameters.update(super_parameters)
            time_mark, parameters = raw2refined_parameters(raw_parameters, data_length)

            # Check if options pass the filter
            if self.filter is not None:
                if not self.filter.test_parameters(parameters):
                    continue

            if not self.only_headers:
                quadratures = self.read_quadratures(data_address, data_length)
            else:
                quadratures = None

            # Create annotated signal time series (realization)
            time_series = SignalTimeSeries(time_mark, parameters, quadratures)
            yield time_series

    def _read_header(self):
        """
        Read header from input data stream.

        Returns
        -------
        code: int
            Code of header.
        block_length: int
            Length of next block, bytes.
        """
        piece = self.stream.read(9)

        if len(piece) != 9:
            return None, None

        piece = struct.unpack('<4sBI', piece)  # Keyword, code, block length

        # If keyword was found
        if piece[0] == KEYWORD:
            return piece[1], piece[2]
        else:
            raise ReadError('Keyword {} was not found in sequence {}'
                            ' at the position {} of the input stream'
                            ''.format(KEYWORD.decode(), piece, self.stream.tell() - 9))

    def read_quadratures(self, data_address, data_byte_length):
        """
        Read quadratures from file_stream given address and length.

        Parameters
        ----------
        data_address: int
            Address of quadratures in the stream.
        data_byte_length: int
            Length of quadratures to be read.

        Returns
        -------
        quadratures: np.ndarray of complex numbers.
        """
        self.stream.seek(data_address)

        quadrature_byte_length = data_byte_length // 2
        quadrature_size = quadrature_byte_length // 2

        # Read consequent I and Q samples. Repeat quadrature_size times.
        dtype = np.dtype([
            ('quad_I', '<i2', quadrature_size),
            ('quad_Q', '<i2', quadrature_size)
        ])

        quadratures = np.fromfile(self.stream, dtype=dtype, count=1)

        return np.array(quadratures['quad_I'][0]) + 1j * np.array(
            quadratures['quad_Q'][0])

    def _read_raw_parameters_block(self, block_length):
        """
        Read raw options of time series from stream.

        Parameters
        ----------
        block_length: int
            Length of block in bytes.

        Returns
        -------
        options: dict
            Raw options of time series.
        """
        parameters = {}

        # Each parameter represented by 2 bytes for code and 2 bytes for value
        n_parameters = block_length // 4
        for _ in range(0, n_parameters):
            bin_code = self.stream.read(2)
            bin_param = self.stream.read(2)

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


class DataFileWriter:
    def __init__(self, stream):
        self.stream = stream
        self.current_global_header = None

    def write(self, series):
        """
        Write series to IISR file stream.

        Parameters
        ----------
        series: SignalTimeSeries
        """
        raw_parameters, data_byte_length = refined2raw_parameters(series.time_mark,
                                                                  series.parameters)

        # Global header
        global_parameters_names = ['number_all', 'offset_st1', 'number_after',
                                   'mode', 'step', 'sample_freq', 'version']
        global_parameters = {}
        for name in global_parameters_names:
            if name in raw_parameters:
                global_parameters[name] = raw_parameters.pop(name)
        self._write_global_block(global_parameters)
        self._write_super_block(raw_parameters)
        self._write_data_block(series.quadratures, data_byte_length)

    def write_series_package(self, block):
        """

        Parameters
        ----------
        block: TimeSeriesPackage
        """
        for series in block:
            self.write(series)

    def _write_global_block(self, parameters):
        """
        If given options comprises new global header, write it. Otherwise, do nothing.

        Parameters
        ----------
        parameters: dict
            Parameters to write.
        """
        if self.current_global_header == parameters:
            return
        elif self.current_global_header is None:
            self.current_global_header = parameters.copy()
        elif self.current_global_header != parameters:
            self.current_global_header = parameters

        block, block_length = self._get_raw_parameters_block(parameters)
        header = self._get_header(HEADER_CODES['global'], block_length)
        self.stream.write(b''.join([header, block]))

    def _write_super_block(self, parameters):
        block, block_length = self._get_raw_parameters_block(parameters)
        header = self._get_header(HEADER_CODES['super'], block_length)
        self.stream.write(b''.join([header, block]))

    def _write_data_block(self, quadratures, byte_length):
        """
        Write data block.
        Parameters
        ----------
        quadratures: np.ndarray
        byte_length: int
        """

        quads_i = (int(number) for number in quadratures.real)
        quads_q = (int(number) for number in quadratures.imag)

        quads_i_bytes = (number.to_bytes(2, BYTEORDER, signed=True) for number in quads_i)
        quads_q_bytes = (number.to_bytes(2, BYTEORDER, signed=True) for number in quads_q)

        data_block = b''.join(it.chain(quads_i_bytes, quads_q_bytes))
        header = self._get_header(HEADER_CODES['data'], byte_length)
        self.stream.write(b''.join([header, data_block]))

    def _get_header(self, code, block_length):
        return b''.join([KEYWORD, code.to_bytes(1, byteorder=BYTEORDER),
                         block_length.to_bytes(4, byteorder=BYTEORDER)])

    def _get_raw_parameters_block(self, raw_parameters):
        block = []
        for name, value in raw_parameters.items():
            code = RAW_PARAMETERS_CODES.index(name)
            block.append(code.to_bytes(2, byteorder=BYTEORDER))
            block.append(value.to_bytes(2, byteorder=BYTEORDER))

        n_bytes = len(block) * 2
        return b''.join(block), n_bytes


def read_files_by_packages(paths, only_headers=False, series_filter=None):
    """
    Read all data files using given paths.

    Parameters
    ----------
    paths: str or list of str
        Paths to data files. May contain directory paths and file paths.
        Function does not consider file in subdirectories.
    only_headers: bool, default False
        If True read only headers, not quadratures.
    series_filter: ParameterFilter, default None
        Filter for certain options.

    Yields
    -------
    data_block: TimeSeriesPackage
        Block of realizations corresponding to the same time mark.
    """
    file_paths = _collect_valid_file_paths(paths)
    for path in file_paths:
        print('Process file: {}'.format(path))
        with open_data_file(path) as data_reader:
            yield from data_reader.read_blocks(only_headers=only_headers,
                                               series_filter=series_filter)


def read_files_by_series(paths, only_headers=False, series_filter=None):
    """
    Read all data files using given paths.

    Parameters
    ----------
    paths: str or list of str
        Paths to data files. May contain directory paths and file paths.
        Function does not consider file in subdirectories.
    only_headers: bool, default False
        If True read only headers, not quadratures.
    series_filter: ParameterFilter, default None
        Filter for certain options.

    Yields
    -------
    time_series: SignalTimeSeries
        Annotated single realization.
    """
    file_paths = _collect_valid_file_paths(paths)
    for path in file_paths:
        print('Process file: {}'.format(path))
        with open_data_file(path) as data_reader:
            yield from data_reader.read_series(only_headers=only_headers,
                                               series_filter=series_filter)


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
        if new_path.upper().endswith(FILE_EXTENSIONS):
            files_paths.append(os.path.abspath(new_path))

    for path in paths:
        if os.path.isfile(path):
            check_and_add_path(path)
        elif os.path.isdir(path):
            for file_in_dir in sorted(os.listdir(path)):
                check_and_add_path(os.path.join(path, file_in_dir))
    return files_paths


@contextlib.contextmanager
def open_data_file(path, mode='r'):
    """
    Open IISR datafile. If file is compressed, creates a temporal file.

    Parameters
    ----------
    path: str
        Path to file.
    mode: 'r' or 'w', default 'w'
        Mode of operation, read or write.

    Returns
    -------
    file_stream: DataFileReader or DataFileWriter
    """
    if mode not in ['w', 'r']:
        raise ValueError('mode should be "w" or "r", not {}'.format(mode))

    archive_extension = '.gz'
    compressed = path.endswith(archive_extension)
    reading = mode is 'r'
    writing = mode is 'w'

    if compressed and reading:
        with tempfile.NamedTemporaryFile(delete=False) as file:
            with open(path, 'rb') as zipped_file:
                file.write(gzip.decompress(zipped_file.read()))
            working_path = file.name
    elif compressed and writing:
        working_path = path.replace(archive_extension, '')
    else:
        working_path = path

    try:
        binary_mode = mode + 'b'
        with open(working_path, binary_mode) as stream:
            if mode is 'r':
                yield DataFileReader(stream, file_path=path)
            else:
                yield DataFileWriter(stream)

    finally:
        # Remove temporal file
        if compressed and reading:
            os.remove(working_path)
        elif compressed and writing:
            with open(path, 'w') as archive_file:
                with open(working_path, 'w') as data_file:
                    archive_file.write(gzip.compress(data_file))
            os.remove(working_path)


class ParameterFilter:
    """
    Filter to separate series with different options during reading.
    This may decrease computational costs because only necessary series will be read.

    Initialize filter valid options and use check_parameters method.
    The filter could also be improved to reject invalid options, but for now such
    functionality is redundant.
    """
    def __init__(self, valid_parameters):
        """
        Initialize filter. Arguments are represented as dictionary with keys as
        options names. Values could be list, tuple or single entity.

        If raw parameter is given, its name must match RAW_PARAMETERS_CODES.
        If refined parameter is given, its must match Parameters.REFINED_PARAMETERS.

        Parameters
        ----------
        valid_parameters: dict
            Parameters that should pass the filter.
        """
        self._valid_parameters = {}
        for key, value in valid_parameters.items():
            if key in RAW_PARAMETERS_CODES or key in Parameters.REFINED_PARAMETERS:
                if not isinstance(value, (list, tuple)):
                    self._valid_parameters[key] = [value]
                else:
                    self._valid_parameters[key] = value
            else:
                raise ValueError('Incorrect parameter name: {}'.format(key))

    def test_parameters(self, parameters):
        """
        Check if given options pass the filter.

        Parameters
        ----------
        parameters: Parameters
            Refined options of signal series.
        Returns
        -------
        valid: bool
        """
        # Check if given options match filter valid options
        for key, values in self._valid_parameters.items():
            if hasattr(parameters, key):
                test_value = getattr(parameters, key)
            elif key in parameters.rest_raw_parameters:
                test_value = parameters.rest_raw_parameters[key]
            else:
                return False

            for val in values:
                # If some of the options correspond to val
                if test_value == val:
                    break
            else:
                # If there is no match between options and values
                return False

        return True


def _check_raw_parameters(raw_parameters):
    """Check raw options for validness"""
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
        raise RuntimeError('Raw options miss key options')


def raw2refined_parameters(raw_parameters, data_byte_length):
    """
    Process raw options of IISR data files to get convenient options and time.
    Consume options from raw_parameters to reduce memory usage.

    Parameters
    ----------
    raw_parameters: dict
        Dictionary of options from raw files.
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
    pulse_type = CHANNELS_INFO[channel]['type']

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
    total_delay = first_delay - offset_st1 - long_pulse_len + DELAY_FORMULA_CONSTANT

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
    parameters.sampling_frequency = units.Frequency(sampling_frequency, 'kHz')
    parameters.pulse_length = units.Time(pulse_length_us, 'us')
    parameters.pulse_type = pulse_type
    parameters.n_samples = n_samples
    parameters.channel = channel
    parameters.phase_code = raw_parameters.pop('phase_code')
    parameters.frequency = units.Frequency(frequency_MHz, 'MHz')
    parameters.total_delay = total_delay

    parameters.rest_raw_parameters = raw_parameters
    return time, parameters


def refined2raw_parameters(time_mark, refined_parameters, default_offset_st1=80,
                           decimation=2):
    """
    Build raw options dictionary from refined options.

    Parameters
    ----------
    time_mark: datetime
    refined_parameters: Parameters
    default_offset_st1: int
        Default value of receiver offset. Applied if offset is not in refined_parameters.
    decimation: int
        Decimation used during observation.

    Returns
    -------
    raw_parameters: dict
        Dictionary of options from IISR data files.
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
    raw_parameters['sample_freq'] = int(refined_parameters.sampling_frequency
                                        * decimation)

    if 'offset_st1' not in raw_parameters:
        raw_parameters['offset_st1'] = default_offset_st1

    if pulse_type is 'long':
        long_pulse_len = refined_parameters.pulse_length
    else:
        long_pulse_len = 0

    raw_parameters['first_delay'] = refined_parameters.total_delay \
                                    + raw_parameters['offset_st1'] \
                                    + long_pulse_len \
                                    - DELAY_FORMULA_CONSTANT

    raw_parameters['channel'] = refined_parameters.channel
    raw_parameters['phase_code'] = refined_parameters.phase_code

    # Time options
    raw_parameters['date_year'] = time_mark.year
    month = time_mark.month
    day = time_mark.day
    raw_parameters['date_mon_day'] = (month << 8) + day
    hour = time_mark.hour
    minute = time_mark.minute
    raw_parameters['time_h_m'] = (minute << 8) + hour
    raw_parameters['time_sec'] = time_mark.second
    raw_parameters['time_msec'] = time_mark.microsecond // 1000

    # Frequency
    frequency_kHz = int(refined_parameters.frequency * 1000)
    fr_lo = frequency_kHz & 0xFFFF
    fr_hi = frequency_kHz >> 16
    raw_parameters['st1_{}_fr_lo'.format(pulse_type)] = fr_lo
    raw_parameters['st1_{}_fr_hi'.format(pulse_type)] = fr_hi

    raw_parameters['st1_{}_len'.format(pulse_type)] = refined_parameters.pulse_length

    return raw_parameters, data_byte_length
