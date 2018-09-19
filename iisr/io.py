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
import logging
from collections import namedtuple, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from typing import IO, Dict, Tuple, Iterable, List, Union

import numpy as np

from iisr.representation import Channel, CHANNELS_INFO
from iisr import units
from iisr.units import Frequency, TimeUnit

__all__ = ['DataFileReader', 'DataFileWriter', 'open_data_file', 'read_files_by']
ARCHIVE_EXTENSION = '.gz'
FILE_EXTENSIONS = ('.ISE', '.ISE.GZ', '.IST', '.IST.GZ')
DELAY_FORMULA_CONSTANT = -960 - 50
KEYWORD = b'ORDA'
BYTEORDER = 'little'
STRUCT_BYTEORDER = {'little': '<', 'big': '>'}[BYTEORDER]
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
    'channel',          # channel number
    'data_type',
    'date_year',        # date: year
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
RAW_NAME_TO_CODE = {RAW_PARAMETERS_CODES[i]: i for i in range(len(RAW_PARAMETERS_CODES))}
RAW_CODE_TO_NAME = {code: name for name, code in RAW_NAME_TO_CODE.items()}


ExperimentParameters = namedtuple('ExperimentParameters',
                                  ['sampling_frequency', 'n_samples', 'total_delay'])
FileInfo = namedtuple('FileInfo', ['field1', 'field2', 'field3', 'field4'])


class ReadError(RuntimeError):
    pass


class InvalidFilenameError(Exception):
    pass


###############################################################
# ########### Data representation and converters ############ #
###############################################################


class SeriesParameters:
    """
    Class representing refined parameters of series.
    """
    REFINED_PARAMETERS = {
        'global_parameters',
        'channel',
        'frequency',
        'pulse_length',
        'phase_code',
        'antenna_end',
    }

    @property
    def sampling_frequency(self):
        return self.global_parameters.sampling_frequency

    @property
    def n_samples(self):
        return self.global_parameters.n_samples

    @property
    def total_delay(self):
        return self.global_parameters.total_delay

    @property
    def pulse_type(self):
        return self.channel.pulse_type

    @property
    def band_type(self):
        return self.channel.band_type

    @property
    def horn(self):
        return self.channel.horn

    def __init__(self, file_info: FileInfo, global_parameters: ExperimentParameters,
                 channel: Channel, frequency: Frequency, pulse_length: TimeUnit, phase_code: int,
                 antenna_end: str = None):
        self.file_info = file_info
        self.global_parameters = global_parameters
        self.channel = channel
        self.frequency = frequency
        self.pulse_length = pulse_length
        self.phase_code = phase_code
        self.antenna_end = antenna_end

    def __str__(self):
        msg = [
            'Series Parameters:',
            'File info: {}'.format(self.file_info),
            'Sampling frequency: {}'.format(self.sampling_frequency),
            'Number of samples: {}'.format(self.n_samples),
            'Total delay: {}'.format(self.total_delay),
            'Channel: {}'.format(self.channel),
            'Frequency: {}'.format(self.frequency),
            'Pulse length: {}'.format(self.pulse_length),
            'Phase code: {}'.format(self.phase_code),
            'Antenna end: {}'.format(self.antenna_end),
        ]

        return '\n'.join(msg)

    def __hash__(self):
        return hash(tuple(getattr(self, name) for name in sorted(self.REFINED_PARAMETERS)))

    def __eq__(self, parameters):
        """
        Compare with another options to check if their refined options match.

        Parameters
        ----------
        parameters: SeriesParameters

        Returns
        -------
        match: bool
        """
        for param_name in self.REFINED_PARAMETERS:
            if getattr(self, param_name) != getattr(parameters, param_name):
                return False
        else:
            return True


class SignalTimeSeries:
    """
    Time series of sampled received signal.
    """
    def __init__(self, time_mark, parameters, quadratures):
        """
        Parameters
        ----------
        time_mark: datetime.datetime
        parameters: SeriesParameters
        quadratures: ndarray of complex numbers
        """
        self.time_mark = time_mark
        self.parameters = parameters
        self.quadratures = quadratures

    @property
    def size(self):
        if self.parameters.n_samples is not None:
            return self.parameters.n_samples
        else:
            raise ValueError('options n_samples is not initialized')

    def __str__(self):
        msg = [
            'Time mark: {}'.format(self.time_mark),
            self.parameters.__str__(),
            'Quadratures: {}'.format(self.quadratures)
        ]
        return '\n'.join(msg)


class TimeSeriesPackage:
    """
    Stores signal time series that correspond to identical time, i.e. that originate from
    the same pulse.
    """
    def __init__(self, time_mark, time_series_list):
        """
        Parameters
        ----------
        time_mark: datetime.datetime
        time_series_list: list of SignalTimeSeries
        """
        for series in time_series_list:
            if series.time_mark != time_mark:
                raise ValueError('Given time series must have identical time_mark: '
                                 '{} != {}'.format(series.time_mark, time_mark))

        if not time_series_list:
            raise ValueError('time series list is empty')

        self.time_mark = time_mark
        self.time_series_list = time_series_list

    def __iter__(self):
        return self.time_series_list.__iter__()


def parse_filename(filename: str) -> Tuple[datetime, FileInfo]:
    """Parse *.ISE, *.ISR files names (and their archived versions).

    Args:
        filename: Name of file.

    Returns:
        dtime: Date and time, stored in file name.
        file_info: Fields in file name.
    """
    dtime_fmt = '%Y%m%d_%H%M_'
    dtime_len = 14
    try:
        dtime_field = filename[:dtime_len]
        dtime = datetime.strptime(dtime_field, dtime_fmt)

        fields = filename[dtime_len:].split('.')[0].split('_')
        field1 = int(fields[0])
        field2 = int(fields[1])
        field3 = int(fields[2])
        field4 = int(fields[3])
    except ValueError:
        raise InvalidFilenameError("Expect filename of form '{}_000_0000_002_000.ISE' (got {})"
                                   "".format(dtime_fmt, filename))

    return dtime, FileInfo(field1, field2, field3, field4)


def _get_antenna_end(raw_parameters: Dict[str, int]) -> str:
    """Extract information about antenna end from given parameters.

    Args:
        raw_parameters: Parameters.

    Returns:
        antenna_end: Name of the antenna end.
    """
    with_st1 = False
    with_st2 = False
    for key, value in raw_parameters.items():
        if 'st1' in key and value != 0:
            with_st1 = True
        elif 'st2' in key and value != 0:
            with_st2 = True

    if with_st1 and with_st2:
        raise ValueError('Non-zero st1 and st2 fields')
    elif not with_st1 and not with_st2:
        raise ValueError('Raw options miss key options')
    elif with_st1:
        return 'st1'
    elif with_st2:
        return 'st2'
    else:
        raise AssertionError()


def _raw2refined_parameters(raw_parameters: Dict[str, int],
                            data_byte_length: int,
                            file_info: FileInfo) -> Tuple[datetime, SeriesParameters]:
    """Process raw options of IISR data files to get convenient options and time.
    Consume options from raw_parameters to reduce memory usage.

    Args:
        raw_parameters: Dictionary of codes and values from raw files.
        data_byte_length: Length of corresponding data block in bytes.
        file_info: Information stored in file name.

    Returns:
        time_mark: Time of observation.
        refined_parameters: Decoded parameters of experiment.
    """
    # Convert codes to names
    raw_parameters = {RAW_CODE_TO_NAME[code]: value for code, value in raw_parameters.items()}

    # Decode parameters
    antenna_end = _get_antenna_end(raw_parameters)
    n_samples = data_byte_length // 4  # two quadratures
    decimation = raw_parameters['number_all'] / n_samples
    sampling_frequency = units.Frequency(raw_parameters['sample_freq'] / decimation, 'kHz')

    channel = Channel(raw_parameters['channel'])
    pulse_type = channel.pulse_type

    first_delay = raw_parameters['first_delay']
    offset_st1 = raw_parameters['offset_st1']

    fr_lo = raw_parameters['{}_{}_fr_lo'.format(antenna_end, pulse_type)]
    fr_hi = raw_parameters['{}_{}_fr_hi'.format(antenna_end, pulse_type)]
    frequency = units.Frequency(fr_lo + (fr_hi << 16), 'kHz')

    pulse_length_us = raw_parameters['{}_{}_len'.format(antenna_end, pulse_type)]

    # 'noise' type is unused
    # # For 900 us pulses, there is no room for short pulse and only channel noise
    # # was recorded
    # if pulse_type is 'short' and pulse_length_us == 0:
    #     pulse_type = 'noise'

    if pulse_type is 'long':
        long_pulse_len = pulse_length_us
    else:
        long_pulse_len = 0

    # Magic vague formula to calculate total delay
    total_delay = first_delay - offset_st1 - long_pulse_len + DELAY_FORMULA_CONSTANT

    # Time
    month_day = raw_parameters['date_mon_day']
    hour_min = raw_parameters['time_h_m']

    year = raw_parameters['date_year']
    second = raw_parameters['time_sec']
    millisecond = raw_parameters['time_msec']

    month = month_day >> 8
    day = month_day & 0x00FF
    minute = hour_min >> 8
    hour = hour_min & 0x00FF

    # It appears sometimes *.ISE files have millisecond >= 1000
    if millisecond >= 1000:
        time = datetime(
            year=year, month=month, day=day, hour=hour, minute=minute,
            second=second
        ) + timedelta(microseconds=millisecond * 1000)
    else:
        time = datetime(
            year=year, month=month, day=day, hour=hour, minute=minute,
            second=second, microsecond=millisecond * 1000
        )

    # Form output
    global_params = ExperimentParameters(
        sampling_frequency=sampling_frequency,
        n_samples=n_samples,
        total_delay=units.TimeUnit(total_delay, 'us'),
    )
    parameters = SeriesParameters(
        file_info=file_info,
        global_parameters=global_params,
        channel=channel,
        frequency=frequency,
        pulse_length=units.TimeUnit(pulse_length_us, 'us'),
        phase_code=raw_parameters['phase_code'],
        antenna_end=antenna_end,
    )

    return time, parameters


def _channel2raw(channel: Channel) -> int:
    return channel.value


def _frequency2raw(frequency: Frequency) -> Tuple[int, int]:
    frequency = int(frequency['kHz'])
    fr_lo = frequency & 0xFFFF
    fr_hi = frequency >> 16
    return fr_lo, fr_hi


def _pulse_length2raw(pulse_length: TimeUnit) -> int:
    return int(pulse_length['us'])


RawDateTime = namedtuple('RawDateTime',
                         ['date_year', 'date_mon_day', 'time_h_m', 'time_sec', 'time_msec'])


def _datetime2raw(dtime: datetime) -> RawDateTime:
    date_year = dtime.year
    date_mon_day = (dtime.month << 8) + dtime.day
    time_h_m = (dtime.minute << 8) + dtime.hour
    time_sec = dtime.second
    time_msec = dtime.microsecond // 1000
    return RawDateTime(date_year, date_mon_day, time_h_m, time_sec, time_msec)


def _refined2raw_parameters(time_mark: datetime, refined_parameters: SeriesParameters,
                            offset_st1: int = 80, decimation: int = 2
                            ) -> Tuple[Dict[int, int], int]:
    """Convert refined series parameters to raw encoded parameters. Additional arguments are used
    to define parameters that are not contained in SeriesParameters.

    Args:
        time_mark:
        refined_parameters:
        offset_st1: Receiver offset.
        decimation: Decimation used during observation.

    Returns:
        raw_parameters: Dictionary of codes and values.
        data_byte_length: Length of corresponding data block.
    """
    raw_parameters = {'number_all': refined_parameters.n_samples * decimation}
    # Reclaim unused raw_parameters
    data_byte_length = refined_parameters.n_samples * 4
    raw_parameters['sample_freq'] = int(refined_parameters.sampling_frequency['kHz'] * decimation)
    raw_parameters['offset_st1'] = offset_st1

    pulse_type = refined_parameters.pulse_type
    if pulse_type is 'long':
        long_pulse_len = int(refined_parameters.pulse_length['us'])
    else:
        long_pulse_len = 0

    raw_parameters['first_delay'] = int(refined_parameters.total_delay['us']) \
                                    + raw_parameters['offset_st1'] \
                                    + long_pulse_len \
                                    - DELAY_FORMULA_CONSTANT

    raw_parameters['channel'] = _channel2raw(refined_parameters.channel)
    raw_parameters['phase_code'] = refined_parameters.phase_code

    # Time options
    raw_dtime = _datetime2raw(time_mark)
    raw_parameters['date_year'] = raw_dtime.date_year
    raw_parameters['date_mon_day'] = raw_dtime.date_mon_day
    raw_parameters['time_h_m'] = raw_dtime.time_h_m
    raw_parameters['time_sec'] = raw_dtime.time_sec
    raw_parameters['time_msec'] = raw_dtime.time_msec

    # Antenna end
    if refined_parameters.antenna_end is not None:
        antenna_end = refined_parameters.antenna_end
    else:
        antenna_end = 'st1'  # default

    # Frequency
    fr_lo, fr_hi = _frequency2raw(refined_parameters.frequency)
    raw_parameters['{}_{}_fr_lo'.format(antenna_end, pulse_type)] = fr_lo
    raw_parameters['{}_{}_fr_hi'.format(antenna_end, pulse_type)] = fr_hi

    raw_parameters['{}_{}_len'.format(antenna_end, pulse_type)] \
        = int(refined_parameters.pulse_length['us'])

    # Encode parameters
    raw_parameters = {RAW_NAME_TO_CODE[name]: value for name, value in raw_parameters.items()}

    return raw_parameters, data_byte_length


class SeriesSelector:
    """Selector for separation of series with different parameters during reading.

    The selector is initialized with valid refined parameters, which are converted
    to raw parameters. Then selector is used as a filter at raw parameter level.

    Separate time check can be used on given time marks. It potentially could be also converted to
    raw parameter level (as done for other parameters), but many peculiarities make this process
    too complicated.
    """
    def __init__(self, start_time: datetime = None, stop_time: datetime = None,
                 channels: Union[Channel, Iterable[Channel]] = None,
                 pulse_types: Union[str, Iterable[str]] = None,
                 frequencies: Union[Frequency, Iterable[Frequency]] = None,
                 pulse_lengths: Union[TimeUnit, Iterable[TimeUnit]] = None):
        """Initialize selector with valid parameters. If any of arguments is None, then all input
        parameters, corresponding to the argument, are valid.

        Args:
            start_time: Start time. All inputs before it will be rejected.
            stop_time: Stop time. All inputs after it will be rejected.
            channels: Valid input channels.
            pulse_types: Valid pulse types.
            frequencies: Valid frequencies.
            pulse_lengths: Valid pulse lengths.
        """
        self.start_time = start_time
        self.stop_time = stop_time

        valid_parameters = defaultdict(set)

        # Channels
        channels_set = set(ch for ch in CHANNELS_INFO)
        if channels is not None:
            if isinstance(channels, Channel):
                channels = [channels]

            channels_set.intersection_update(_channel2raw(ch) for ch in channels)

        if pulse_types is not None:
            if isinstance(pulse_types, str):
                pulse_types = [pulse_types]

            for pulse_type in pulse_types:
                type_channels = (ch for ch, info in CHANNELS_INFO.items()
                                 if info['type'] == pulse_type)
                channels_set.intersection_update(type_channels)

        valid_parameters[RAW_NAME_TO_CODE['channel']] = channels_set

        # Frequencies
        if frequencies is not None:
            if isinstance(frequencies, Frequency):
                frequencies = [frequencies]

            fr_lo_codes = [RAW_NAME_TO_CODE[name] for name in RAW_PARAMETERS_CODES
                           if name.endswith('fr_lo')]
            fr_hi_codes = [RAW_NAME_TO_CODE[name] for name in RAW_PARAMETERS_CODES
                           if name.endswith('fr_hi')]
            for freq in frequencies:
                fr_lo, fr_hi = _frequency2raw(freq)

                for code in fr_lo_codes:
                    valid_parameters[code].add(fr_lo)

                for code in fr_hi_codes:
                    valid_parameters[code].add(fr_hi)

        # Pulse lengths
        if pulse_lengths is not None:
            if isinstance(pulse_lengths, TimeUnit):
                pulse_lengths = [pulse_lengths]

            pulse_len_codes = [RAW_NAME_TO_CODE[name] for name in RAW_PARAMETERS_CODES
                               if name.endswith('len')]

            for pulse_length in pulse_lengths:
                for code in pulse_len_codes:
                    valid_parameters[code].add(_pulse_length2raw(pulse_length))

        self._valid_parameters = valid_parameters

    def validate_parameters(self, parameters: Dict[int, int]) -> bool:
        """Check if given options pass the selector.

        Args:
            parameters: Refined parameters of signal series.

        Returns:
            valid: True if valid.
        """
        # Check if given options match selector valid parameters
        for key, test_values_set in self._valid_parameters.items():
            if key in parameters and parameters[key] not in test_values_set:
                return False

        return True

    def validate_time_mark(self, time_mark: datetime) -> bool:
        """Check if given time mark is within selector time limits.

        Args:
            time_mark: Time mark.

        Returns:
            valid: True if valid.
        """
        if self.start_time is not None and time_mark < self.start_time:
            return False

        if self.stop_time is not None and time_mark > self.stop_time:
            return False

        return True


###############################################################
# ###########      Reader and Writer Classes     ############ #
###############################################################


class DataFileIO:
    pass


class DataFileReader(DataFileIO):
    """Read binary data stream."""
    def __init__(self, stream: IO, file_info: FileInfo, series_selector: SeriesSelector = None,
                 only_headers: bool = False):
        """Create reader instance.

        Args:
            stream: Input data stream.
            file_info: Information stored in file name.
            only_headers: If True return only annotation of time series,
                leaving quadratures field of SignalTimeSeries instance as None. Defaults to False.
            series_selector: Filter for options. Defaults to None.
        """
        self.stream = stream
        self.file_info = file_info
        self._series = self._series_generator()
        self.selector = series_selector
        self.only_headers = only_headers

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
        global_parameters = {}
        if header_code == HEADER_CODES['global']:
            self._read_raw_parameters_block(global_parameters, block_length)
        # Empty file does not raise error
        elif header_code is None:
            return
        else:
            raise ReadError(
                'Global header is not at the first position of file {} '
                '(get code {} instead of {})'
                ''.format(self.file_info, header_code, HEADER_CODES['global'])
            )

        # Reading remaining data
        while True:
            # Read super block, annotation of data
            header_code, block_length = self._read_header()
            raw_parameters = global_parameters.copy()
            if header_code == HEADER_CODES['super']:
                self._read_raw_parameters_block(raw_parameters, block_length)
            # End of file
            elif header_code is None:
                break
            else:
                raise ReadError(
                    'Incorrect code [{}] in file {} (superheader code {} expected)'
                    ''.format(header_code, self.file_info, HEADER_CODES['super']))

            # Read data block. It must come after each super header
            header_code, data_length = self._read_header()

            if header_code != HEADER_CODES['data']:
                raise ReadError(
                    'Incorrect code [{}] in file {} (data code {} expected)'
                    ''.format(header_code, self.file_info, HEADER_CODES['data']))
            # End of file
            elif header_code is None:
                break

            # Check if parameters pass selector
            if self.selector is not None and not self.selector.validate_parameters(raw_parameters):
                self.stream.seek(data_length, 1)  # from current position
                continue

            # Form refined options
            time_mark, series_parameters = \
                _raw2refined_parameters(raw_parameters, data_length, self.file_info)

            # Check if time mark within the limits
            if self.selector is not None and not self.selector.validate_time_mark(time_mark):
                self.stream.seek(data_length, 1)  # from current position
                continue

            if not self.only_headers:
                quadratures = self.read_quadratures(data_length)
            else:
                self.stream.seek(data_length, 1)  # from current position
                quadratures = None

            # Create annotated signal time series (realization)
            time_series = SignalTimeSeries(time_mark, series_parameters, quadratures)
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

    def read_quadratures(self, data_byte_length):
        """
        Read quadratures from file_stream given address and length.

        Parameters
        ----------
        data_byte_length: int
            Length of quadratures to be read.

        Returns
        -------
        quadratures: np.ndarray of complex numbers.
        """
        quadrature_byte_length = data_byte_length // 2
        quadrature_size = quadrature_byte_length // 2

        # Read consequent I and Q samples. Repeat quadrature_size times.
        dtype = np.dtype([
            ('quad_I', '<i2', quadrature_size),
            ('quad_Q', '<i2', quadrature_size)
        ])

        quadratures = np.fromfile(self.stream, dtype=dtype, count=1)

        result = np.array(quadratures['quad_Q'][0]).astype(np.complex64)

        # Invert Q quadrature to compensate for IISR demodulation
        np.multiply(result, -1j, out=result)
        result += np.array(quadratures['quad_I'][0])
        return result

    def _read_raw_parameters_block(self, parameters: dict, block_length: int):
        """
        Read raw options of time series from stream to parameters dictionary (modify dict in-place).

        Parameters
        ----------
        block_length: int
            Length of block in bytes.
        """
        # Each parameter represented by 2 bytes for code and 2 bytes for value
        n_bytes = 4
        n_parameters = block_length // n_bytes

        piece = self.stream.read(n_parameters * n_bytes)
        # Unpack 2 x n_params for codes and values
        unpacked_piece = struct.unpack('<{}H'.format(n_parameters * 2), piece)

        for code, value in zip(unpacked_piece[::2], unpacked_piece[1::2]):
            # Searching for parameter code in pre-defined tuple
            if code in parameters:
                raise RuntimeError('Parameters dictionary already has code {}'.format(code))
            parameters[code] = value


class DataFileWriter(DataFileIO):
    global_parameters_names = ['number_all', 'offset_st1', 'number_after',
                               'mode', 'step', 'sample_freq', 'version']

    def __init__(self, stream):
        self.stream = stream
        self.current_global_header = None

    def _isnew_global_header(self, params: Dict[int, int]):
        if self.current_global_header is None:
            return True

        for code in self.current_global_header:
            if code not in params or params[code] != self.current_global_header[code]:
                return True
        return False

    def write(self, series: SignalTimeSeries):
        """Write series to IISR file stream."""
        raw_parameters, data_byte_length = _refined2raw_parameters(series.time_mark,
                                                                   series.parameters)

        # Global header
        global_header = {}
        for name in self.global_parameters_names:
            if name in raw_parameters:
                code = RAW_NAME_TO_CODE[name]
                global_header[code] = raw_parameters.pop(code)

        if self._isnew_global_header(global_header):
            self._write_header('global', global_header)
            self.current_global_header = global_header

        self._write_header('super', raw_parameters)
        self._write_data_block(series.quadratures, data_byte_length)

    def write_series_package(self, block):
        """

        Parameters
        ----------
        block: TimeSeriesPackage
        """
        for series in block:
            self.write(series)

    def _write_header(self, header_type, parameters):
        block, block_length = self._get_raw_parameters_block(parameters)
        header = self._get_header(HEADER_CODES[header_type], block_length)
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
        # Invert Q quadrature to mimic inversion in original .ise data
        quads_q = (-int(number) for number in quadratures.imag)

        quads_i_bytes = (number.to_bytes(2, BYTEORDER, signed=True) for number in quads_i)
        quads_q_bytes = (number.to_bytes(2, BYTEORDER, signed=True) for number in quads_q)

        data_block = b''.join(it.chain(quads_i_bytes, quads_q_bytes))
        header = self._get_header(HEADER_CODES['data'], byte_length)
        self.stream.write(b''.join([header, data_block]))

    @staticmethod
    def _get_header(code, block_length):
        return b''.join([KEYWORD, code.to_bytes(1, byteorder=BYTEORDER),
                         block_length.to_bytes(4, byteorder=BYTEORDER)])

    @staticmethod
    def _get_raw_parameters_block(raw_parameters):
        block = []
        for code, value in raw_parameters.items():
            block.append(code.to_bytes(2, byteorder=BYTEORDER))
            block.append(value.to_bytes(2, byteorder=BYTEORDER))

        n_bytes = len(block) * 2
        return b''.join(block), n_bytes


###############################################################
# ###########      IO convenience functions      ############ #
###############################################################


@contextlib.contextmanager
def open_data_file(path: Path, mode: str = 'r', compress_on_write: bool = False,
                   only_headers: bool = False, series_selector: SeriesSelector = None
                   ) -> DataFileIO:
    """Open IISR datafile. Creates a temporal file for compress operations.

    Args:
        path: Path to file.
        mode: Mode of operation, read or write.
        compress_on_write: If file should be compressed on write. Do nothing at 'r' mode.
        only_headers: If True, read only headers in the data files.
        series_selector: Selector that passes only series with valid parameters.

    Returns:
        data_file_io: Data file reader or writer.

    """
    if mode not in ['w', 'r']:
        raise ValueError('mode should be "w" or "r", not {}'.format(mode))

    reading = mode == 'r'
    writing = mode == 'w'

    is_archive_suffix = path.suffix.lower() == ARCHIVE_EXTENSION

    file_dtime, file_info = parse_filename(path.name)

    if reading and is_archive_suffix:
        compressed = True
    elif writing and compress_on_write:
        compressed = True
        if is_archive_suffix:
            path = Path(str(path)[:-len(ARCHIVE_EXTENSION)])
    else:
        compressed = False

    if compressed and reading:
        with tempfile.TemporaryFile() as file:
            with open(str(path), 'rb') as zipped_file:
                file.write(gzip.decompress(zipped_file.read()))
            file.seek(0)
            yield DataFileReader(file, file_info, series_selector, only_headers)

    elif reading:
        with open(str(path), 'rb') as file:  # type: IO
            yield DataFileReader(file, file_info, series_selector, only_headers)

    elif compressed and writing:
        with open(str(path), 'wb') as file:
            yield DataFileWriter(file)

        with open(str(path) + ARCHIVE_EXTENSION, 'w') as archive_file:
            with open(str(path), 'r') as data_file:
                archive_file.write(gzip.compress(data_file))
        os.remove(str(path))

    elif writing:
        with open(str(path), 'wb') as file:
            yield DataFileWriter(file)

    else:
        raise AssertionError('Unexpected behaviour')


def _collect_valid_file_paths(paths: Union[Path, Iterable[Path]]) -> List[Path]:
    """Search through given paths to gather file list of IISR data files.

    Args:
        paths: Paths to data files. May contain directory paths and file paths.
            Function does not consider file in subdirectories.

    Returns:
        gathered_paths: Sorted list on IISR data file paths.
    """
    if isinstance(paths, Path):
        paths = [paths]

    # Get list of all files paths
    files_paths = []

    def check_and_add_path(new_path: Path):
        if new_path.exists() and new_path.suffix.upper().endswith(FILE_EXTENSIONS):
            files_paths.append(new_path.resolve())

    for path in paths:
        if path.is_dir():
            for file_in_dir in sorted(os.listdir(str(path))):
                check_and_add_path(path / file_in_dir)
        else:
            check_and_add_path(path)

    return sorted(files_paths)


@contextlib.contextmanager
def read_files_by(read_type: str, paths: Iterable[Path], only_headers: bool = False,
                  series_selector: SeriesSelector = None):
    """
    Read all data files using given paths.

    Parameters
    ----------
    read_type
    paths: str or list of str
        Paths to data files. May contain directory paths and file paths.
        Function does not consider file in subdirectories.
    only_headers: bool, default False
        If True read only headers, not quadratures.
    series_selector: ParameterFilter, default None
        Filter for certain options.

    Yields
    -------
    data_block: TimeSeriesPackage
        Block of realizations corresponding to the same time mark.
    """
    valid_types = ['series', 'blocks']
    if read_type not in valid_types:
        raise ValueError('Incorrect by = {} (expect {})'.format(read_type, valid_types))

    file_paths = _collect_valid_file_paths(paths)

    def _generator():
        for path_num, path in enumerate(file_paths, 1):
            logging.info('[{}/{}] Process file: {}'.format(path_num, len(file_paths), path))

            with open_data_file(path, only_headers=only_headers,
                                series_selector=series_selector) as data_reader:
                if read_type == 'blocks':
                    yield from data_reader.read_blocks()
                elif read_type == 'series':
                    yield from data_reader.read_series()
                else:
                    raise AssertionError()

    generator = _generator()
    try:
        yield generator
    finally:
        generator.close()
