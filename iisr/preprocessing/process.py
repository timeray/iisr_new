"""
First-stage processing of IISR data.
"""
from collections import defaultdict
from datetime import datetime, timedelta

import logging
import numpy as np
from typing import Iterator

import iisr.io
from iisr import io
from iisr.data_manager import DataManager
from iisr.preprocessing.active import LongPulseActiveHandler, ShortPulseActiveHandler


class LaunchConfig:
    def __init__(self, paths, mode, n_accumulation, channels, frequencies=None,
                 pulse_length=None, phase_code=None, accumulation_timeout=60):
        """
        Create launch configuration. Check if input arguments are valid.

        Parameters
        ----------
        paths: list of str
             Paths to files or directories.
        mode: 'incoherent', 'satellite' or 'passive'
            Mode of operation.
        n_accumulation: int
            Number of accumulated samples.
        channels: list of int
            Channels to process. Must be in [0..3] range.
        frequencies: list of float or None, default None
            Frequencies to process, MHz. If None process all frequencies.
        pulse_length: list of int or None, default None
            Pulse length (in us) that should be processed. If None process all lengths.
        phase_code: list of int or None, default None
            Phase code to process. If None process all phase codes.
        accumulation_timeout: timedelta or int
            Maximum time in minutes between first and last accumulated samples.
        """
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if not isinstance(path, str):
                raise ValueError('Incorrect path: {}'.format(path))
        self.paths = paths

        if mode not in ['incoherent', 'satellite', 'passive']:
            raise ValueError('Incorrect mode: {}'.format(mode))
        self.mode = mode

        if not isinstance(n_accumulation, int):
            raise ValueError('Incorrect number of accumulated samples: {}'
                             ''.format(n_accumulation))
        self.n_accumulation = n_accumulation

        for ch in channels:
            if not isinstance(ch, int) or (ch not in [0, 1, 2, 3]):
                raise ValueError('Incorrect channel: {}'.format(ch))
        self.channels = channels

        if frequencies is not None:
            for freq in frequencies:
                if not isinstance(freq, (int, float)) or (freq < 150. or freq > 170.):
                    raise ValueError('Incorrect frequency: {}'.format(freq))
        self.frequencies = frequencies

        if pulse_length is not None:
            for len_ in pulse_length:
                if not isinstance(len_, int) or (len_ < 0 or len_ > 3000):
                    raise ValueError('Incorrect pulse length: {}'.format(len_))
        self.pulse_length = pulse_length

        if phase_code is not None:
            for code in phase_code:
                if not isinstance(code, int) or (code < 0):
                    raise ValueError('Incorrect code: {}'.format(code))
        self.phase_code = phase_code

        if isinstance(accumulation_timeout, timedelta):
            self.accumulation_timeout = accumulation_timeout
        elif isinstance(accumulation_timeout, int):
            self.accumulation_timeout = timedelta(minutes=accumulation_timeout)
        else:
            raise ValueError('Incorrect accumulation timeout: {}'
                             ''.format(accumulation_timeout))

    def __str__(self):
        msg = [
            'Launch configuration',
            '--------------------',
            'Mode: {}'.format(self.mode),
            'Number of accumulated samples: {}'.format(self.n_accumulation),
            'Channels: {}'.format(self.channels),
            'Frequencies: {} MHz'.format(self.frequencies),
            'Pulse lengths: {} us'.format(self.pulse_length),
            'Phase code: {}'.format(self.phase_code),
        ]
        return '\n'.join(msg)


def aggregate_packages(packages: Iterator[iisr.io.TimeSeriesPackage],
                       n_accumulation: int,
                       timeout: timedelta = timedelta(minutes=5),
                       drop_timeout_series: bool = True):
    def to_arrays(records):
        marks = []
        quads = []
        for time_mark, quad in records:
            marks.append(time_mark)
            quads.append(quad)

        marks = np.array(marks, dtype=datetime)
        quads = np.array(quads, dtype=np.complex)  # [n_acc, quad_length]
        return marks, quads

    # List of unique quadratures: keys - parameters, values - (time_marks, quadratures)
    unique_list = defaultdict(list)
    prev_time_mark = datetime.min  # Same for all tracked parameters
    for package in packages:
        for time_series in package.time_series_list:
            params = time_series.parameters
            new_record = (time_series.time_mark, time_series.quadratures)

            # Check for timeout
            if unique_list[params]:
                time_diff = time_series.time_mark - prev_time_mark
                if time_diff > timeout:
                    if drop_timeout_series:
                        # Reset buffer (now all parameters are invalid)
                        unique_list = defaultdict(list)
                    else:
                        # Yield what was accumulated
                        time_marks, quadratures = to_arrays(unique_list[params])
                        del unique_list[params]
                        yield params, time_marks, quadratures
                elif time_diff < 0:
                    raise RuntimeError(
                        'New time mark is earlier than previous (new {}, prev {})'
                        ''.format(time_series.time_mark, prev_time_mark)
                    )

            prev_time_mark = time_series.time_mark

            # Append new record. If full, yield and reset buffer.
            unique_list[params].append(new_record)
            if len(unique_list[params]) >= n_accumulation:
                time_marks, quadratures = to_arrays(unique_list[params])
                del unique_list[params]
                yield params, time_marks, quadratures


def run_processing(config: LaunchConfig):
    """
    Launch processing given configuration.

    Parameters
    ----------
    config: LaunchConfig

    Returns
    -------
    results: FirstStageResults
    """
    logging.info('Start processing')
    logging.info(config)

    # Filter realizations for each time step by channels, frequencies and other options
    filter_parameters = {}
    if config.frequencies is not None:
        filter_parameters['frequency'] = config.frequencies
    if config.channels is not None:
        filter_parameters['channel'] = config.channels
    if config.pulse_length is not None:
        filter_parameters['pulse_length'] = config.pulse_length
    if config.phase_code is not None:
        filter_parameters['phase_code'] = config.phase_code

    # Initialize handlers based on options
    if config.mode == 'active':
        handlers = [ShortPulseActiveHandler(), LongPulseActiveHandler()]
    elif config.mode == 'passive':
        raise NotImplementedError()
    else:
        raise ValueError('Unknown mode: {}'.format(config.mode))

    # Pick series
    series_filter = io.SeriesSelector(valid_parameters=filter_parameters)
    series_package_generator = io.read_files_by_packages(paths=config.paths,
                                                         series_selector=series_filter)

    # Group series to form arrays of length equal to accumulation duration, check for timeouts
    accumulated_quadratures = aggregate_packages(series_package_generator,
                                                 config.n_accumulation,
                                                 timeout=config.accumulation_timeout)

    # Pass arrays to handler
    for params, time_marks, quadratures in accumulated_quadratures:
        for handler in handlers:
            if handler.validate(params):
                handler.process(params, time_marks, quadratures)

    # Gather results from handlers and save them
    manager = DataManager()
    for handler in handlers:
        result = handler.finish()
        manager.save_preprocessing_result(result)
    logging.info('Processing successful')


