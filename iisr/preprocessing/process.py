"""
First-stage processing of IISR data.
"""
from collections import defaultdict
from datetime import datetime, timedelta

import logging
from pathlib import Path

import numpy as np
from typing import Iterator, List, Union, Generator, Tuple, Any

from iisr import io
from iisr.data_manager import DataManager
from iisr.preprocessing.active import LongPulseActiveHandler, ShortPulseActiveHandler
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit


class LaunchConfig:
    def __init__(self, paths: List[Path], mode: str, n_accumulation: int, channels: List[Channel],
                 frequencies: List[Frequency] = None, pulse_length: List[TimeUnit] = None,
                 accumulation_timeout: Union[int, timedelta] = 60):
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
        accumulation_timeout: timedelta or int
            Maximum time in minutes between first and last accumulated samples.
        """
        if isinstance(paths, str):
            paths = Path(paths)

        if isinstance(paths, Path):
            paths = [paths]

        for path in paths:
            if not isinstance(path, Path):
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
            if not isinstance(ch, Channel):
                raise ValueError('Incorrect channel: {}'.format(ch))
        self.channels = channels

        if frequencies is not None:
            for freq in frequencies:
                if not isinstance(freq, Frequency) or (freq['MHz'] < 150. or freq['MHz'] > 170.):
                    raise ValueError('Incorrect frequency: {}'.format(freq))
        self.frequencies = frequencies

        if pulse_length is not None:
            for len_ in pulse_length:
                if not isinstance(len_, TimeUnit) or (len_['us'] < 0 or len_['us'] > 3000):
                    raise ValueError('Incorrect pulse length: {}'.format(len_))
        self.pulse_length = pulse_length

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
            'Accumulation timeout: {:.2f} s'.format(self.accumulation_timeout.total_seconds())
        ]
        return '\n'.join(msg)


def aggregate_packages(packages: Iterator[io.TimeSeriesPackage],
                       n_accumulation: int,
                       timeout: timedelta = timedelta(minutes=5),
                       drop_timeout_series: bool = True
                       ) -> Generator[Tuple[io.SeriesParameters, np.ndarray, np.ndarray], Any, Any]:
    """Aggregate series with equal parameters from packages.
     Quadratures are accumulated to form 2-d arrays.

    Args:
        packages: Series packages iterator.
        n_accumulation: Number of accumulated series.
        timeout: Maximum distance between 2 consecutive series.
        drop_timeout_series: Defines behaviour when timeout occur. If True, drop already accumulated
        series. If False, yield them.

    Yields:
        params: Unique parameters that corresponds to accumulated series.
        time_marks: All consecutive time marks.
        quadratures: 2-D Array of accumulated quadratures. Shape [n_accumulation x n_samples]
    """

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

                elif time_diff < timedelta(0):
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
        filter_parameters['frequencies'] = config.frequencies
    if config.channels is not None:
        filter_parameters['channels'] = config.channels
    if config.pulse_length is not None:
        filter_parameters['pulse_lengths'] = config.pulse_length

    # Initialize handlers based on options
    if config.mode == 'active':
        handlers = [ShortPulseActiveHandler(), LongPulseActiveHandler()]
    elif config.mode == 'passive':
        raise NotImplementedError()
    else:
        raise ValueError('Unknown mode: {}'.format(config.mode))

    # Pick series
    series_filter = io.SeriesSelector(**filter_parameters)
    with io.read_files_by_packages(paths=config.paths, series_selector=series_filter) as generator:
        # Group series to form arrays of length equal to accumulation duration, check for timeouts
        accumulated_quadratures = aggregate_packages(generator,
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


