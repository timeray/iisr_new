"""
First-stage processing of iisr data.

"""
import numpy as np

from datetime import datetime, timedelta
from iisr.representation import FirstStageResults
from iisr import io
from collections import namedtuple, defaultdict


class LaunchConfig:
    def __init__(self, paths, mode, n_accumulation, channels, frequencies=None,
                 pulse_length=None, phase_code=None, accumulation_timeout=60):
        """
        Create launch configuration. Check input arguments for validness.

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


def form_arrays_from_packages(packages, n_accumulation, timeout=timedelta(minutes=5)):
    """
    Sort series in packages by their parameters until their count reach n_accumulation.

    Parameters
    ----------
    packages: iterable of TimeSeriesPackage
    n_accumulation: int
        Number of accumulated unique series.
    timeout: timedelta

    Yields
    -------
    dict
    """
    unique_arrays = {}
    series_counter = defaultdict(lambda: {'time_marks': [], 'quadratures': []})
    for package in packages:
        # Wipe out any series beyond timeout
        for series_params, counter in series_counter.items():
            previous_mark = counter['time_marks'][-1]
            series_timedelta = package.time_mark - previous_mark
            if series_timedelta > timeout:
                del series_counter[series.parameters] ??? cannot delete during iteration
                continue
            elif series_timedelta < 0:
                raise RuntimeError('Package time mark earlier than series')

        # Accumulate
        for series in package:
            series_counter[series.parameters]['time_marks'].append(package.time_mark)
            series_counter[series.parameters]['quadratures'].append(series.quadratures)

        # Check if accumulation finished
        for unique_params, series in series_counter.items():
            if len(series) == n_accumulation:
                unique_arrays[unique_params] = np.stack(series)
                series_counter[unique_params] = []

        # If arrays ready - yield
        if unique_arrays:
            yield unique_arrays
            unique_arrays = {}


def run_processing(config):
    """
    Launch processing given configuration.

    Parameters
    ----------
    config: LaunchConfig

    Returns
    -------
    results: FirstStageResults
    """
    print('Start processing')
    print(config)

    # Filter realizations for each time step by channels, frequencies and other parameters
    filter_parameters = {}
    if config.frequencies is not None:
        filter_parameters['frequency'] = config.frequencies
    if config.channels is not None:
        filter_parameters['channel'] = config.channels
    if config.pulse_length is not None:
        filter_parameters['pulse_length'] = config.pulse_length
    if config.phase_code is not None:
        filter_parameters['phase_code'] = config.phase_code

    # Pick series
    series_filter = io.ParameterFilter(valid_parameters=filter_parameters)
    series_package_generator = io.read_files_by_packages(paths=config.paths,
                                                         series_filter=series_filter)

    # Group series by parameters to form arrays of length equal to accumulation duration
    print('Assume that series, corresponding to the same transmission times, always '
          'come together, i.e. no such united series are oversampled compared to others.')

    # Gather series from each package
    arrays = form_arrays_from_packages(series_package_generator, config.n_accumulation)

    # Pass arrays to handlers

    # After series expire save results from handlers
    parameters = None
    data = None
    results = FirstStageResults(parameters, data)


class Handler:
    """Parent class for various types of first-stage processing."""
    def online(self, value):
        """Online processing algorithm"""

    def batch(self, array):
        """Batch processing algorithm"""

    def finish(self):
        """Finish processing and return results"""


class PowerHandler(Handler):
    def __init__(self):
        self.power = []

    def batch(self, array):
        """

        Parameters
        ----------
        array: np.ndarray
            NxM array, where N is a number of samples, M is a number of pulses.
        Returns
        -------
        average_power: int
        """
        self.power.append((array.real ** 2 + array.imag ** 2).mean(axis=1))

    def finish(self):
        return self.power


class CrossCorrelationHandler(Handler):
    def batch(self, array):
        """

        Parameters
        ----------
        array:

        Returns
        -------

        """
        # compute correlation between two arrays


class SpectrumHandler(Handler):
    pass
