"""
First-stage processing of iisr data.

"""
import numpy as np

from datetime import datetime, timedelta, date
from iisr.representation import FirstStageResults, Parameters
from iisr.data_manager import DataManager
from iisr import io
from collections import namedtuple, defaultdict
from typing import List, Iterator


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


def aggregate_packages(packages: Iterator[io.TimeSeriesPackage],
                       n_accumulation: int,
                       timeout: timedelta = timedelta(minutes=5),
                       drop_timeout_series: bool = True):
    """
    Pack series by parameters until their count reaches n_accumulation. Check if difference between
    two consecutive time marks exceeds timeout. If this situation occurs, already aggregated
    series may be yielded or dropped depending on the flag.

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
    for package in packages:
        package.time_series_list


    unique_arrays = {}
    series_counter = defaultdict(lambda: {'time_marks': [], 'quadratures': []})
    for package in packages:
        # Wipe out any series beyond timeout
        for series_params, counter in series_counter.items():
            previous_mark = counter['time_marks'][-1]
            series_timedelta = package.time_mark - previous_mark
            if series_timedelta > timeout:
                del series_counter[series.parameters]  # ??? cannot delete during iteration
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
    print('Start processing')
    print(config)

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

    # Initialize handler based on options
    handler = Handler()

    # Pick series
    series_filter = io.ParameterFilter(valid_parameters=filter_parameters)
    series_package_generator = io.read_files_by_packages(paths=config.paths,
                                                         series_filter=series_filter)

    # Group series to form arrays of length equal to accumulation duration, check for timeouts
    packed_data = aggregate_packages(series_package_generator,
                                     config.n_accumulation,
                                     timeout=config.accumulation_timeout)

    # Pass arrays to handler
    results = handler.handle_batch(packed_data)

    # After series expire gather results from handlers and save them
    manager = DataManager()

    manager.save_first_stage_results(results)
    print('Processing successful')


class Handler:
    """Parent class for various types of first-stage processing."""
    def online(self, value):
        """Online processing algorithm"""

    def handle_batch(self, series_batches) -> FirstStageResults:
        """Batch processing algorithm"""

    def calc_power(self, quadratures: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate signal power.

        Args:
            quadratures: array of complex numbers.
            axis: Averaging axis. Defaults to 0.

        Returns:
            power: array of floats.
        """
        return (quadratures.real ** 2 + quadratures.imag ** 2).mean(axis=axis)


class ActiveResults:
    def __init__(self, time_marks: List[datetime], parameters: Parameters,
                 options: dict, power: np.ndarray, spectrum: np.ndarray):
        self.time_marks = time_marks
        self.parameters = parameters
        self.options = options
        self.power = power
        self.spectrum = spectrum

        # Calculate all involved dates
        self.dates = sorted(set((date(dt.year, dt.month, dt.day) for dt in self.time_marks)))

        # Gather set of experiment parameters and processing options
        # that uniquely identify the results
        self.results_specification = {
            'parameters': parameters,
            'options': options,
        }

    def save(self, path_to_dir: str, save_date: date = None):
        """Save results to specific directory. If date was passed, save only results corresponding
        to this date.

        Args:
            path_to_dir: Path to save directory.
            save_date: Date to save.
        """
        # Save data for given date
        if save_date is not None:
            if save_date in self.dates:
                with open(path_to_dir, 'w') as file:
                    do the thing
            else:
                raise ValueError('Not results for given date {}'.format(save_date))
        # Save data for all dates to single file
        else:
            pass


class ActiveHandler(Handler):
    def __init__(self, n_fft, h_step):
        self.nfft = n_fft
        self.h_step = h_step
        self.time_marks = {}
        self.power = {}
        self.spectrum = {}

    def handle_batch_(self, params: Parameters, time_marks: List[datetime], quadratures: np.ndarray):
        """Process batch of quadratures corresponding to unique parameters..

        Args:
            params: Parameters for given quadratures.
            time_marks: N-length list of datetimes.
            quadratures: (N x M) array of complex numbers. N - number of samples to average.
                M - number of samples from single pulse.
        """
        power = self.calc_power(quadratures)
        self.power.append(power)

    def finish(self):
        """Output results and free memory.

        Returns: FirstStageResults

        """
        return self.power


class CrossCorrelationHandler(Handler):
    def handle_batch(self, array):
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
