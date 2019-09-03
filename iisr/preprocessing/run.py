"""
Pre processing stage processing of IISR data.
"""
import logging
import time
import datetime as dt
from pathlib import Path
from abc import ABCMeta, abstractmethod
from typing import List, Union

from iisr import iisr_io
from iisr.data_manager import DataManager
from iisr.preprocessing.active import ActiveSupervisor
from iisr.preprocessing.passive import PassiveSupervisor
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit


class LaunchConfig(metaclass=ABCMeta):
    paths = NotImplemented
    valid_output_formats = NotImplemented

    def _check_output_formats(self, output_formats: List[str]):
        output_formats = [fmt.lower() for fmt in output_formats]
        for ofmt in output_formats:
            if ofmt not in self.valid_output_formats:
                raise ValueError('Incorrect output format: {}'.format(ofmt))
        return output_formats

    @abstractmethod
    def series_filter(self) -> iisr_io.SeriesSelector:
        pass

    @staticmethod
    def _check_paths(paths: List[Path]):
        if isinstance(paths, Path):
            paths = [paths]

        for path in paths:
            if not isinstance(path, Path):
                raise TypeError(f'Expect pathlib.Path, got {type(path)}')

            if not path.exists():
                raise ValueError(f'Given path not exists: {path}')

        return paths

    @staticmethod
    def _check_positive_int(value: int):
        assert isinstance(value, int)
        assert value > 0
        return value

    @staticmethod
    def _check_channels(channels: List[Channel]):
        for ch in channels:
            assert isinstance(ch, Channel)
        return tuple(sorted(set(channels)))

    @staticmethod
    def _check_frequencies(frequencies: List[Frequency]):
        if frequencies is not None:
            for freq in frequencies:
                assert isinstance(freq, Frequency)
                assert (freq['MHz'] > 100) and (freq['MHz'] < 200.)
        return frequencies

    @staticmethod
    def _check_timedelta(period: Union[int, dt.timedelta]):
        assert isinstance(period, (int, dt.timedelta))
        if isinstance(period, int):
            period = dt.timedelta(minutes=period)
        return period

    @abstractmethod
    def __str__(self):
        pass


class IncoherentConfig(LaunchConfig):
    valid_output_formats = ['std', 'txt']

    def __init__(self, paths: List[Path], output_formats: List[str],
                 n_accumulation: int, channels: List[Channel],
                 frequencies: List[Frequency] = None, pulse_lengths: List[TimeUnit] = None,
                 accumulation_timeout: Union[int, dt.timedelta] = 60, n_fft: int = None,
                 n_spectra: int = None,
                 output_dir_suffix: str = '', clutter_estimate_window: int = None,
                 clutter_drift_compensation: bool = False):
        """
        Create launch configuration. Check if input arguments are valid.

        Parameters
        ----------
        paths: Path or list of Path
            Paths to files or directories.
        output_formats: str or list of str
            Output formats.
        n_accumulation: int
            Number of accumulated samples.
        channels: list of int
            Channels to process. Must be in [0..3] range.
        frequencies: list of float or None, default None
            Frequencies to process, MHz. If None process all frequencies.
        pulse_lengths: list of int or None, default None
            Pulse length (in us) that should be processed. If None process all lengths.
        accumulation_timeout: dt.timedelta or int
            Maximum time in minutes between first and last accumulated samples.
        n_fft: int
        """
        # Check, transform and save input paramters
        self.paths = self._check_paths(paths)
        self.output_formats = self._check_output_formats(output_formats)
        self.n_accumulation = self._check_positive_int(n_accumulation)
        self.channels = self._check_channels(channels)
        self.frequencies = self._check_frequencies(frequencies)
        self.pulse_lengths = self._check_pulse_length(pulse_lengths)
        self.accumulation_timeout = self._check_timedelta(accumulation_timeout)

        if clutter_estimate_window is not None:
            if not isinstance(clutter_estimate_window, int) or clutter_estimate_window < 1:
                raise ValueError('Incorrect clutter estimation window: {}'
                                 ''.format(clutter_estimate_window))

        self.n_fft = self._check_positive_int(n_fft)
        self.n_spectra = self._check_positive_int(n_spectra)

        assert isinstance(output_dir_suffix, str)
        self.output_dir_suffix = output_dir_suffix
        self.clutter_estimate_window = self._check_positive_int(clutter_estimate_window)
        assert isinstance(clutter_drift_compensation, bool)
        self.clutter_drift_compensation = clutter_drift_compensation

    @staticmethod
    def _check_pulse_length(pulse_lengths):
        if pulse_lengths is not None:
            for len_ in pulse_lengths:
                assert isinstance(len_, TimeUnit)
                assert len_['us'] > 0
        return pulse_lengths

    def series_filter(self) -> iisr_io.SeriesSelector:
        filter_parameters = {}
        if self.frequencies is not None:
            filter_parameters['frequencies'] = self.frequencies
        if self.channels is not None:
            filter_parameters['channels'] = self.channels
        if self.pulse_lengths is not None:
            filter_parameters['pulse_lengths'] = self.pulse_lengths
        return iisr_io.SeriesSelector(**filter_parameters)

    def __str__(self):
        msg = [
            'Launch configuration',
            '--------------------',
            'Paths:\n{}'.format('\n'.join((str(path) for path in self.paths))),
            'Output formats: {}'.format(', '.join(self.output_formats)),
            'Output directory suffix: {}'.format(self.output_dir_suffix),
            'Number of accumulated samples: {}'.format(self.n_accumulation),
            'Channels: {}'.format(self.channels),
            'Frequencies: {} MHz'.format(self.frequencies),
            'Pulse lengths: {} us'.format(self.pulse_lengths),
            'Accumulation timeout: {:.2f} s'.format(self.accumulation_timeout.total_seconds()),
            'FFT length: {}'.format(self.n_fft),
            'Number of power spectra: {}'.format(self.n_spectra),
            'Number of series to estimate clutter: {}'.format(self.clutter_estimate_window),
            'Compensate for amplitude drift during clutter subtraction: {}'
            ''.format(self.clutter_drift_compensation),
        ]
        return '\n'.join(msg)


class PassiveConfig(LaunchConfig):
    valid_output_formats = ['pkl']

    def __init__(self, paths: List[Path], output_formats: List[str],
                 output_dir_suffix: str, n_accumulation: int,
                 n_fft: int, channels: List[Channel], frequencies: List[Frequency] = None,
                 accumulation_timeout: Union[int, dt.timedelta] = dt.timedelta(minutes=60)):
        self.paths = self._check_paths(paths)

        self.output_formats = self._check_output_formats(output_formats)
        assert isinstance(output_dir_suffix, str)
        self.output_dir_suffix = output_dir_suffix
        self.n_accumulation = self._check_positive_int(n_accumulation)
        self.n_fft = self._check_positive_int(n_fft)
        self.channels = self._check_channels(channels)
        self.frequencies = self._check_frequencies(frequencies)
        self.accumulation_timeout = self._check_timedelta(accumulation_timeout)

    def series_filter(self) -> iisr_io.SeriesSelector:
        filter_parameters = {}
        if self.frequencies is not None:
            filter_parameters['frequencies'] = self.frequencies
        if self.channels is not None:
            filter_parameters['channels'] = self.channels
        return iisr_io.SeriesSelector(**filter_parameters)

    def __str__(self):
        msg = [
            f'Launch configuration',
            f'--------------------',
            'Paths:\n{}'.format('\n'.join((str(path) for path in self.paths))),
            f'Output directory suffix: {self.output_dir_suffix}',
            'Number of accumulated samples: {}'.format(self.n_accumulation),
            f'Channels: {self.channels}',
            f'Frequencies: {self.frequencies} MHz',
            f'Accumulation timeout: {self.accumulation_timeout.total_seconds():.2f} s',
            f'FFT length: {self.n_fft}',
        ]
        return '\n'.join(msg)


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
    start_time = time.time()
    logging.info(config)

    # Initialize supervisor based on options
    if isinstance(config, IncoherentConfig):
        supervisor = ActiveSupervisor(config.n_accumulation,
                                      n_fft=config.n_fft,
                                      n_spectra=config.n_spectra,
                                      timeout=config.accumulation_timeout,
                                      clutter_drift_compensation=config.clutter_drift_compensation,
                                      clutter_estimate_window=config.clutter_estimate_window)
    elif isinstance(config, PassiveConfig):
        supervisor = PassiveSupervisor(n_accumulation=config.n_accumulation,
                                       n_fft=config.n_fft,
                                       timeout=config.accumulation_timeout)
    else:
        raise RuntimeError('Unknown config type')

    manager = DataManager()
    # Process series
    try:
        with iisr_io.read_files_by('blocks',
                                   paths=config.paths,
                                   series_selector=config.series_filter()) as generator:
            supervisor.process_packages(generator,
                                        data_manager=manager,
                                        output_formats=config.output_formats,
                                        subfolders=[config.output_dir_suffix])
    finally:
        logging.info(f'Elapsed time: {time.time() - start_time:.0f} s')
    logging.info(f'Processing successful')
