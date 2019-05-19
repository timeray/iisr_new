"""
Pre processing stage processing of IISR data.
"""
import logging
import time
from datetime import timedelta
from pathlib import Path
from abc import ABCMeta, abstractmethod
from typing import List, Union, Dict

from iisr import iisr_io
from iisr.data_manager import DataManager
from iisr.preprocessing.active import ActiveSupervisor, ActiveResult
from iisr.preprocessing.passive import PassiveSupervisor
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit
from iisr.utils import merge, infinite_defaultdict
from iisr import StdFile, AnnotatedData


def _merge_stdfiles(file1: StdFile, file2: StdFile) -> StdFile:
    def key_fn(data: AnnotatedData):
        return data.header.start_time

    new_power = merge(file1.power, file2.power, key=key_fn)
    new_spectra = merge(file1.spectra, file2.spectra, key=key_fn)
    return StdFile(new_power, new_spectra)


class LaunchConfig(metaclass=ABCMeta):
    paths = NotImplemented

    @abstractmethod
    def series_filter(self) -> iisr_io.SeriesSelector:
        pass

    @staticmethod
    def _check_paths(paths: List[Path]):
        if isinstance(paths, str):
            paths = Path(paths)

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
        return channels

    @staticmethod
    def _check_frequencies(frequencies: List[Frequency]):
        if frequencies is not None:
            for freq in frequencies:
                assert isinstance(freq, Frequency)
                assert (freq['MHz'] > 100) and (freq['MHz'] < 200.)
        return frequencies

    @staticmethod
    def _check_timedelta(period: Union[int, timedelta]):
        assert isinstance(period, (int, timedelta))
        if isinstance(period, int):
            period = timedelta(minutes=period)
        return period

    @abstractmethod
    def __str__(self):
        pass


class IncoherentConfig(LaunchConfig):
    def __init__(self, paths: List[Path], output_formats: List[str],
                 n_accumulation: int, channels: List[Channel],
                 frequencies: List[Frequency] = None, pulse_lengths: List[TimeUnit] = None,
                 accumulation_timeout: Union[int, timedelta] = 60, n_fft: int = None,
                 n_spectra: int = None,
                 output_dir_prefix: str = '', clutter_estimate_window: int = None,
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
        accumulation_timeout: timedelta or int
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

        assert isinstance(output_dir_prefix, str)
        self.output_dir_suffix = output_dir_prefix
        self.clutter_estimate_window = self._check_positive_int(clutter_estimate_window)
        assert isinstance(clutter_drift_compensation, bool)
        self.clutter_drift_compensation = clutter_drift_compensation

    @staticmethod
    def _check_output_formats(output_formats: List[str]):
        output_formats = [fmt.lower() for fmt in output_formats]
        for ofmt in output_formats:
            if ofmt not in ['std', 'txt']:
                raise ValueError('Incorrect output format: {}'.format(ofmt))
        return output_formats

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
    pass


def _merge_and_save_stdfiles(results, manager):
    assert all(results[0].dates == res.dates for res in results)
    dates = results[0].dates
    for date in dates:
        grouped_files = infinite_defaultdict()
        for result in results:  # type: ActiveResult
            stdfiles = result.to_std(date)
            for ch, stdfile in stdfiles.items():
                horn = ch.horn
                pulse_type = ch.pulse_type
                stdfile = stdfiles[ch]
                freq = stdfile.power[0].header.frequency_hz / 1e6
                pulse_len = stdfile.power[0].header.pulse_length_us

                if pulse_type == 'short':
                    if pulse_len == 0:
                        # Noise channel, ignore
                        continue

                    # Shift grouping frequency to long channel equivalent
                    freq -= 0.3

                grouped_files[horn, freq][pulse_type][pulse_len] = stdfile

        for (horn, freq), files in grouped_files.items():
            files: Dict[str, Dict[int, StdFile]]
            if len(files) != 2:
                raise ValueError('Expect short and long pulses to be present in files')

            if len(files['short']) != 1:
                raise ValueError('Expect single short pulse')

            short_stdfile = list(files['short'].values())[0]

            if len(files['long']) > 2:
                raise ValueError('Expect at most 2 long pulses (700 and 900)')

            for pulse_len, long_stdfile in files['long'].items():
                stdfile = _merge_stdfiles(short_stdfile, long_stdfile)

                filename = '{}_{}_f{:.2f}_len{}.std' \
                           ''.format(date.strftime('%Y%m%d'), horn, freq, int(pulse_len))
                manager.save_stdfile(stdfile, filename)


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
    else:
        supervisor = PassiveSupervisor(config.n_accumulation, config.n_fft,
                                       timeout=config.accumulation_timeout)

    # Process series
    with iisr_io.read_files_by('blocks',
                               paths=config.paths,
                               series_selector=config.series_filter()) as generator:
        results = supervisor.process_packages(generator)

    # Gather results from handlers and save them
    manager = DataManager()
    for out_fmt in config.output_formats:
        if out_fmt == 'txt':
            for result in results:
                manager.save_preprocessing_result(result, save_dir_suffix=config.output_dir_suffix)
        elif out_fmt == 'std':
            _merge_and_save_stdfiles(results, manager)
        else:
            logging.warning('Unexpected format from config: {}'.format(out_fmt))

    logging.info('Processing successful. Elapsed time: {:.0f} s'.format(time.time() - start_time))
