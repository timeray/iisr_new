"""
Pre processing stage processing of IISR data.
"""
import logging
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from typing import List, Union, Dict, Tuple

from iisr import iisr_io
from iisr.data_manager import DataManager
from iisr.preprocessing.active import ActiveSupervisor, ActiveResult
from iisr.preprocessing.passive import PassiveSupervisor
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit
from iisr.utils import merge, infinite_defaultdict
from pyasp.stdparse import StdFile, AnnotatedData


def _merge_stdfiles(file1: StdFile, file2: StdFile) -> StdFile:
    def key_fn(data: AnnotatedData):
        return data.header.start_time

    new_power = merge(file1.power, file2.power, key=key_fn)
    new_spectra = merge(file1.spectra, file2.spectra, key=key_fn)
    return StdFile(new_power, new_spectra)


class LaunchConfig:
    def __init__(self, paths: List[Path], output_formats: List[str],
                 mode: str, n_accumulation: int, channels: List[Channel],
                 frequencies: List[Frequency] = None, pulse_length: List[TimeUnit] = None,
                 accumulation_timeout: Union[int, timedelta] = 60, n_fft: int = None,
                 n_spectra: int = None,
                 output_dir_prefix: str = '', clutter_estimate_window: int = None,
                 clutter_drift_compensation: bool = False):
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
        n_fft: int
        """
        if isinstance(paths, str):
            paths = Path(paths)

        if isinstance(paths, Path):
            paths = [paths]

        for path in paths:
            if not isinstance(path, Path):
                raise ValueError('Incorrect path: {}'.format(path))
        self.paths = paths

        output_formats = [fmt.lower() for fmt in output_formats]
        for ofmt in output_formats:
            if ofmt not in ['std', 'txt']:
                raise ValueError('Incorrect output format: {}'.format(ofmt))
        self.output_formats = output_formats

        if mode not in ['incoherent', 'satellite', 'passive']:
            raise ValueError('Incorrect mode: {}'.format(mode))
        self.mode = mode

        if not isinstance(n_accumulation, int):
            raise ValueError('Incorrect number of accumulated samples: {}'
                             ''.format(n_accumulation))
        self.n_accumulation = n_accumulation

        for ch in channels:
            if not isinstance(ch, Channel):
                raise TypeError('Incorrect channel type: {}'.format(type(ch)))
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

        if clutter_estimate_window is not None:
            if not isinstance(clutter_estimate_window, int) or clutter_estimate_window < 1:
                raise ValueError('Incorrect clutter estimation window: {}'
                                 ''.format(clutter_estimate_window))

        self.n_fft = n_fft

        if n_spectra is not None:
            assert isinstance(n_spectra, int)
            assert n_spectra > 0

        self.n_spectra = n_spectra
        self.output_dir_suffix = output_dir_prefix
        self.clutter_estimate_window = clutter_estimate_window
        self.clutter_drift_compensation = clutter_drift_compensation

    def __str__(self):
        msg = [
            'Launch configuration',
            '--------------------',
            'Paths:\n{}'.format('\n'.join((str(path) for path in self.paths))),
            'Output formats: {}'.format(', '.join(self.output_formats)),
            'Output directory suffix: {}'.format(self.output_dir_suffix),
            'Mode: {}'.format(self.mode),
            'Number of accumulated samples: {}'.format(self.n_accumulation),
            'Channels: {}'.format(self.channels),
            'Frequencies: {} MHz'.format(self.frequencies),
            'Pulse lengths: {} us'.format(self.pulse_length),
            'Accumulation timeout: {:.2f} s'.format(self.accumulation_timeout.total_seconds()),
            'FFT length: {}'.format(self.n_fft),
            'Number of power spectra: {}'.format(self.n_spectra),
            'Number of series to estimate clutter: {}'.format(self.clutter_estimate_window),
            'Compensate for amplitude drift during clutter subtraction: {}'
            ''.format(self.clutter_drift_compensation),
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

    # Filter realizations for each time step by channels_set, frequencies and other options
    filter_parameters = {}
    if config.frequencies is not None:
        filter_parameters['frequencies'] = config.frequencies
    if config.channels is not None:
        filter_parameters['channels'] = config.channels
    if config.pulse_length is not None:
        filter_parameters['pulse_lengths'] = config.pulse_length

    series_filter = iisr_io.SeriesSelector(**filter_parameters)

    # Initialize supervisor based on options
    if config.mode == 'incoherent':
        supervisor = ActiveSupervisor(config.n_accumulation,
                                      n_fft=config.n_fft,
                                      n_spectra=config.n_spectra,
                                      timeout=config.accumulation_timeout,
                                      clutter_drift_compensation=config.clutter_drift_compensation,
                                      clutter_estimate_window=config.clutter_estimate_window)
    elif config.mode == 'passive':
        supervisor = PassiveSupervisor(config.n_accumulation, config.n_fft,
                                       timeout=config.accumulation_timeout)
    else:
        raise ValueError('Unknown mode: {}'.format(config.mode))

    # Process series
    with iisr_io.read_files_by('blocks', paths=config.paths, series_selector=series_filter) as generator:
        results = supervisor.process_packages(generator)

    # Gather results from handlers and save them
    manager = DataManager()
    for out_fmt in config.output_formats:
        if out_fmt == 'txt':
            for result in results:
                manager.save_preprocessing_result(result, save_dir_suffix=config.output_dir_suffix)
        elif out_fmt == 'std':
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

        else:
            logging.warn('Unexpected format from config: {}'.format(out_fmt))

    logging.info('Processing successful. Elapsed time: {:.0f} s'.format(time.time() - start_time))
