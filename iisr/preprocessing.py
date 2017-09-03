"""
First-stage processing of iisr data.

"""
from datetime import datetime, timedelta
from iisr.representation import FirstStageResults
from iisr import io


class LaunchConfig:
    def __init__(self, paths, mode, n_accumulation, channels, frequencies=None,
                 pulse_length_us=None, phase_code=None, accumulation_timeout=60):
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
        pulse_length_us: list of int or None, default None
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

        if pulse_length_us is not None:
            for len_ in pulse_length_us:
                if not isinstance(len_, int) or (len_ < 0 or len_ > 3000):
                    raise ValueError('Incorrect pulse length: {}'.format(len_))
        self.pulse_length_us = pulse_length_us

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
            'Pulse lengths: {} us'.format(self.pulse_length_us),
            'Phase code: {}'.format(self.phase_code),
        ]
        return '\n'.join(msg)


def run_processing(config):
    """
    Launch processing given configuration.
    Prints information about the process.

    Parameters
    ----------
    config: LaunchConfig

    Returns
    -------
    results: FirstStageResults
    """
    print(config)

    # Filter realizations of each time step by channels, frequencies and other parameters
    filter_parameters = {}
    if config.frequencies is not None:
        filter_parameters['frequency_MHz'] = config.frequencies
    if config.channels is not None:
        filter_parameters['channel'] = config.channels
    if config.pulse_length_us is not None:
        filter_parameters['pulse_length_us'] = config.pulse_length_us
    if config.phase_code is not None:
        filter_parameters['phase_code'] = config.phase_code

    series_filter = io.ParameterFilter(valid_parameters=filter_parameters)

    # Pass file paths to io to get realizations
    series_packages = io.read_files_by_packages(paths=config.paths,
                                                series_filter=series_filter)

    # Form arrays with length corresponding to accumulation duration
    # Process arrays according to configuration

    parameters = None
    data = None
    return FirstStageResults(parameters, data)


class Handler:
    """Parent class for various types of first-stage processing."""
    def _online(self, value):
        """Online processing algorithm"""

    def _batch(self, array):
        """Batch processing algorithm"""


class PowerHandler(Handler):
    pass


class SpectrumHandler(Handler):
    pass
