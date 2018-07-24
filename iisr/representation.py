"""
Collect classes for IISR data representation.
"""
__all__ = ['Parameters', 'SignalTimeSeries', 'TimeSeriesPackage', 'CHANNELS_INFO']

from iisr.units import Frequency
import json
import os


class SeriesParameters:
    """Base class representing options of quadratures."""
    def __init__(self, sampling_frequency: Frequency,
                 n_samples: int,
                 channel: int,
                 frequency: Frequency):
        self.sampling_frequency = sampling_frequency
        self.n_samples = n_samples
        self.channel = channel
        self.frequency = frequency

    def match(self, other_params: 'SeriesParameters'):
        return NotImplemented


class PassiveSeriesParameters(SeriesParameters):
    """Parameters of quadratures for passive mode of operation."""


class ActiveSeriesParameters(SeriesParameters):
    """Parameters of quadratures for active mode of operation."""


class Parameters:
    """
    Class representing refined options.
    Use REFINED_PARAMETERS to access main parameter names.
    """
    REFINED_PARAMETERS = {
        'sampling_frequency',
        'n_samples',
        'total_delay',
        'channel',
        'frequency',
        'pulse_length',
        'pulse_type',
        'phase_code'
    }

    def __init__(self):
        self.sampling_frequency = None
        self.n_samples = None
        self.total_delay = None

        self.channel = None
        self.frequency = None
        self.pulse_length = None
        self.pulse_type = None
        self.phase_code = None

        assert set(self.__dict__.keys()) == self.REFINED_PARAMETERS

        self.rest_raw_parameters = {}

    def __str__(self):
        msg = [
            '==== Parameters ====',
            'Sampling frequency: {} MHz'.format(self.sampling_frequency),
            'Number of samples: {}'.format(self.n_samples),
            'Total delay: {} us'.format(self.total_delay),
            'Channel: {}'.format(self.channel),
            'Frequency: {} MHz'.format(self.frequency),
            'Pulse length: {} us'.format(self.pulse_length),
            'Pulse type: {}'.format(self.pulse_type),
            'Phase code: {}'.format(self.phase_code),
            'Rest raw options:'
        ]

        for k, v in self.rest_raw_parameters.items():
            msg.append('\t{}:  {}'.format(k, v))
        return '\n'.join(msg)

    def match_refined(self, parameters):
        """
        Compare with another options to check if their refined options match.

        Parameters
        ----------
        parameters: Parameters

        Returns
        -------
        match: bool
        """
        for param_name in self.REFINED_PARAMETERS:
            if getattr(self, param_name) != getattr(parameters, param_name):
                return False
        else:
            return True

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SignalTimeSeries:
    """
    Time series of sampled received signal.
    """
    def __init__(self, time_mark, parameters, quadratures):
        """
        Parameters
        ----------
        time_mark: datetime.datetime
        parameters: Parameters
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
            '======================',
            '== Time series info ==',
            '======================',
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


# Channels 0, 2 for narrow band pulse, channels 1, 3 for wide band pulse
CHANNELS_INFO = {
    0: {'type': 'long', 'horn': 'upper'},
    1: {'type': 'short', 'horn': 'upper'},
    2: {'type': 'long', 'horn': 'lower'},
    3: {'type': 'short', 'horn': 'lower'}
}


class Results:
    """Processing results"""
    def save(self, path_to_dir: str):
        """Save results to directory."""


class FirstStageResults(Results):
    options_file = 'options.json'

    def __init__(self, options: dict, results: list):
        self.options = options
        self.results = results

    def save(self, path_to_dir: str):
        """Save results and options of processing to directory.

        Args:
            path_to_dir: Path to directory.
        """
        with open(os.path.join(path_to_dir, self.options_file), 'w') as file:
            json.dump(self.options, file, indent=4)

        for result in self.results:
            result.save(path_to_dir)

    @classmethod
    def load(cls, path_to_dir: str):
        """Load results from directory.

        Args:
            path_to_dir: Directory with first stage results.
        """
        with open(os.path.join(path_to_dir, cls.options_file)) as file:
            options = json.load(file)

        # Load all results
        results = []
        for filename in os.listdir(path_to_dir):
            # Parse filenames and load corresponding results
            if '.dat' in filename:
                pass

        return FirstStageResults(options, results)


class SecondStageResults(Results):
    """Results of second stage processing."""
