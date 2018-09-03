"""
Collect classes for IISR data representation.
"""
__all__ = ['SeriesParameters', 'SignalTimeSeries', 'TimeSeriesPackage', 'CHANNELS_INFO']

from iisr.units import UnitsJSONDecoder, UnitsJSONEncoder
from collections import namedtuple
import json
import os


ExperimentGlobalParameters = namedtuple('ExperimentGlobalParameters',
                                        ['sampling_frequency', 'n_samples', 'total_delay'])


class SeriesParameters:
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

    def __init__(self, global_parameters, channel, frequency, pulse_length, phase_code, pulse_type,
                 antenna_end=None):
        self.global_parameters = global_parameters
        self.channel = channel
        self.frequency = frequency
        self.pulse_length = pulse_length
        self.phase_code = phase_code
        self.pulse_type = pulse_type
        self.antenna_end = antenna_end

        self._hash = None

    def __str__(self):
        msg = [
            '==== Parameters ====',
            'Sampling frequency: {}'.format(self.sampling_frequency),
            'Number of samples: {}'.format(self.n_samples),
            'Total delay: {} us'.format(self.total_delay),
            'Channel: {}'.format(self.channel),
            'Frequency: {}'.format(self.frequency),
            'Pulse length: {}'.format(self.pulse_length),
            'Pulse type: {}'.format(self.pulse_type),
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


CHANNELS_INFO = {
    0: {'type': 'long', 'horn': 'upper', 'band_type': 'narrow'},
    1: {'type': 'short', 'horn': 'upper', 'band_type': 'wide'},
    2: {'type': 'long', 'horn': 'lower', 'band_type': 'narrow'},
    3: {'type': 'short', 'horn': 'lower', 'band_type': 'wide'}
}


class Channel:
    __slots__ = ['value', 'pulse_type', 'horn', 'band_type']

    def __init__(self, value):
        _valid_channels = [0, 1, 2, 3]
        if value not in _valid_channels:
            raise ValueError('Channel can be one of {}'.format(_valid_channels))
        self.value = value
        self.pulse_type = CHANNELS_INFO[value]['type']
        self.horn = CHANNELS_INFO[value]['horn']
        self.band_type = CHANNELS_INFO[value]['band_type']

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other: 'Channel'):
        if not isinstance(other, Channel):
            raise TypeError('Types {} and {} are not comparable'.format(Channel, int))
        return self.value == other.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)

    def __le__(self, other: 'Channel'):
        return self.value.__le__(other.value)

    def __lt__(self, other: 'Channel'):
        return self.value.__lt__(other.value)

    def __ge__(self, other: 'Channel'):
        return self.value.__ge__(other.value)

    def __gt__(self, other: 'Channel'):
        return self.value.__gt__(other.value)


CHANNELS = [Channel(0), Channel(1), Channel(2), Channel(3)]

JSON_REPR_TYPE_STR = '_repr_type'


class ReprJSONEncoder(UnitsJSONEncoder):
    def default(self, obj):
        if isinstance(obj, Channel):
            return {JSON_REPR_TYPE_STR: Channel.__name__, 'value': obj.value}
        return super().default(obj)


class ReprJSONDecoder(UnitsJSONDecoder):
    def object_hook(self, obj):
        if JSON_REPR_TYPE_STR not in obj:
            return super().object_hook(obj)

        return Channel(obj['value'])


class Results:
    """Processing results"""
    def save(self, path_to_dir: str):
        """Save results to directory."""


class FirstStageResults(Results):
    options_file = 'options.json'

    def __init__(self, results, options: dict = None):
        self.results = results
        self.options = options

    def save(self, path_to_dir: str):
        """Save results and options of processing to directory.

        Args:
            path_to_dir: Path to directory.
        """
        with open(os.path.join(path_to_dir, self.options_file), 'w') as file:
            json.dump(self.options, file, indent=4)

        for result in self.results:
            result.save_txt(path_to_dir)

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

        return FirstStageResults(results, options)


class SecondStageResults(Results):
    """Results of second stage processing."""
