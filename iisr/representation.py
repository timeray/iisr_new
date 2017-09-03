"""
Collect classes for IISR data representation.
"""
__all__ = ['Parameters', 'SignalTimeSeries', 'TimeSeriesPackage', 'CHANNELS_NUMBER_INFO']


class Parameters:
    """
    Class representing refined parameters.
    Use REFINED_PARAMETERS to access main parameter names.
    """
    REFINED_PARAMETERS = {
        'sampling_frequency',
        'n_samples',
        'total_delay',
        'channel',
        'frequency_MHz',
        'pulse_length_us',
        'pulse_type',
        'phase_code'
    }

    def __init__(self):
        self.sampling_frequency = None
        self.n_samples = None
        self.total_delay = None

        self.channel = None
        self.frequency_MHz = None
        self.pulse_length_us = None
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
            'Frequency: {} MHz'.format(self.frequency_MHz),
            'Pulse length: {} us'.format(self.pulse_length_us),
            'Pulse type: {}'.format(self.pulse_type),
            'Phase code: {}'.format(self.phase_code),
            'Rest raw parameters:'
        ]

        for k, v in self.rest_raw_parameters.items():
            msg.append('\t{}:  {}'.format(k, v))
        return '\n'.join(msg)

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
            raise ValueError('parameters n_samples is not initialized')

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
CHANNELS_NUMBER_INFO = {
    0: {'type': 'long', 'horn': 'upper'},
    1: {'type': 'short', 'horn': 'upper'},
    2: {'type': 'long', 'horn': 'lower'},
    3: {'type': 'short', 'horn': 'lower'}
}


class Results:
    """Processing results"""
    def save(self, path_to_dir):
        """Save results to directory."""


class FirstStageResults(Results):
    def __init__(self, parameters, data):
        self.parameters = parameters
        self.data = data

    def save(self, path):
        pass


class SecondStageResults(Results):
    """Results of second stage processing."""
