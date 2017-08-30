"""
Collect classes for IISR data representation.
"""
__all__ = ['Parameters', 'SignalTimeSeries', 'SignalBlock', 'CHANNELS_NUMBER_INFO']


class Parameters:
    def __init__(self):
        self.sampling_frequency = None
        self.n_samples = None
        self.total_delay = None

        self.channel = None
        self.frequency_MHz = None
        self.pulse_length_us = None
        self.pulse_type = None
        self.phase_code = None

        self.rest_raw_parameters = None

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
        if self.rest_raw_parameters is not None:
            for k, v in self.rest_raw_parameters.items():
                msg.append('\t{}:  {}'.format(k, v))
        return '\n'.join(msg)


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
        return self.parameters.n_samples

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


class SignalBlock:
    """
    Stores signal time series that correspond to identical time, i.e. that originate from
    the same pulse.
    """
    def __init__(self, time_mark, time_series_list):
        """
        Parameters
        ----------
        time_mark: datetime
        time_series_list: list of SignalTimeSeries
        """
        self.time_mark = time_mark
        self.time_series_list = time_series_list

# Channels 0, 2 for narrow band pulse, channels 1, 3 for wide band pulse
CHANNELS_NUMBER_INFO = {
    0: {'type': 'long', 'horn': 'upper'},
    1: {'type': 'short', 'horn': 'upper'},
    2: {'type': 'long', 'horn': 'lower'},
    3: {'type': 'short', 'horn': 'lower'}
}