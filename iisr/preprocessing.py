"""
First-stage processing of iisr data.

"""
from iisr.representation import TimeSeriesPackage, SignalTimeSeries, Parameters
import iisr.io as dataread


class Results:
    def __init__(self, parameters, data):
        self.parameters = parameters
        self.data = data

    def save(self, path):
        pass


class LaunchConfig:
    def __init__(self, paths, mode, channels, frequencies=None):
        """
        Create launch configuration. Check input arguments for validness.

        Parameters
        ----------
        paths: list of str
             Paths to files or directories.
        mode: 'incoherent', 'satellite' or 'passive'
            Mode of operation.
        channels: list of int
            Channels to process. Must be in [0..3] range.
        frequencies: list of float or None, default None
            Frequencies to process, MHz. If None process all frequencies.
        """
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if not isinstance(path, str):
                raise ValueError('Incorrect path: {}'.format(path))

        if mode not in ['incoherent', 'satellite', 'passive']:
            raise ValueError('Incorrect mode: {}'.format(mode))
        self.mode = mode

        for ch in channels:
            if not isinstance(ch, int) and (ch < 0 or ch > 3):
                raise ValueError('Incorrect channel: {}'.format(ch))
        self.channels = channels

        if frequencies is not None:
            for freq in frequencies:
                if not isinstance(freq, (int, float)) and (freq < 150. or freq > 170.):
                    raise ValueError('Incorrect frequency: {}'.format(freq))
        self.frequencies = frequencies

    def __str__(self):
        msg = [
            'Launch configuration',
            '--------------------',
            'Mode: {}'.format(self.mode),
            'Channels: {}'.format(self.channels),
            'Frequencies: {}'.format(self.frequencies),
        ]
        return '\n'.join(msg)


def read_config():
    """
    Read run configuration.

    Returns
    -------
    config: LaunchConfig
    """
    pass


def run_processing(config: LaunchConfig) -> Results:
    print(config)
    parameters = None
    data = None

    # Pass config file paths to dataread to get realizations
    # Filter realizations of each time step by channels, frequencies and other parameters
    # Form arrays with length corresponding to averaging duration defined in config
    #
    # Process arrays according to config
    #

    return Results(parameters, data)


