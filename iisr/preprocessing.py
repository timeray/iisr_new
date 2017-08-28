"""
First-stage processing of iisr data.

"""
from iisr.representation import SignalBlock, SignalTimeSeries, Parameters
import iisr.io as dataread


class Results:
    def __init__(self, parameters, data):
        self.parameters = parameters
        self.data = data

    def save(self, path):
        pass


class RunConfig:
    def __init__(self, datapaths, mode, channels, frequncies):
        pass


def read_config():
    """
    Read run configuration.

    Returns
    -------
    config: RunConfig
    """
    pass


def run_processing(config: RunConfig) -> Results:
    parameters = None
    data = None

    # Pass config file paths to dataread to get realizations
    # Filter realizations of each time step by channels, frequencies and other parameters
    # Form arrays with length corresponding to averaging duration defined in config
    #
    # Process arrays according to config
    #

    return Results(parameters, data)


