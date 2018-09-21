"""
Command line program for first stage processing of IISR data.
Create configuration files, based on default .ini file, to modify options of processing.
"""
import sys
import argparse
import configparser
import logging
from pathlib import Path

from iisr.preprocessing import LaunchConfig, run_processing
from iisr import IISR_PATH
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit
from typing import Callable, List


logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')
DEFAULT_CONFIG_FILE = IISR_PATH / 'pre_processing_config.ini'

description = """
Manages the launch of first stage processing. Uses configuration file passed to -c 
argument to set main processing options, or uses default configuration.
"""

config = configparser.ConfigParser()
SEPARATOR = ','


def option_parser_decorator(parser: Callable) -> Callable:
    def _parser_wrapper(option):
        if not option:
            raise ValueError('Empty option string')

        if option.lower() == 'none':
            return None
        else:
            return parser(option)

    return _parser_wrapper


@option_parser_decorator
def _parse_path(paths: str) -> List[Path]:
    return [Path(path) for path in paths.split(SEPARATOR)]


@option_parser_decorator
def _parse_channels(channels: str) -> List[Channel]:
    return [Channel(int(ch)) for ch in channels.split(SEPARATOR)]


@option_parser_decorator
def _parse_frequency(frequencies: str) -> List[Frequency]:
    return [Frequency(float(freq), 'MHz') for freq in frequencies.split(SEPARATOR)]


@option_parser_decorator
def _parse_time_units(time_units_values_us: str) -> List[TimeUnit]:
    return [TimeUnit(float(val), 'us') for val in time_units_values_us]


def main(argv=None):
    parser = argparse.ArgumentParser(description=description)
    # Parse arguments
    parser.add_argument('-c', '--config-file', default=str(DEFAULT_CONFIG_FILE),
                        help='configuration file')
    parser.add_argument('--paths', nargs='*', help='paths to file or directory')
    parser.add_argument('-m', '--mode', help='mode of operation: incoherent, satellite or passive')
    parser.add_argument('--channels_set', nargs='*', help='channels_set to process')
    parser.add_argument('--frequencies', nargs='*', help='frequencies to process')
    parser.add_argument('--n-accumulation', help='number of samples to accumulate')
    args = parser.parse_args(argv)

    # Read given configuration file
    config.read(args.config_file)

    # Exchange config fields that where passed as command line arguments
    for name in ['paths', 'mode', 'channels_set', 'frequencies', 'n_accumulation']:
        if getattr(args, name) is not None:
            config['Common'][name] = getattr(args, name)

    # Create LaunchConfig instance and pass it to processing
    launch_config = LaunchConfig(
        paths=_parse_path(config['Common']['paths']),
        n_accumulation=int(config['Common']['n_accumulation']),
        mode=config['Common']['mode'],
        channels=_parse_channels(config['Common']['channels_set']),
        frequencies=_parse_frequency(config['Common']['frequencies']),
        pulse_length=_parse_time_units(config['Common']['pulse_length']),
        accumulation_timeout=int(config['Common']['accumulation_timeout']),
    )
    run_processing(launch_config)


if __name__ == '__main__':
    main()
