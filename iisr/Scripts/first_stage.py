"""
Command line program for first stage processing of IISR data.
Create configuration files, based on default .ini file, to modify options of processing.
"""
import os
import argparse
import configparser
from iisr.preprocessing import LaunchConfig, run_processing
from iisr import units

DEFAULT_CONFIG_FILE = os.path.join('..', 'first_stage_config.ini')

description = """
Manages the launch of first stage processing. Uses configuration file passed to -c 
argument to set main processing options, or uses default configuration.
"""

config = configparser.ConfigParser()
SEPARATOR = ','


def parse_options(option, option_type, *type_args):
    """
    Parse configuration option that should be None, list or single value.

    Parameters
    ----------
    option: str
        String option to parse.
    option_type: type
        Type of option. Must accept str.

    Returns
    -------
    value: None or list of given type
    """
    if not option:
        raise ValueError('Empty option string')

    if option.lower() == 'none':
        return None
    else:
        return [option_type(option, *type_args) for option in option.split(SEPARATOR)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=description)
    # Parse arguments
    parser.add_argument('-c', '--config-file', default=DEFAULT_CONFIG_FILE, type=str,
                        help='configuration file')
    parser.add_argument('--paths', type=str, nargs='*',
                        help='paths to file or directory')
    parser.add_argument('-m', '--mode', type=str,
                        help='mode of operation: incoherent, satellite or passive')
    parser.add_argument('--channels', type=int, nargs='*',
                        help='channels to process')
    parser.add_argument('--frequencies', type=float, nargs='*',
                        help='frequencies to process')
    parser.add_argument('--n-accumulation', type=int,
                        help='number of samples to accumulate')
    args = parser.parse_args(argv)

    # Read given configuration file
    config.read(args.config_file)

    # Exchange config fields that where passed as command line arguments
    for name in ['paths', 'mode', 'channels', 'frequencies', 'n_accumulation']:
        if getattr(args, name) is not None:
            config['Common'][name] = getattr(args, name)

    # Create LaunchConfig instance and pass it to processing
    launch_config = LaunchConfig(
        paths=parse_options(config['Common']['paths'], str),
        n_accumulation=int(config['Common']['n_accumulation']),
        mode=config['Common']['mode'],
        channels=parse_options(config['Common']['channels'], int),
        frequencies=parse_options(config['Common']['frequencies'], units.Frequency, 'MHz'),
        pulse_length=parse_options(config['Common']['pulse_length'], units.Time, 'us'),
        phase_code=parse_options(config['Common']['phase_code'], int),
        accumulation_timeout=int(config['Common']['accumulation_timeout']),
    )
    run_processing(launch_config)


if __name__ == '__main__':
    main()
