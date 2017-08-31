"""
Command line program for first stage processing of IISR data.
Create configuration files, based on default .ini file, to modify options of processing.
"""
import os
import argparse
import configparser
from iisr.preprocessing import LaunchConfig, run_processing

DEFAULT_CONFIG_FILE = os.path.join('..', 'default_config.ini')

description = """
Manages the launch of first stage processing. Uses configuration file passed to -c 
argument to set main processing options, or uses default configuration.
"""

config = configparser.ConfigParser()

if __name__ == '__main__':
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
    args = parser.parse_args()

    # Read given configuration file
    config.read(args.config_file)

    # Exchange fields that where passed as arguments in configuration
    for name in ['paths', 'mode', 'channels', 'frequencies']:
        if getattr(args, name) is not None:
            config['Common'][name] = getattr(args, name)

    # Create LaunchConfig instance and pass it to processing
    launch_config = LaunchConfig(
        paths=config['Common']['paths'],
        mode=config['Common']['mode'],
        channels=[int(ch) for ch in config['Common']['channels'].split(',')],
        frequencies=[float(freq) for freq in config['Common']['frequencies'].split(',')]
    )
    run_processing(launch_config)
