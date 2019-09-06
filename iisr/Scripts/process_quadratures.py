"""
Command line program for first stage processing of IISR data.
Create configuration files, based on default .ini file, to modify options of processing.
"""
import sys
import argparse
import configparser
import logging
from iisr.preprocessing.run import IncoherentConfig, PassiveConfig, run_processing
from iisr import IISR_PATH
import iisr.config_utils as cu


logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')
DEFAULT_CONFIG_FILE = IISR_PATH / 'iisr' / 'default_preprocessing.ini'

description = """
Manages the launch of pre-processing. Uses configuration file passed to -c 
argument to set main processing options, otherwise runs default configuration.
"""

config = configparser.ConfigParser()


def main(argv=None):
    parser = argparse.ArgumentParser(description=description)
    # Parse arguments
    parser.add_argument('mode', type=str, choices=['incoherent', 'passive'],
                        help='mode of operation: incoherent or passive')
    parser.add_argument('-c', '--config-file', default=str(DEFAULT_CONFIG_FILE),
                        help='configuration file')
    parser.add_argument('--paths', nargs='*', help='paths to file or directory')
    parser.add_argument('--channels', nargs='*', help='channels to process')
    parser.add_argument('--frequencies', nargs='*', help='frequencies to process')
    parser.add_argument('--n-accumulation', help='number of samples to accumulate')
    args = parser.parse_args(argv)

    # Read given configuration file
    if not config.read(args.config_file):
        raise FileNotFoundError('Wrong config file: ' + args.config_file)

    mode = args.mode

    # Exchange config fields that where passed as command line arguments
    for name in ['paths', 'channels', 'frequencies', 'n_accumulation']:
        if getattr(args, name) is not None:
            config[mode][name] = getattr(args, name)

    # Instantiate LaunchConfig subclass and pass it to processing
    cfg_mode = config[mode]
    paths = cu.parse_path(cfg_mode['paths'])
    output_dir_suffix = cfg_mode['output_folder_suffix']
    n_accumulation = int(cfg_mode['n_accumulation'])
    channels = cu.parse_channels(cfg_mode['channels'])
    frequencies = cu.parse_frequency(cfg_mode['frequencies'])
    accumulation_timeout = int(cfg_mode['accumulation_timeout'])
    n_fft = int(cfg_mode['n_fft'])

    if mode == 'incoherent':
        launch_config = IncoherentConfig(
            paths=paths,
            output_formats=cu.parse_list(cfg_mode['output_formats']),
            output_dir_suffix=output_dir_suffix,
            n_accumulation=n_accumulation,
            channels=channels,
            frequencies=frequencies,
            pulse_lengths=cu.parse_time_units(cfg_mode['pulse_lengths']),
            accumulation_timeout=accumulation_timeout,
            n_fft=n_fft,
            n_spectra=int(cfg_mode['n_spectra']),
            clutter_estimate_window=cu.parse_optional_int(cfg_mode['clutter_estimate_window']),
            clutter_drift_compensation=cu.parse_boolean(
                cfg_mode['clutter_amplitude_drift_compensation']
            ),
        )
    elif mode == 'passive':
        launch_config = PassiveConfig(
            paths=paths,
            output_formats=cu.parse_list(cfg_mode['output_formats']),
            output_dir_suffix=output_dir_suffix,
            n_accumulation=n_accumulation,
            n_fft=n_fft,
            channels=channels,
            frequencies=frequencies,
            accumulation_timeout=accumulation_timeout,
        )
    else:
        raise RuntimeError('Should not reach here')
    run_processing(launch_config)


if __name__ == '__main__':
    main()
