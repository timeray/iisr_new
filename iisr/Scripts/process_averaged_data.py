"""
Command line program for second stage processing.
Receive type of processing, dates or ID of first stage results, and other option.
"""
import configparser
import sys
import logging
import argparse
from pathlib import Path

from iisr import IISR_PATH
import iisr.config_utils as cu
from iisr.data_manager import DataManager
from iisr.postprocessing.run import compute_source_track, compute_sky_power, perform_calibration, \
    compute_sun_pattern, compute_sun_flux

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')
DEFAULT_CONFIG_FILE = IISR_PATH / 'iisr' / 'default_postprocessing.ini'

description = """
Control for second stage processing. Contain multiple types of processing and depend 
on results from first stage.
"""

config = configparser.ConfigParser()


def main(argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('mode', type=str, choices=['track', 'sky_power', 'calibration',
                                                   'sun_pattern', 'sun_flux'])
    parser.add_argument('-c', '--config-file', default=str(DEFAULT_CONFIG_FILE),
                        help='configuration file')
    args = parser.parse_args(argv)

    if not config.read(args.config_file):
        raise FileNotFoundError('Wrong config')

    cfg_common = config['Common']
    dates = cu.parse_dates_ranges(cfg_common['dates'])
    riometer_data_dirpath = cfg_common['riometer_data_dirpath']
    riometer_data_dirpath = Path(riometer_data_dirpath) if riometer_data_dirpath else None
    subfolder_pre = cfg_common['preprocessing_subfolder']
    subfolder_post = cfg_common['postprocessing_subfolder']

    with DataManager(riometer_data_folder=riometer_data_dirpath) as manager:
        mode = args.mode
        if mode == 'track':
            for date in dates:
                compute_source_track(manager, date, subfolder_pre, subfolder_post)
        elif mode == 'sky_power':
            for date in dates:
                compute_sky_power(manager, date, subfolder_pre, subfolder_post)
        elif mode == 'calibration':
            for date in dates:
                perform_calibration(manager, date, subfolder_pre, subfolder_post, subfolder_post)
        elif mode == 'sun_pattern':
            for date in dates:
                compute_sun_pattern(manager, date, subfolder_pre, subfolder_post)
        elif mode == 'sun_flux':
            for date in dates:
                compute_sun_flux(manager, date, subfolder_pre, subfolder_post, subfolder_post)

    logging.info('Postprocessing done.')


if __name__ == '__main__':
    main()
