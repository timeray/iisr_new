"""
Command line program for second stage processing.
Receive type of processing, dates or ID of first stage results, and other option.
"""
import configparser
import sys
import logging
import argparse

from iisr import IISR_PATH
import iisr.config_utils as cu
from iisr.postprocessing.run import compute_source_track, compute_sky_noise

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')
DEFAULT_CONFIG_FILE = IISR_PATH / 'iisr' / 'default_postprocessing.ini'

description = """
Control for second stage processing. Contain multiple types of processing and depend 
on results from first stage.
"""

config = configparser.ConfigParser()


def main(argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('mode', type=str, choices=['track', 'sky_power'])
    parser.add_argument('-c', '--config-file', default=str(DEFAULT_CONFIG_FILE),
                        help='configuration file')
    args = parser.parse_args(argv)

    if not config.read(args.config_file):
        raise FileNotFoundError('Wrong config')

    cfg_common = config['Common']
    dates = cu.parse_dates_ranges(cfg_common['dates'])
    subfolder_pre = cfg_common['preprocessing_subfolder']
    subfolder_post = cfg_common['postprocessing_subfolder']

    mode = args.mode
    if mode == 'track':
        for date in dates:
            compute_source_track(date, subfolder_pre, subfolder_post)
    elif mode == 'sky_power':
        for date in dates:
            compute_sky_noise(date, subfolder_pre, subfolder_post)

    logging.info('Postprocessing done.')


if __name__ == '__main__':
    main()
