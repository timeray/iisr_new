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
from iisr.data_manager import DataManager
from iisr.postprocessing.run import compute_source_track

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')
DEFAULT_CONFIG_FILE = IISR_PATH / 'iisr' / 'default_postprocessing.ini'

description = """
Control for second stage processing. Contain multiple types of processing and depend 
on results from first stage.
"""

config = configparser.ConfigParser()


def main(argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('mode', type=str, choices=['track'])
    parser.add_argument('-c', '--config-file', default=str(DEFAULT_CONFIG_FILE),
                        help='configuration file')
    args = parser.parse_args(argv)

    if not config.read(args.config_file):
        raise FileNotFoundError('Wrong config')

    cfg_common = config['Common']
    dates = cu.parse_dates_ranges(cfg_common['dates'])
    subfolder_pre = cfg_common['preprocessing_subfolder']
    subfolder_post = cfg_common['postprocessing_subfolder']
    data_manager = DataManager()
    dirpaths = []
    for date in dates:
        dirpaths.append(data_manager.get_preproc_folder_path(date, subfolders=[subfolder_pre]))

    mode = args.mode
    if mode == 'track':
        compute_source_track(dirpaths, subfolder_post)


if __name__ == '__main__':
    main()
