import sys
import logging
import configparser
import argparse

import iisr.config_utils as cu
from iisr.plots.run import *
from iisr import IISR_PATH


logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')
DEFAULT_CONFIG_FILE = IISR_PATH / 'iisr' / 'default_plot_config.ini'

description = """
Plot predefined figures.
"""


config = configparser.ConfigParser()


def main(argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('mode', type=str, choices=['spectra_coherence', 'track', 'sky_power',
                                                   'calibration', 'sun_pattern', 'sun_flux'])
    parser.add_argument('-c', '--config-file', default=str(DEFAULT_CONFIG_FILE),
                        help='configuration file')
    args = parser.parse_args(argv)

    if not config.read(args.config_file):
        raise FileNotFoundError('Wrong config')

    cfg_common = config['Common']
    dates = cu.parse_dates_ranges(cfg_common['dates'])
    pre_subfolder = cfg_common['preprocessing_subfolder']
    post_subfolder = cfg_common['postprocessing_subfolder']
    figures_subfolder = cfg_common['figures_subfolder']
    mode = args.mode

    if mode == 'spectra_coherence':
        for date in dates:
            plot_spectra_and_coherence(date, data_subfolder=pre_subfolder,
                                       figures_subfolder=figures_subfolder)
    elif mode == 'track':
        for date in dates:
            plot_processed_tracks(date, data_subfolder=post_subfolder,
                                  figures_subfolder=figures_subfolder)
    elif mode == 'sky_power':
        for date in dates:
            plot_sky_power(date, data_subfolder=post_subfolder,
                           figures_subfolder=figures_subfolder)
    elif mode == 'calibration':
        for date in dates:
            plot_calibration(date, data_subfolder=post_subfolder,
                             figures_subfolder=figures_subfolder)
    elif mode == 'sun_pattern':
        for date in dates:
            plot_sun_pattern_vs_power(date, data_subfolder=post_subfolder,
                                      figures_subfolder=figures_subfolder)
    elif mode == 'sun_flux':
        for date in dates:
            plot_sun_flux(date, data_subfolder=post_subfolder, figures_subfolder=figures_subfolder)

    logging.info('Plotting done.')


if __name__ == '__main__':
    main()
