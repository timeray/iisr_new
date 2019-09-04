from iisr.data_manager import DataManager
from iisr.preprocessing.passive import PassiveScan, PassiveTrack
from iisr.plots.passive import plot_daily_spectra, plot_daily_coherence
import datetime as dt
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')


def main():
    manager = DataManager()

    date = dt.date(2017, 7, 11)

    data_dir = manager.get_preproc_folder_path(date, subfolders=['test'])
    save_dir = manager.get_figures_folder_path(date)

    scan_filepath = data_dir / 'passive_scan_wide.pkl'
    sources = ['sun']
    tracks_filepaths = [data_dir / f'passive_{source_name}_wide.pkl' for source_name in sources]

    scan = PassiveScan.load_pickle(scan_filepath) if scan_filepath.exists() else None
    tracks = [PassiveTrack.load_pickle(filepath) for filepath in tracks_filepaths
              if filepath.exists()]

    logging.info('Plot daily spectra')
    plot_daily_spectra(scan, tracks, save_folder=save_dir, decimation=4)

    logging.info('Plot daily coherence')
    plot_daily_coherence(scan, tracks, save_folder=save_dir, decimation=4)

    logging.info('Plotting done.')


if __name__ == '__main__':
    main()
