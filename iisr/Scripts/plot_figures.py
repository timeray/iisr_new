from iisr.data_manager import DataManager
from iisr.postprocessing.passive import SourceTrackInfo
from iisr.preprocessing.passive import PassiveScan, PassiveTrack, PassiveMode
import iisr.plots.passive as pp
import datetime as dt
import logging
import traceback
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(message)s')


def plot_spectra_and_coherence(date: dt.date, subfolder: str = '', decimation: int = 4):
    with DataManager() as manager:
        data_dir = manager.get_preproc_folder_path(date, subfolders=[subfolder])
        save_dir = manager.get_figures_folder_path(date)

        scan_filepath = data_dir / 'passive_scan_wide.pkl'
        tracks_filepaths = [data_dir / f'passive_{mode}_wide.pkl' for mode in PassiveMode
                            if mode != PassiveMode.scan]

        scan = PassiveScan.load_pickle(scan_filepath) if scan_filepath.exists() else None
        tracks = [PassiveTrack.load_pickle(filepath) for filepath in tracks_filepaths
                  if filepath.exists()]

        if scan is None and not tracks:
            raise FileNotFoundError(f'No scan/track data for {date}, dirpath={data_dir}')

        logging.info('Plot daily spectra')
        pp.plot_daily_spectra(scan, tracks, save_folder=save_dir, decimation=decimation)

        logging.info('Plot daily coherence')
        pp.plot_daily_coherence(scan, tracks, save_folder=save_dir, decimation=decimation)


def plot_processed_tracks(date: dt.date, subfolder: str = ''):
    with DataManager() as manager:
        data_dir = manager.get_postproc_folder_path(date, subfolders=[subfolder])
        save_dir = manager.get_figures_folder_path(date, subfolders=['Track'])

        tracks_filepaths = [data_dir / f'track_{mode}_wide.pkl' for mode in PassiveMode
                            if mode != PassiveMode.scan]
        tracks = [SourceTrackInfo.load_pickle(filepath) for filepath in tracks_filepaths
                  if filepath.exists()]

        if not tracks:
            raise FileNotFoundError(f'No tracks for given date {date}, dirpath={data_dir}')

        logging.info('Plot processed tracks')
        for track in tracks:
            pp.plot_processed_tracks(track, save_folder=save_dir)


def main():
    subfolder_pre = 'test'
    subfolder_post = 'test'

    dates = [dt.date(2017, 7, 12)]

    for date in dates:
        try:
            plot_spectra_and_coherence(date=date, subfolder=subfolder_pre)
            plot_processed_tracks(date=date, subfolder=subfolder_post)
        except FileNotFoundError:
            logging.error(traceback.format_exc())
            logging.info('Continue plotting...')

    logging.info('Plotting done.')


if __name__ == '__main__':
    main()
