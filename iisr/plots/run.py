import datetime as dt
import logging

from iisr.data_manager import DataManager
from iisr.plots import passive as pp
from iisr.postprocessing.passive import SourceTrackInfo, SkyPowerInfo
from iisr.preprocessing.passive import PassiveMode, PassiveScan, PassiveTrack


def plot_spectra_and_coherence(date: dt.date, data_subfolder: str = '', figures_subfolder: str = '',
                               decimation: int = 4):
    with DataManager() as manager:
        data_dir = manager.get_preproc_folder_path(date, subfolders=[data_subfolder])
        save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder])

        scan_filepath = data_dir / 'passive_scan_wide.pkl'
        tracks_filepaths = [data_dir / f'passive_{mode.name}_wide.pkl' for mode in PassiveMode
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


def plot_processed_tracks(date: dt.date, data_subfolder: str = '', figures_subfolder: str = ''):
    with DataManager() as manager:
        data_dir = manager.get_postproc_folder_path(date, subfolders=[data_subfolder])
        save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder, 'Track'])

        tracks_filepaths = [data_dir / f'track_{mode.name}_wide.pkl' for mode in PassiveMode
                            if mode != PassiveMode.scan]
        tracks = [SourceTrackInfo.load_pickle(filepath) for filepath in tracks_filepaths
                  if filepath.exists()]

        if not tracks:
            raise FileNotFoundError(f'No tracks for given date {date}, dirpath={data_dir}')

        logging.info('Plot processed tracks')
        for track in tracks:
            pp.plot_processed_tracks(track, save_folder=save_dir)


def plot_sky_power(date: dt.date, data_subfolder: str = '', figures_subfolder: str = ''):
    with DataManager() as manager:
        data_dir = manager.get_postproc_folder_path(date, subfolders=[data_subfolder])
        save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder])

        sky_power_filepath = data_dir / f'sky_power.pkl'

        if sky_power_filepath.exists():
            sky_power_info = SkyPowerInfo.load_pickle(sky_power_filepath)
        else:
            raise FileNotFoundError(f'No sky power for date {date}, dirpath: {data_dir}')

        logging.info('Plot sky power')
        pp.plot_sky_power(sky_power_info, save_folder=save_dir)