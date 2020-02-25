import datetime as dt
import logging

from iisr.data_manager import DataManager
from iisr.plots import passive as pp
from iisr.postprocessing.passive import SourceTrackInfo, SkyPowerInfo, CalibrationInfo, \
    SunPatternInfo, SunFluxInfo
from iisr.preprocessing.passive import PassiveMode, PassiveTrack
from iisr.postprocessing.run import _load_scan_data


__all__ = ['plot_spectra_and_coherence', 'plot_sky_power', 'plot_calibration',
           'plot_processed_tracks', 'plot_sun_pattern_vs_power', 'plot_sun_flux']


def plot_spectra_and_coherence(manager: DataManager, date: dt.date,
                               data_subfolder: str = '', figures_subfolder: str = '',
                               decimation: int = 4):
    data_dir = manager.get_preproc_folder_path(date, subfolders=[data_subfolder])
    save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder])

    tracks_filepaths = [data_dir / f'passive_{mode.name}_wide.pkl' for mode in PassiveMode
                        if mode != PassiveMode.scan]

    try:
        scan = _load_scan_data(manager, date, preproc_subfolder=data_subfolder)
    except FileNotFoundError:
        scan = None

    tracks = [PassiveTrack.load_pickle(filepath) for filepath in tracks_filepaths
              if filepath.exists()]

    if scan is None and not tracks:
        raise FileNotFoundError(f'No scan/track data for {date}, dirpath={data_dir}')

    logging.info('Plot daily spectra')
    pp.plot_daily_spectra(scan, tracks, save_folder=save_dir, decimation=decimation)

    logging.info('Plot daily coherence')
    pp.plot_daily_coherence(scan, tracks, save_folder=save_dir, decimation=decimation)


def plot_processed_tracks(manager: DataManager, date: dt.date,
                          data_subfolder: str = '', figures_subfolder: str = ''):
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


def plot_sky_power(manager: DataManager, date: dt.date,
                   data_subfolder: str = '', figures_subfolder: str = ''):
    data_dir = manager.get_postproc_folder_path(date, subfolders=[data_subfolder])
    save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder])

    sky_power_filepath = data_dir / f'sky_power.pkl'

    if sky_power_filepath.exists():
        sky_power_info = SkyPowerInfo.load_pickle(sky_power_filepath)
    else:
        raise FileNotFoundError(f'No sky power for date {date}, dirpath: {data_dir}')

    logging.info('Plot sky power')
    pp.plot_sky_power(sky_power_info, save_folder=save_dir)


def plot_calibration(manager: DataManager, date: dt.date,
                     data_subfolder: str = '', figures_subfolder: str = ''):
    data_dir = manager.get_postproc_folder_path(date, subfolders=[data_subfolder])
    save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder])

    calibration_filepath = data_dir / f'scan_calibration__simple_fit_v1.pkl'

    if calibration_filepath.exists():
        calibration_info = CalibrationInfo.load_pickle(calibration_filepath)
    else:
        raise FileNotFoundError(f'No calibration for date {date}, dirpath: {data_dir}')

    logging.info('Plot calibration')
    pp.plot_calibration(calibration_info, save_folder=save_dir)


def plot_sun_pattern_vs_power(manager: DataManager, date: dt.date,
                              data_subfolder: str = '', figures_subfolder: str = ''):
    data_dir = manager.get_postproc_folder_path(date, subfolders=[data_subfolder])
    save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder])

    filepath = data_dir / f'sun_pattern_gaussian_kernel.pkl'

    if filepath.exists():
        track = SunPatternInfo.load_pickle(filepath)
    else:
        raise FileNotFoundError(f'No sun track for given date {date}, dirpath={data_dir}')

    logging.info('Plot sun pattern info')
    pp.plot_sun_pattern_vs_power(track, save_folder=save_dir)


def plot_sun_flux(manager: DataManager, date: dt.date,
                  data_subfolder: str = '', figures_subfolder: str = ''):
    data_dir = manager.get_postproc_folder_path(date, subfolders=[data_subfolder])
    save_dir = manager.get_figures_folder_path(date, subfolders=[figures_subfolder])

    filepath = data_dir / f'sun_flux.pkl'

    if filepath.exists():
        sun_flux_info = SunFluxInfo.load_pickle(filepath)
    else:
        raise FileNotFoundError(f'No sun track for given date {date}, dirpath={data_dir}')

    logging.info('Plot sun flux')
    pp.plot_sun_flux(sun_flux_info, save_folder=save_dir)

