import logging
import datetime as dt

from iisr.data_manager import DataManager
from iisr.postprocessing.passive import SourceTrackInfo, SkyPowerInfo, CalibrationInfo, \
    SunPatternInfo, SunFluxInfo, TRANSITION_TO_TRACK_MODE_DATE, TRANSITION_TO_11_FREQUENCY_MODE_DATE
from iisr.preprocessing.passive import PassiveTrack, PassiveScan
from iisr.preprocessing.passive import PassiveMode


def _load_scan_data(manager: DataManager, date: dt.date, preproc_subfolder: str = ''):
    if date < TRANSITION_TO_11_FREQUENCY_MODE_DATE:
        filepath = manager.get_riometer_filepath(date, 'narrow')
        scan_data = PassiveScan.from_riometer_data(filepath)
    else:
        data_dir = manager.get_preproc_folder_path(date, subfolders=[preproc_subfolder])
        filename = PassiveScan.save_name_fmt.format(PassiveMode.scan, 'wide') + '.pkl'
        filepath = data_dir / filename
        scan_data = PassiveScan.load_pickle(filepath)
    return scan_data


def compute_source_track(manager: DataManager, date: dt.date,
                         preproc_subfolder: str = '', save_subfolder: str = ''):
    logging.info(f'Compute source track for {date}')
    data_dir = manager.get_preproc_folder_path(date, subfolders=[preproc_subfolder])

    for mode in PassiveMode:
        if mode == PassiveMode.scan:
            continue

        filepath = data_dir / (PassiveTrack.save_name_fmt.format(mode.name, 'wide') + '.pkl')

        if not filepath.exists():
            logging.error(f'File path {filepath} not exists. '
                          f'Maybe there is no processed files for mode {mode.name}')
            continue

        track = PassiveTrack.load_pickle(filepath)
        SourceTrackInfo(track).save_pickle(manager, subfolders=[save_subfolder])


def compute_sky_power(manager: DataManager, date: dt.date,
                      preproc_subfolder: str = '', save_subfolder: str = ''):
    logging.info(f'Compute sky power for {date}')

    try:
        scan_result = _load_scan_data(manager, date, preproc_subfolder=preproc_subfolder)
        sky_power = SkyPowerInfo(scan_result)
        sky_power.save_pickle(manager, subfolders=[save_subfolder])
        return sky_power
    except FileNotFoundError:
        logging.error(f'Maybe there is no processed files for mode {PassiveMode.scan}')
        raise


def perform_calibration(manager: DataManager, date: dt.date,
                        preproc_subfolder: str = '', postproc_subfolder: str = '',
                        save_subfolder: str = None):
    """Perform calibration using scan processing results and model sky power.

    Args:
        manager: DataManager
            Data file manager.
        date: dt.date
            Date of experiment.
        preproc_subfolder: str, default ''
            Name of preprocessing directory subfolder. If string is empty, preprocessing directory
            will be used.
        postproc_subfolder: str, default ''
            Name of postprocessing directory subfolder. If string is empty, postprocessing directory
            will be used.
        save_subfolder: str, default None
            Directory where result should be saved. If None - postproc_subfolder will be used.

    """
    logging.info(f'Perform calibration for {date}')
    scan_data = _load_scan_data(manager, date, preproc_subfolder=preproc_subfolder)
    sky_power_dir = manager.get_postproc_folder_path(date, subfolders=[postproc_subfolder])

    sky_power_filepath = sky_power_dir / f'sky_power.pkl'
    if sky_power_filepath.exists():
        sky_power = SkyPowerInfo.load_pickle(sky_power_filepath)
    else:
        logging.info(f'Sky power data not found - try to calculate')
        sky_power = SkyPowerInfo(scan_data)
        sky_power.save_pickle(manager, subfolders=[save_subfolder])

    if save_subfolder is None:
        save_subfolder = postproc_subfolder

    calibration = CalibrationInfo(scan_data, sky_power)
    calibration.save_pickle(manager, subfolders=[save_subfolder])
    return calibration


def compute_sun_pattern(manager: DataManager, date: dt.date,
                        preproc_subfolder: str = '', save_subfolder: str = None):
    logging.info(f'Perform computation of sun pattern for {date}')
    data_dir = manager.get_preproc_folder_path(date, subfolders=[preproc_subfolder])

    if date >= TRANSITION_TO_TRACK_MODE_DATE:
        filename = PassiveTrack.save_name_fmt.format(PassiveMode.sun, 'wide') + '.pkl'
        sun_track_filepath = data_dir / filename
        if sun_track_filepath.exists():
            sun_track_data = PassiveTrack.load_pickle(sun_track_filepath)
        else:
            logging.error(f'Sun track result at path {sun_track_filepath} not exist.')
            return
        sun_pattern = SunPatternInfo(sun_track_data=sun_track_data)
    else:
        scan_data = _load_scan_data(manager, date, preproc_subfolder=preproc_subfolder)
        sun_pattern = SunPatternInfo(scan_data=scan_data)

    sun_pattern.save_pickle(manager, subfolders=[save_subfolder])
    return sun_pattern


def compute_sun_flux(manager: DataManager, date: dt.date,
                     preproc_subfolder: str = '', postproc_subfolder: str = '',
                     save_subfolder: str = None):
    logging.info(f'Perform computation of sun flux for {date}')
    postproc_dir = manager.get_postproc_folder_path(date, subfolders=[postproc_subfolder])

    sun_pattern_filepath = postproc_dir / f'sun_pattern_gaussian_kernel.pkl'
    if sun_pattern_filepath.exists():
        sun_pattern = SunPatternInfo.load_pickle(sun_pattern_filepath)
    else:
        logging.error(f'Sun pattern data not found - try to calculate')
        sun_pattern = compute_sun_pattern(manager, date, preproc_subfolder, postproc_subfolder)
        if sun_pattern is None:
            logging.info(f'Sun pattern cannot be calculated')
            return

    calibration_filepath = postproc_dir / f'scan_calibration__simple_fit_v1.pkl'
    if calibration_filepath.exists():
        calibration = CalibrationInfo.load_pickle(calibration_filepath)
    else:
        logging.info(f'Calibration data not found - try to calculate')
        calibration = perform_calibration(manager, date, preproc_subfolder, postproc_subfolder)
        if calibration is None:
            logging.info(f'Calibration cannot be calculated')
            return

    if save_subfolder is None:
        save_subfolder = postproc_subfolder

    SunFluxInfo(sun_pattern, calibration).save_pickle(manager, subfolders=[save_subfolder])
