import logging
import datetime as dt

from iisr.data_manager import DataManager
from iisr.postprocessing.passive import SourceTrackInfo, SkyPowerInfo, CalibrationInfo
from iisr.preprocessing.passive import PassiveTrack, PassiveScan
from iisr.preprocessing.passive import PassiveMode


def compute_source_track(date: dt.date, preproc_subfolder: str = '', save_subfolder: str = ''):
    logging.info(f'Compute source track for {date}')
    with DataManager() as manager:
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


def compute_sky_power(date: dt.date, preproc_subfolder: str = '', save_subfolder: str = ''):
    logging.info(f'Compute sky power for {date}')
    with DataManager() as manager:
        data_dir = manager.get_preproc_folder_path(date, subfolders=[preproc_subfolder])

        filepath = data_dir / (PassiveScan.save_name_fmt.format(PassiveMode.scan, 'wide') + '.pkl')
        if filepath.exists():
            scan_result = PassiveScan.load_pickle(filepath)
            sky_power = SkyPowerInfo(scan_result)
            sky_power.save_pickle(manager, subfolders=[save_subfolder])
            return sky_power

        else:
            logging.error(f'File path {filepath} not exists. '
                          f'Maybe there is no processed files for mode {PassiveMode.scan}')


def perform_calibration(date: dt.date, preproc_subfolder: str = '',
                        postproc_subfolder: str = '', save_subfolder: str = None):
    """Perform calibration using scan processing results and model sky power.

    Args:
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
    with DataManager() as manager:
        scan_dir = manager.get_preproc_folder_path(date, subfolders=[preproc_subfolder])
        sky_power_dir = manager.get_postproc_folder_path(date, subfolders=[postproc_subfolder])

        scan_filename = PassiveScan.save_name_fmt.format(PassiveMode.scan, 'wide') + '.pkl'
        scan_filepath = scan_dir / scan_filename
        if scan_filepath.exists():
            scan_date = PassiveScan.load_pickle(scan_filepath)
        else:
            logging.error(f'Scan result at path {scan_filepath} not exist.')
            return

        sky_power_filepath = sky_power_dir / f'sky_power.pkl'
        if sky_power_filepath.exists():
            sky_power = SkyPowerInfo.load_pickle(sky_power_filepath)
        else:
            logging.info(f'Sky power data not found - try to calculate')
            sky_power = compute_sky_power(date, preproc_subfolder, postproc_subfolder)
            if sky_power is None:
                logging.info(f'Sky power cannot be calculated')
                return

        if save_subfolder is None:
            save_subfolder = postproc_subfolder

        CalibrationInfo(scan_date, sky_power).save_pickle(manager, subfolders=[save_subfolder])
