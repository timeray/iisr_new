import logging
import datetime as dt

from iisr.data_manager import DataManager
from iisr.postprocessing.passive import SourceTrackInfo, SkyPowerInfo
from iisr.preprocessing.passive import PassiveTrack, PassiveScan
from iisr.preprocessing.passive import PassiveMode


def compute_source_track(date: dt.date, preproc_subfolder: str = '', save_subfolder: str = ''):
    with DataManager() as manager:
        data_dir = manager.get_preproc_folder_path(date, subfolders=[preproc_subfolder])
        save_subfolders = [save_subfolder] if save_subfolder else None

        for mode in PassiveMode:
            if mode == PassiveMode.scan:
                continue

            filepath = data_dir / (PassiveTrack.save_name_fmt.format(mode.name, 'wide') + '.pkl')

            if not filepath.exists():
                logging.info(f'File path {filepath} not exists. '
                             f'Maybe there is no processed files for mode {mode.name}')
                continue

            track = PassiveTrack.load_pickle(filepath)
            SourceTrackInfo(track).save_pickle(manager, subfolders=save_subfolders)


def compute_sky_noise(date: dt.date, preproc_subfolder: str = '', save_subfolder: str = ''):
    with DataManager() as manager:
        data_dir = manager.get_preproc_folder_path(date, subfolders=[preproc_subfolder])
        save_subfolders = [save_subfolder] if save_subfolder else None

        filepath = data_dir / (PassiveScan.save_name_fmt.format(PassiveMode.scan, 'wide') + '.pkl')
        if filepath.exists():
            scan_result = PassiveScan.load_pickle(filepath)
            SkyPowerInfo(scan_result).save_pickle(manager, subfolders=save_subfolders)

        else:
            logging.info(f'File path {filepath} not exists. '
                         f'Maybe there is no processed files for mode {PassiveMode.scan}')
