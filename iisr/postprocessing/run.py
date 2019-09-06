import logging

from pathlib import Path
from typing import List

from iisr.data_manager import DataManager
from iisr.postprocessing.passive import SourceTrackInfo
from iisr.preprocessing.passive import PassiveTrack
from iisr.preprocessing.passive import PassiveMode


def compute_source_track(dirpaths: List[Path], save_subfolder: str = ''):
    data_manager = DataManager()
    subfolders = [save_subfolder] if save_subfolder else None
    for dirpath in dirpaths:
        for mode in PassiveMode:
            if mode == PassiveMode.scan:
                continue
            filepath = dirpath / (PassiveTrack.save_name_fmt.format(mode.name, 'wide') + '.pkl')
            if not filepath.exists():
                logging.info(f'File path {filepath} not exists. '
                             f'Maybe there is no processed files for mode {mode.name}')
                continue

            track = PassiveTrack.load_pickle(filepath)
            SourceTrackInfo(track).save_pickle(data_manager, subfolders=subfolders)
