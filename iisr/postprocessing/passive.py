from pathlib import Path
from typing import List

import numpy as np
import pickle as pkl

from iisr.preprocessing.passive import PassiveScan, PassiveTrack
from iisr.data_manager import DataManager


class SourceTrackInfo:
    def __init__(self, track_data: PassiveTrack):
        if len(track_data.dates) > 1:
            raise ValueError('Source track is defined for one date data only')
        self.date = track_data.dates[0]
        self.mode = track_data.parameters.mode
        self.time_marks = track_data.time_marks
        mid_freq_num = track_data.parameters.n_fft // 2
        self.spectra_central_track = {
            ch: sp[:, mid_freq_num] for ch, sp in track_data.spectra.items()
        }
        self.coherence_central_track = track_data.coherence[:, mid_freq_num]

    def save_pickle(self, data_manager: DataManager, subfolders: List[str] = None):
        dirpath = data_manager.get_postproc_folder_path(self.date, subfolders=subfolders)
        filepath = dirpath / f'track_{self.mode.name}_wide.pkl'
        with open(str(filepath), 'wb') as file:
            pkl.dump(self, file)

    @classmethod
    def load_pickle(cls, filepath: Path):
        with open(str(filepath), 'rb') as file:
            return pkl.load(file)