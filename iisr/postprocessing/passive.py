import numpy as np

from iisr.preprocessing.passive import PassiveScan, PassiveTrack
from iisr.data_manager import DataManager


class SourceTrack:
    def __init__(self, track_data: PassiveTrack):
        self.time_marks = track_data.time_marks
        mid_freq_num = track_data.parameters.n_fft // 2
        self.spectra_central_track = {
            ch: sp[:, mid_freq_num] for ch, sp in track_data.spectra.items()
        }
        self.coherence_central_track = track_data.coherence[:, mid_freq_num]

    def save_txt(self, data_manager: DataManager):
        data_manager.get_figures_folder_path()