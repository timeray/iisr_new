from collections import namedtuple
from pathlib import Path
from typing import List

import numpy as np
import pickle as pkl
import logging

from scipy.constants import Boltzmann

from iisr.antenna.sky_noise import SkyNoiseInterpolator
from iisr.fitting import fit_sky_noise_2d
from iisr.preprocessing.passive import PassiveScan, PassiveTrack, PassiveScanParameters
from iisr.data_manager import DataManager
from iisr.units import Frequency


class PickleLoadable:
    @classmethod
    def load_pickle(cls, filepath: Path):
        with open(str(filepath), 'rb') as file:
            return pkl.load(file)


class SourceTrackInfo(PickleLoadable):
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
        logging.info(f'Dump source track info to {filepath}')
        with open(str(filepath), 'wb') as file:
            pkl.dump(self, file)


class SkyPowerInfo(PickleLoadable):
    def __init__(self, scan_data: PassiveScan):
        if len(scan_data.dates) > 1:
            raise ValueError('Sky power info is defined for one date data only')
        self.date = scan_data.dates[0]

        shape = scan_data.frequencies.size, scan_data.time_marks.size
        params = scan_data.parameters
        self.time_marks = scan_data.time_marks
        self.frequencies = scan_data.frequencies
        self.values = {ch: np.full(shape, np.nan) for ch in params.channels}
        bin_band = Frequency(scan_data.frequencies['Hz'][1] - scan_data.frequencies['Hz'][0], 'Hz')

        for ch in params.channels:
            sky_noise_model = SkyNoiseInterpolator(ch.horn)

            for freq_num, freq_MHz in enumerate(scan_data.frequencies['MHz']):
                for bin_num, time_bin in enumerate(scan_data.time_marks):
                    self.values[ch][freq_num, bin_num] = \
                        sky_noise_model.get_sky_temperature(time_bin, freq_MHz)

            # Temperature to power
            self.values[ch] *= Boltzmann * bin_band['Hz']

    def save_pickle(self, data_manager: DataManager, subfolders: List[str] = None):
        dirpath = data_manager.get_postproc_folder_path(self.date, subfolders=subfolders)
        filepath = dirpath / f'sky_power.pkl'
        logging.info(f'Dump sky power info to {filepath}')
        with open(str(filepath), 'wb') as file:
            pkl.dump(self, file)


class CalibrationInfo(PickleLoadable):
    FitResult2d = namedtuple('FitResults1d', ['dtimes', 'gains', 'biases', 'gain_params'])

    def __init__(self, scan_data: PassiveScan, sky_power: SkyPowerInfo, method: str = 'simple_fit'):
        self.version = 1
        self.method = method

        self.dates = scan_data.dates
        params = scan_data.parameters  # type: PassiveScanParameters

        self.band_masks = self._get_band_masks(scan_data.frequencies, params.central_frequencies)

        # Calculate sky noise power for given date
        sky_power = sky_power.values

        # Fit observed data with sky power
        fit_result = {}
        for ch in scan_data.parameters.channels:
            spectrum = scan_data.spectra[ch]
            fit_result[ch] = self._fit2d(
                scan_data.time_marks, scan_data.frequencies, params.central_frequencies,
                spectrum, sky_power[ch], time_masks=~np.isnan(spectrum)
            )

        self.gains = {ch: result.gains for ch, result in fit_result.items()}
        self.biases = {ch: result.biases for ch, result in fit_result.items()}

    def _get_band_masks(self, spectra_frequencies: Frequency, central_frequencies: Frequency):
        freq_band_nums = np.argmin(
            np.abs(spectra_frequencies['kHz'][:, None] - central_frequencies['kHz']), axis=1
        )
        return [freq_band_nums == band_num for band_num in range(central_frequencies.size)]

    def _fit2d(self, time_marks: np.ndarray, frequencies: Frequency, central_frequencies: Frequency,
               spectrum: np.ndarray, sky_noise: np.ndarray,
               time_masks=None, window=None, timeout=None):
        regression_dtimes = []
        gains = []
        biases = []
        gain_params = []

        if time_masks is None:
            time_masks = np.ones_like(sky_noise, dtype=bool)

        masked_data = np.ma.array(spectrum, mask=~time_masks)
        masked_model = np.ma.array(sky_noise, mask=~time_masks)

        # Split spectrum by bands of each central frequency
        for band_num, central_freq in enumerate(central_frequencies['kHz']):
            logging.info(
                f'[{band_num + 1}/{central_frequencies.size}] fitting for '
                f'{central_freq} kHz central frequency'
            )
            band_mask = self.band_masks[band_num]
            band_freqs = frequencies['kHz'][band_mask]
            dt_reg, gain, params = fit_sky_noise_2d(
                time_marks, band_freqs, central_freq,
                masked_model[band_mask], masked_data[band_mask],
                window=window, timeout=timeout, n_humps=0, vary_freq0=True
            )

            regression_dtimes.append(dt_reg)
            gains.append(gain)
            biases.append(params.pop('bias'))
            gain_params.append(params)

        fit_results = self.FitResult2d(regression_dtimes, gains, biases,
                                       gain_params)
        return fit_results

    def save_pickle(self, data_manager: DataManager, subfolders: List[str] = None):
        for date in self.dates:
            dirpath = data_manager.get_postproc_folder_path(date, subfolders=subfolders)
            filepath = dirpath / f'scan_calibration__{self.method}_v{self.version}.pkl'
            logging.info(f'Dump calibration info to {filepath}')
            with open(str(filepath), 'wb') as file:
                pkl.dump(self, file)
