from collections import namedtuple
from pathlib import Path
from typing import List, Union

import numpy as np
import pickle as pkl
import logging

from scipy.constants import Boltzmann
from scipy.interpolate import interp1d

from iisr.antenna.sky_noise import SkyNoiseInterpolator
from iisr.fitting import fit_sky_noise_2d, fit_gauss
from iisr.preprocessing.passive import PassiveScan, PassiveTrack, PassiveScanParameters, PassiveMode
from iisr.data_manager import DataManager
from iisr.units import Frequency


def fit_gauss_coherence(frequencies_megahertz, coherence):
    return fit_gauss(
        frequencies_megahertz, coherence,
        a_min=0., a_max=1.,
        b_min=0., b_max=1.,
        var_max=0.25
    )


def fit_gauss_power(frequencies_megahertz, power):
    return fit_gauss(
        frequencies_megahertz, power,
        a_min=0., a_max=power.max() * 2,
        b_min=0., b_max=power.max(),
        var_max=0.25
    )


def find_max_track_frequencies(frequencies: Frequency, coherence: np.ndarray, clip_by: int = None):
    max_freqs_args = []
    for freq_slice, value_slice in zip(frequencies['MHz'].T, np.abs(coherence.T)):
        max_freqs_args.append(fit_gauss_coherence(freq_slice, value_slice)[1].argmax())

    n_freqs = frequencies.shape[1]
    min_lo_idx = n_freqs // 2 - clip_by
    max_up_idx = n_freqs // 2 + clip_by
    args = np.array(max_freqs_args)
    args.clip(min_lo_idx, max_up_idx, out=args)
    return args


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

        shape = scan_data.time_marks.size, scan_data.frequencies.size
        params = scan_data.parameters
        self.time_marks = scan_data.time_marks
        self.frequencies = scan_data.frequencies
        self.values = {ch: np.full(shape, np.nan) for ch in params.channels}
        bin_band = Frequency(scan_data.frequencies['Hz'][1] - scan_data.frequencies['Hz'][0], 'Hz')

        for ch in params.channels:
            sky_noise_model = SkyNoiseInterpolator(ch.horn)

            for freq_num, freq_MHz in enumerate(scan_data.frequencies['MHz']):
                for bin_num, time_bin in enumerate(scan_data.time_marks):
                    self.values[ch][bin_num, freq_num] = \
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
        self._gain_interpolator = None
        self._bias_interpolator = None

        self.dates = scan_data.dates

        self.frequencies = scan_data.frequencies
        self.central_frequencies = scan_data.parameters.central_frequencies
        self.band_masks = self._get_band_masks()

        # Calculate sky noise power for given date
        sky_power = sky_power.values

        # Fit observed data with sky power
        fit_result = {}
        for ch in scan_data.parameters.channels:
            spectrum = scan_data.spectra[ch]
            fit_result[ch] = self._fit2d(
                scan_data.time_marks, spectrum, sky_power[ch], time_masks=~np.isnan(spectrum)
            )
        self.gains = {ch: result.gains for ch, result in fit_result.items()}
        self.biases = {ch: result.biases for ch, result in fit_result.items()}

    def _get_band_masks(self):
        central_frequencies_khz = np.array([fr['kHz'] for fr in self.central_frequencies])
        freq_band_nums = np.argmin(
            np.abs(self.frequencies['kHz'][:, None] - central_frequencies_khz), axis=1
        )
        return [freq_band_nums == band_num for band_num in range(len(self.central_frequencies))]

    def _fit2d(self, time_marks: np.ndarray, spectrum: np.ndarray, sky_noise: np.ndarray,
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
        for band_num, central_freq in enumerate(self.central_frequencies):
            central_freq_khz = central_freq['kHz']
            logging.info(
                f'[{band_num + 1}/{len(self.central_frequencies)}] fitting for '
                f'{central_freq_khz} kHz central frequency'
            )
            band_mask = self.band_masks[band_num]
            band_freqs = self.frequencies['kHz'][band_mask]
            dt_reg, gain, params = fit_sky_noise_2d(
                time_marks, band_freqs, central_freq_khz,
                masked_model[:, band_mask], masked_data[:, band_mask],
                window=window, timeout=timeout, n_humps=0, vary_freq0=True
            )

            regression_dtimes.append(dt_reg)
            gains.append(np.squeeze(gain))
            biases.append(params.pop('bias'))
            gain_params.append(params)

        fit_results = self.FitResult2d(regression_dtimes, np.concatenate(gains),
                                       np.concatenate(biases), gain_params)
        return fit_results

    @property
    def gain_interpolator(self):
        if self._gain_interpolator is None:
            self._gain_interpolator = interp1d(self.frequencies['MHz'], self.gains,
                                               kind='cubic', bounds_error=False)
        else:
            return self._gain_interpolator

    @property
    def bias_interpolator(self):
        if self._bias_interpolator is None:
            self._bias_interpolator = interp1d(self.frequencies['MHz'], self.biases,
                                               kind='cubic', bounds_error=False)
        else:
            return self._bias_interpolator

    def calibrate_power(self, frequency: Frequency, power: Union[float, np.ndarray]):
        gain = self.gain_interpolator(frequency['MHz'])
        bias = self.bias_interpolator(frequency['MHz'])
        return power / gain - bias

    def save_pickle(self, data_manager: DataManager, subfolders: List[str] = None):
        for date in self.dates:
            dirpath = data_manager.get_postproc_folder_path(date, subfolders=subfolders)
            filepath = dirpath / f'scan_calibration__{self.method}_v{self.version}.pkl'
            logging.info(f'Dump calibration info to {filepath}')
            with open(str(filepath), 'wb') as file:
                pkl.dump(self, file)


class SunTrackInfo(PickleLoadable):
    def __init__(self, sun_track_data: PassiveTrack):
        if sun_track_data.parameters.mode != PassiveMode.sun:
            raise ValueError(f'Input data should be sun track, '
                             f'but {sun_track_data.parameters.mode} was given.')

        track = sun_track_data
        self.time_marks = track.time_marks
        self.pattern_value = None  # Value of the IISR pattern in the Sun position

        # Fit gaussian coherence to find frequency where maximal Sun intensity supposed to be
        # This should remove variation of pattern caused by temperature
        visible_max_freq_args = find_max_track_frequencies(track.frequencies, track.coherence)

        def take_each(arr: np.ndarray, indices: np.ndarray):
            return np.array([sl[idx] for sl, idx in zip(arr, indices)])

        self.visible_max_freqs = Frequency(
            take_each(track.frequencies['MHz'], visible_max_freq_args), 'MHz'
        )
        self.max_power = {
            ch: take_each(sp, visible_max_freq_args) for ch, sp in track.spectra.items()
        }
        # TODO: test both get_sun from astropy and sky_position (equinox_of_date=False) from sunpy
