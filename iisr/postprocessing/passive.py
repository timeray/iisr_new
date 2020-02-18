from collections import namedtuple
from pathlib import Path
from typing import List, Union

import numpy as np
import pickle as pkl
import logging

from tqdm import tqdm
from scipy.constants import Boltzmann, speed_of_light
from scipy.interpolate import interp1d

from iisr.antenna.dnriisr import approximate_directivity
from iisr.antenna.sky_noise import SkyNoiseInterpolator
from iisr.antenna.radio_sources import GaussianKernel, sun_pattern_1d
from iisr.fitting import fit_sky_noise_2d, fit_gauss
from iisr.preprocessing.passive import PassiveScan, PassiveTrack, PassiveMode
from iisr.data_manager import DataManager
from iisr.representation import Channel
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
    n_times = frequencies.shape[0]
    max_freqs_args = []
    for freq_slice, value_slice in tqdm(zip(frequencies['MHz'], np.abs(coherence)),
                                        total=n_times, desc='Find maximum frequencies of track'):
        max_freqs_args.append(fit_gauss_coherence(freq_slice, value_slice)[1].argmax())

    n_freqs = frequencies.shape[1]
    if clip_by is not None:
        min_lo_idx = n_freqs // 2 - clip_by
        max_up_idx = n_freqs // 2 + clip_by
    else:
        min_lo_idx = 0
        max_up_idx = n_freqs - 1

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
        logging.info(f'Dump {__class__.__name__} to {filepath}')
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
        logging.info(f'Dump {__class__.__name__} to {filepath}')
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
        self.central_frequencies = Frequency(
            np.array([f['Hz'] for f in scan_data.parameters.central_frequencies]),
            'Hz'
        )
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

        self._gain_interpolators = {}
        self._bias_interpolators = {}

    def _get_band_masks(self):
        central_frequencies_khz = self.central_frequencies['kHz']
        freq_band_nums = np.argmin(
            np.abs(self.frequencies['kHz'][:, None] - central_frequencies_khz), axis=1
        )
        return [freq_band_nums == band_num for band_num in range(self.central_frequencies.size)]

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
        for band_num, central_freq_khz in enumerate(self.central_frequencies['kHz']):
            logging.info(
                f'[{band_num + 1}/{self.central_frequencies.size}] fitting for '
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

    def interpolate_gain(self, ch: Channel, frequencies: Frequency):
        if ch not in self._gain_interpolators:
            self._gain_interpolators[ch] = interp1d(self.frequencies['MHz'], self.gains[ch],
                                                    kind='cubic', bounds_error=False)
        return self._gain_interpolators[ch](frequencies['MHz'])

    def interpolate_bias(self, ch: Channel, frequencies: Frequency):
        if ch not in self._bias_interpolators:
            self._bias_interpolators[ch] = interp1d(self.central_frequencies['MHz'],
                                                    self.biases[ch],
                                                    kind='cubic', bounds_error=False)
        return self._bias_interpolators[ch](frequencies['MHz'])

    def calibrate_power(self, ch: Channel, frequencies: Frequency, power: Union[float, np.ndarray]):
        gain = self.interpolate_gain(ch, frequencies)
        bias = self.interpolate_bias(ch, frequencies)
        return power / gain - bias

    def save_pickle(self, data_manager: DataManager, subfolders: List[str] = None):
        for date in self.dates:
            dirpath = data_manager.get_postproc_folder_path(date, subfolders=subfolders)
            filepath = dirpath / f'scan_calibration__{self.method}_v{self.version}.pkl'
            logging.info(f'Dump {__class__.__name__} to {filepath}')
            with open(str(filepath), 'wb') as file:
                pkl.dump(self, file)


class SunPatternInfo(PickleLoadable):
    def __init__(self, sun_track_data: PassiveTrack):
        if sun_track_data.parameters.mode != PassiveMode.sun:
            raise ValueError(f'Input data should be sun track, '
                             f'but {sun_track_data.parameters.mode} was given.')

        track = sun_track_data

        if len(track.dates) > 1:
            raise ValueError('Multiple dates processing not implemented')

        self.date = track.dates[0]
        self.time_marks = track.time_marks
        self.pattern_value = None  # Value of the IISR pattern in the Sun position
        self.bin_band = Frequency(
            sun_track_data.frequencies['Hz'][0, 1] - sun_track_data.frequencies['Hz'][0, 0],
            'Hz'
        )
        self.channels = sun_track_data.parameters.channels

        # Fit gaussian coherence to find frequency where maximal Sun intensity supposed to be
        # This should remove variation of pattern caused by temperature
        visible_max_freq_args = find_max_track_frequencies(track.frequencies, track.coherence,
                                                           clip_by=30)

        def take_each(arr: np.ndarray, indices: np.ndarray):
            return np.array([sl[idx] for sl, idx in zip(arr, indices)])

        self.visible_max_freqs = Frequency(
            take_each(track.frequencies['MHz'], visible_max_freq_args), 'MHz'
        )
        self.max_power = {
            ch: take_each(sp, visible_max_freq_args) for ch, sp in track.spectra.items()
        }
        self.convolution_kernel = 'gaussian'
        self.pattern = {}
        self.sky_power = {}
        self.convolved_pattern = {}
        kernel = GaussianKernel()
        self.kernel_integral = kernel.integral
        for ch in track.parameters.channels:
            dnr_type = ch.horn
            pattern, convolved_pattern = sun_pattern_1d(
                self.time_marks, self.visible_max_freqs, dnr_type=dnr_type, kernel=kernel
            )
            self.pattern[ch] = pattern
            self.convolved_pattern[ch] = convolved_pattern

            sky_noise_model = SkyNoiseInterpolator(horn_type=ch.horn)
            track_sky_noise = np.array([
                sky_noise_model.get_sky_temperature(tm, fr)
                for tm, fr in zip(self.time_marks, self.visible_max_freqs['MHz'])
            ])
            self.sky_power[ch] = track_sky_noise * self.bin_band['Hz'] * Boltzmann

    def save_pickle(self, data_manager: DataManager, subfolders: List[str] = None):
        dirpath = data_manager.get_postproc_folder_path(self.date, subfolders=subfolders)
        filepath = dirpath / f'sun_pattern_{self.convolution_kernel}_kernel.pkl'
        logging.info(f'Dump {__class__.__name__} to {filepath}')
        with open(str(filepath), 'wb') as file:
            pkl.dump(self, file)


class SunFluxInfo(PickleLoadable):
    def __init__(self, sun_pattern: SunPatternInfo, calibration: CalibrationInfo):
        self.date = sun_pattern.date
        self.channels = sun_pattern.channels
        self.time_marks = sun_pattern.time_marks
        self.sun_flux = {}

        polarization = 1 / 2  # coefficient to get total flux

        frequencies = sun_pattern.visible_max_freqs
        wavelengths = speed_of_light / frequencies['Hz']
        bandwidth = sun_pattern.bin_band
        integral_over_normalized_brightness = sun_pattern.kernel_integral

        for ch in sun_pattern.channels:
            raw_power = sun_pattern.max_power[ch]
            sky_power = sun_pattern.sky_power[ch]
            cal_raw_power = calibration.calibrate_power(ch, frequencies, raw_power)
            sun_power = cal_raw_power - sky_power
            conv_pattern = sun_pattern.convolved_pattern[ch]

            directivity = approximate_directivity(frequencies['MHz'], dnr_type=ch.horn)
            denominator = conv_pattern \
                          * directivity \
                          * bandwidth['Hz'] \
                          * wavelengths ** 2 \
                          * polarization \
                          / (4 * np.pi)

            self.sun_flux[ch] = sun_power * integral_over_normalized_brightness / denominator

    def save_pickle(self, data_manager: DataManager, subfolders: List[str] = None):
        dirpath = data_manager.get_postproc_folder_path(self.date, subfolders=subfolders)
        filepath = dirpath / f'sun_flux.pkl'
        logging.info(f'Dump {__class__.__name__} info to {filepath}')
        with open(str(filepath), 'wb') as file:
            pkl.dump(self, file)
