from collections import namedtuple
from pathlib import Path
from typing import List, Union, Dict

import datetime as dt
import numpy as np
import pickle as pkl
import logging

from tqdm import tqdm
from scipy.constants import Boltzmann, speed_of_light
from scipy.interpolate import interp1d

from iisr.antenna.dnriisr import approximate_directivity
from iisr.antenna.sky_noise import SkyNoiseInterpolator
from iisr.antenna.radio_sources import GaussianKernel, sun_pattern_1d, find_sun_max_frequencies
from iisr.fitting import fit_sky_noise_2d, fit_gauss, fit_sky_noise_1d
from iisr.preprocessing.passive import PassiveScan, PassiveTrack, PassiveMode, \
    PassiveTrackParameters
from iisr.data_manager import DataManager
from iisr.representation import Channel
from iisr.units import Frequency


TRANSITION_TO_11_FREQUENCY_MODE_DATE = dt.date(2015, 6, 17)
TRANSITION_TO_TRACK_MODE_DATE = dt.date(2017, 2, 7)


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
        if np.isnan(value_slice).any():
            max_freqs_args.append(freq_slice.size // 2)
        else:
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
    FitResult1d = namedtuple('FitResults1d', ['gains', 'biases'])
    FitResult2d = namedtuple('FitResults2d', ['dtimes', 'gains', 'biases', 'gain_params'])

    def __init__(self, scan_data: PassiveScan, sky_power: SkyPowerInfo, method: str = 'simple_fit'):
        for ch in scan_data.spectra.keys():
            if scan_data.spectra[ch].shape != sky_power.values[ch].shape:
                raise ValueError('Cannot calibrate data - scan data and sky power '
                                 'have different shapes')

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
        if self.frequencies.size == self.central_frequencies.size:
            self.apply_1d_fit = True
            self.band_masks = None
        else:
            self.apply_1d_fit = False
            self.band_masks = self._get_band_masks()

        # Calculate sky noise power for given date
        sky_power = sky_power.values

        # Fit observed data with sky power
        fit_result = {}
        for ch in scan_data.parameters.channels:
            spectrum = scan_data.spectra[ch]

            if self.apply_1d_fit:
                fit_result[ch] = self._fit1d(
                    scan_data.time_marks, spectrum, sky_power[ch], time_mask=~np.isnan(spectrum)
                )
            else:
                fit_result[ch] = self._fit2d(
                    scan_data.time_marks, spectrum, sky_power[ch], time_mask=~np.isnan(spectrum)
                )
        self.gains = {ch: result.gains for ch, result in fit_result.items()}
        self.biases = {ch: result.biases for ch, result in fit_result.items()}

        self.gains = {ch: np.ma.array(a, mask=np.isnan(a)) for ch, a in self.gains.items()}
        self.biases = {ch: np.ma.array(a, mask=np.isnan(a)) for ch, a in self.biases.items()}

        self._gain_interpolators = {}
        self._bias_interpolators = {}

        self.channels = scan_data.parameters.channels
        self.time_marks = scan_data.time_marks
        self.calibrated_spectra = {
            ch: self.calibrate_power(ch, self.frequencies, scan_data.spectra[ch])
            for ch in self.channels
        }

    def _get_band_masks(self):
        central_frequencies_khz = self.central_frequencies['kHz']
        freq_band_nums = np.argmin(
            np.abs(self.frequencies['kHz'][:, None] - central_frequencies_khz), axis=1
        )
        return [freq_band_nums == band_num for band_num in range(self.central_frequencies.size)]

    def _fit1d(self, time_marks: np.ndarray, spectrum: np.ndarray, sky_noise: np.ndarray,
               time_mask=None, window=None, timeout=None):
        gains = []
        biases = []

        if time_mask is None:
            time_mask = np.ones_like(time_marks, dtype=bool)

        masked_data = np.ma.array(spectrum, mask=~time_mask)
        masked_model = np.ma.array(sky_noise, mask=~time_mask)

        for data, model in zip(masked_data.T, masked_model.T):
            _, gain, bias, _ = fit_sky_noise_1d(time_marks, model, data,
                                                window=window, timeout=timeout)
            gains.append(gain)
            biases.append(bias / gain)

        return self.FitResult1d(np.array(gains), np.array(biases))

    def _fit2d(self, time_marks: np.ndarray, spectrum: np.ndarray, sky_noise: np.ndarray,
               time_mask=None, window=None, timeout=None):
        regression_dtimes = []
        gains = []
        biases = []
        gain_params = []

        if time_mask is None:
            time_mask = np.ones_like(sky_noise, dtype=bool)

        masked_data = np.ma.array(spectrum, mask=~time_mask)
        masked_model = np.ma.array(sky_noise, mask=~time_mask)

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
            mask = self.gains[ch].mask
            self._gain_interpolators[ch] = interp1d(self.frequencies['MHz'][~mask],
                                                    self.gains[ch].compressed(),
                                                    kind='cubic', bounds_error=False)
        return self._gain_interpolators[ch](frequencies['MHz'])

    def interpolate_bias(self, ch: Channel, frequencies: Frequency):
        if ch not in self._bias_interpolators:
            mask = self.biases[ch].mask
            self._bias_interpolators[ch] = interp1d(self.central_frequencies['MHz'][~mask],
                                                    self.biases[ch].compressed(),
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
    class PseudoPassiveTrack:
        class Params:
            pass

        def __init__(self, dates, time_marks: np.ndarray, frequencies: Frequency,
                     spectra: Dict[Channel, np.ndarray], coherence: np.ndarray,
                     params: PassiveTrackParameters):
            self.dates = dates
            self.time_marks = time_marks
            self.frequencies = frequencies
            self.spectra = spectra
            self.coherence = coherence
            self.parameters = params

    def __init__(self, *, sun_track_data: PassiveTrack = None, scan_data: PassiveScan = None):
        if (sun_track_data is None and scan_data is None) \
                or (sun_track_data is not None and scan_data is not None):
            raise ValueError('Either sun track or scan data should be provided')

        if sun_track_data is not None:
            if sun_track_data.parameters.mode != PassiveMode.sun:
                raise ValueError(f'Input data should be sun track, '
                                 f'but {sun_track_data.parameters.mode} was given.')

            track = sun_track_data
        elif scan_data is not None:
            track = self._extract_sun_track_from_scan_data(scan_data)
        else:
            raise AssertionError('Unexpected behavior')

        if len(track.dates) > 1:
            raise ValueError('Multiple dates processing not implemented')

        self.date = track.dates[0]
        self.time_marks = track.time_marks
        self.pattern_value = None  # Value of the IISR pattern in the Sun position
        self.bin_band = Frequency(
            track.frequencies['Hz'][0, 1] - track.frequencies['Hz'][0, 0],
            'Hz'
        )
        self.channels = track.parameters.channels

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

    def _extract_sun_track_from_scan_data(self, scan_data: PassiveScan) -> PseudoPassiveTrack:
        # Find frequencies with expected maximum power from a source at the position of the Sun
        expected_max_freqs = find_sun_max_frequencies(scan_data.time_marks)

        # Find arguments of frequencies, closest to the found maximum, in the scan data
        args = self._find_closest(expected_max_freqs['MHz'], scan_data.frequencies['MHz'])

        # Cut the 1 MHz Sun track from coherence and power data
        margin_frequency = 1  # MHz
        freq_delta = scan_data.frequencies['MHz'][1] - scan_data.frequencies['MHz'][0]
        margin = int(margin_frequency / freq_delta)

        if margin % 2 == 0:
            margin_args = [margin // 2, margin // 2]
        else:
            margin_args = [margin // 2, margin // 2 + 1]

        shape = len(scan_data.time_marks), margin
        # Frequencies should not have nans
        frequencies = np.stack(
            [np.linspace(152, 152 + margin * freq_delta, margin, endpoint=False)
             for _ in range(shape[0])]
        )

        coherence = np.full(shape, np.nan, dtype=np.complex)
        spectra = {ch: np.full(shape, np.nan, dtype=np.float)
                   for ch in scan_data.parameters.channels}
        n_scan_freqs = scan_data.frequencies.size

        for time_num, arg in enumerate(args):
            if arg == 0 or arg == n_scan_freqs - 1:
                # Then our track is out of range
                continue

            low_old = max(arg - margin_args[0], 0)
            upp_old = min(arg + margin_args[1], n_scan_freqs)
            sl_old = slice(low_old, upp_old)

            low_new = max(-(arg - margin_args[1]), 0)
            upp_new = margin - max(arg + margin_args[1] - n_scan_freqs, 0)
            sl_new = slice(low_new, upp_new)

            # Extrapolate frequencies if some indexes fell out of range
            if sl_old.stop - sl_old.start == margin:
                frequencies[time_num] = scan_data.frequencies['MHz'][sl_old]
            elif upp_old == n_scan_freqs:
                # Upper boundary
                low_freq = scan_data.frequencies['MHz'][low_old]
                upp_freq = low_freq + margin * freq_delta
                frequencies[time_num] = np.linspace(low_freq, upp_freq, margin, endpoint=False)
            else:
                # Lower boundary
                upp_freq = scan_data.frequencies['MHz'][upp_old]
                low_freq = upp_freq - (margin - 1) * freq_delta
                frequencies[time_num] = np.linspace(low_freq, upp_freq, margin)

            coherence[time_num, sl_new] = scan_data.coherence[time_num, sl_old]
            for ch in scan_data.parameters.channels:
                spectra[ch][time_num, sl_new] = scan_data.spectra[ch][time_num, sl_old]
        frequencies = Frequency(frequencies, 'MHz')

        scan_params = scan_data.parameters
        params = PassiveTrackParameters(
            mode=PassiveMode.sun,
            sampling_frequency=scan_params.sampling_frequency,
            n_accumulation=scan_params.n_accumulation,
            n_fft=margin,
            channels=scan_params.channels,
            band_type=scan_params.band_type
        )
        return self.PseudoPassiveTrack(scan_data.dates, scan_data.time_marks,
                                       frequencies, spectra, coherence, params)

    @staticmethod
    def _find_closest(ref1d, values2d):
        return np.argmin(np.abs(values2d - ref1d[:, None]), axis=1)

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
