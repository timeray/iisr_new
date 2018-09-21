from collections import defaultdict
import datetime as dtime

import numpy as np
from typing import List, TextIO, Dict, Sequence

from iisr.io import SeriesParameters, FileInfo
from iisr.preprocessing.representation import HandlerResult, Handler, ResultParameters, Supervisor
from iisr.representation import Channel, ADJACENT_CHANNELS
from iisr.units import Frequency
from iisr.utils import central_time


SOURCES_FILE_CODES = {
    1: 'cyg',
    2: 'crab',
    4: 'cass',
    8: 'sun',
}


def get_source_type(file_info: FileInfo) -> str:
    """Return type of passive data given information in file name.

    Args:
        file_info: Information from file name.

    Returns:
        passive_type: Name of passive data type.
    """
    if file_info.field4 == 0:
        return 'scan'
    else:
        return SOURCES_FILE_CODES[file_info.field1]


def calc_ref_band(n_fft: int, sampling_frequency: Frequency) -> Frequency:
    """Calculate reference frequency band.

    Args:
        n_fft: Number of points in fft.
        sampling_frequency: Sampling frequency.

    Returns:
        band: Array of frequencies.
    """
    period = 1 / sampling_frequency['Hz']
    return Frequency(np.fft.fftshift(np.fft.fftfreq(n_fft, period)), 'Hz')


class PassiveParameters(ResultParameters):
    params_to_save = ['sampling_frequency', 'n_samples', 'n_fft', 'channels_set']

    def __init__(self, sampling_frequency: Frequency, n_avg: int, n_fft: int,
                 channels: List[Channel]):
        self.sampling_frequency = sampling_frequency
        self.n_avg = n_avg
        self.n_fft = n_fft

        if not channels:
            raise ValueError('Empty channels_set')

        self.channels = channels

    @classmethod
    def load_txt(cls, file: TextIO):
        params = cls.read_params_from_txt(file)
        return cls(**params)


class PassiveScanResult(HandlerResult):
    mode_name = 'passive_scan'

    def __init__(self, date: dtime.date, parameters: PassiveParameters, time_marks: np.ndarray,
                 frequencies: Frequency, central_frequencies: Frequency,
                 spectra: Dict[Channel, np.ndarray], coherence: np.ndarray = None):
        self.dates = date  # Limit handler to a single date to save space for large arrays

        if spectra is not None:
            if sorted(spectra.keys()) != parameters.channels:
                raise AssertionError('Channels in spectra dictionary must be identical '
                                     'to channels_set in parameters object')

            for spectrum in spectra.values():
                if spectrum.shape != (len(time_marks), frequencies.size):
                    raise ValueError('Expected shape for spectrum: [n_times x n_freqs]')

        if coherence is not None:
            if coherence.shape != (len(time_marks), frequencies.size):
                raise ValueError('Expected shape for coherence: [n_times x n_freqs]')

            if not np.iscomplexobj(coherence):
                raise ValueError('Expect coherence to be complex valued')

        self.parameters = parameters
        self.time_marks = np.array(time_marks, dtype=dtime.datetime)
        self.frequencies = frequencies
        self.central_frequencies = central_frequencies
        self.spectra = spectra
        self.coherence = coherence

    @property
    def short_name(self) -> str:
        return 'ch{}'.format(self.parameters.channels)


class PassiveTrackResult(HandlerResult):
    mode_name = 'passive_track'

    def __init__(self, date, source_name: str, parameters: PassiveParameters,
                 time_marks: np.ndarray, central_frequencies: Frequency,
                 spectra: Dict[Channel, np.ndarray], coherence: np.ndarray = None):
        self.dates = date  # Limit handler to a single date to save space for large arrays

        if central_frequencies.size != time_marks.size:
            raise ValueError('Expect central frequencies to be 1-d array with value for each '
                             'time mark')

        if spectra is not None:
            if sorted(spectra.keys()) != parameters.channels:
                raise AssertionError('Channels in spectra dictionary must be identical '
                                     'to channels_set in parameters object')

            for spectrum in spectra.values():
                if spectrum.shape != (len(time_marks), parameters.n_fft):
                    raise ValueError('Expected shape for spectrum: [n_times x n_freqs]')

        if coherence is not None:
            if coherence.shape != (len(time_marks), parameters.n_fft):
                raise ValueError('Expected shape for coherence: [n_times x n_freqs]')

            if not np.iscomplexobj(coherence):
                raise ValueError('Expect coherence to be complex valued')

        self.source_name = source_name
        self.parameters = parameters
        self.time_marks = np.array(time_marks, dtype=dtime.datetime)
        self._frequencies = None
        self.central_frequencies = central_frequencies
        self.spectra = spectra
        self.coherence = coherence

    @property
    def n_fft(self):
        return self.parameters.n_fft

    @property
    def sampling_frequency(self):
        return self.parameters.sampling_frequency

    @property
    def frequencies(self):
        """Frequencies array of size [n_times x n_fft].

        Differs from scan-mode frequencies, because frequencies vary with time.
        It is large array so its evaluated on demand.
        """
        if self._frequencies is None:
            ref_band = calc_ref_band(self.n_fft, self.sampling_frequency)
            self._frequencies = np.empty((len(self.time_marks), self.n_fft), dtype=float)
            for time_num, freq in enumerate(self.central_frequencies['Hz']):
                self._frequencies[time_num] = freq + ref_band

        return self._frequencies


class PassiveHandler(Handler):
    valid_source_types = NotImplemented

    def __init__(self, date, n_avg, n_fft, eval_spectra=True, eval_coherence=True):
        self.date = date  # Limit handler to a single date to save space for large arrays
        self.n_avg = n_avg
        self.n_fft = n_fft

        self.eval_spectra = eval_spectra
        self.eval_coherence = eval_coherence

        self.await_input = None
        self.valid_channels = None
        self.channels_set = None
        self.valid_source_type = None
        self.sampling_frequency = None

        self.time_marks = None
        self.central_frequencies = None
        self.spectra = None
        self.coherence = None

        self.initialized = False

    def reset_buffers(self):
        self.channels_set = set()
        self.valid_channels = set()
        self.valid_source_type = None
        self.await_input = {}

        self.time_marks = defaultdict([])
        self.central_frequencies = set()
        # type: Dict[Channel, Dict[Frequency, List[np.ndarray]]]
        self.spectra = defaultdict(lambda: defaultdict(list)) if self.eval_spectra else None
        # type: Dict[Frequency, List[np.ndarray]]
        self.coherence = defaultdict(list) if self.eval_coherence else None

    def init(self, params: SeriesParameters):
        self.reset_buffers()
        self.initialized = True
        self.valid_channels.add(params.channel)
        self.valid_channels.add(ADJACENT_CHANNELS[params.channel])
        self.valid_source_type = get_source_type(params.file_info)
        self.sampling_frequency = params.sampling_frequency

    @property
    def channels(self):
        """Sorted list of channels"""
        return sorted(self.channels_set)

    @property
    def ref_band(self) -> Frequency:
        """Reference frequency band"""
        return calc_ref_band(self.n_fft, self.sampling_frequency)

    def validate(self, params: SeriesParameters) -> bool:
        if get_source_type(params.file_info) != self.valid_source_type:
            return False

        if params.channel not in self.valid_channels:
            return False

        return True

    def calc_power_spectra(self, fft: np.ndarray) -> np.ndarray:
        power_spectra = np.abs(fft)  # new array
        np.power(power_spectra, 2, out=power_spectra)

        power_spectra = np.mean(power_spectra.real, axis=0)
        # spectra_std = np.std(power_spectra.real, axis=0)

        power_spectra /= self.n_fft ** 2
        # spectra_std /= self.n_fft ** 2
        return power_spectra

    def calc_spectral_coherence(self, fft: Dict[Channel, np.ndarray],
                                power_spectra: Dict[Channel, np.ndarray]) -> np.ndarray:
        scale = self.n_fft ** 2

        fft1 = fft[self.channels[0]]
        fft2 = fft[self.channels[1]]

        power_spectra1 = power_spectra[self.channels[0]] * scale
        power_spectra2 = power_spectra[self.channels[1]] * scale

        cross_spectra = fft1 * fft2.conj()
        cross_spectra_mean = cross_spectra.mean(axis=0)

        return cross_spectra_mean / np.sqrt(power_spectra1 * power_spectra2)

    def _process_quads(self, central_frequency, time_marks: np.ndarray,
                       quadratures: Dict[Channel, np.ndarray]):
        self.time_marks[central_frequency].append(central_time(time_marks))

        # Reshape quadratures
        for ch in self.channels:
            quadratures[ch] = quadratures[ch].reshape(self.n_avg, self.n_fft)

        channels_fft = {}
        power_spectra = {}
        if self.eval_spectra or self.eval_coherence:
            for ch in self.channels:
                fft = self.calc_fft(quadratures[ch], axis=1)
                fft = np.fft.fftshift(fft)
                channels_fft[ch] = fft
                power_spectra[ch] = self.calc_power_spectra(fft)

        if self.eval_spectra:
            for ch in self.channels:
                self.spectra[ch][central_frequency].append(power_spectra[ch])

        if self.eval_coherence:
            coherence = self.calc_spectral_coherence(channels_fft, power_spectra)
            self.coherence[central_frequency].append(coherence)

    def process(self, params: SeriesParameters, time_marks: np.ndarray, quadratures: np.ndarray):
        if not self.validate(params):
            raise ValueError('Given parameters are invalid')

        # Initialization
        if not self.initialized:
            self.init(params)

        if params.channel not in self.channels_set:
            self.channels_set.append(params.channel)

        if params.frequency not in self.central_frequencies:
            self.central_frequencies.add(params.frequency)

        if len(quadratures.shape) != 2:
            raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        if len(time_marks) != len(quadratures):
            raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        # Await channels_set
        if params.frequency in self.await_input:
            prev_params, prev_time_marks, prev_quadratures = self.await_input[params.frequency]
            prev_channel = prev_params.channel
            # If adjacent channel did not appear
            if params.channel == prev_channel:
                self._process_quads(prev_params.frequency, prev_time_marks,
                                    {prev_channel: prev_quadratures})
                self.await_input[params.frequency] = params, time_marks, quadratures
            # If adjacent channel appeared, process and reset await_input
            elif params.channel == ADJACENT_CHANNELS[prev_channel]:
                if not (time_marks == prev_time_marks).all():
                    raise RuntimeError('Invalid time marks of adjacent channel {}'
                                       ''.format(prev_channel))
                joint_quads = {params.channel: quadratures, prev_channel: prev_quadratures}
                self._process_quads(params.frequency, time_marks, joint_quads)
                del self.await_input[params.frequency]
            else:
                raise RuntimeError('Invalid input parameters (expect same or adjacent channels_set)')
        else:
            self.await_input[params.frequency] = params, time_marks, quadratures

    def finish(self):
        self.initialized = False


class PassiveScanHandler(PassiveHandler):
    valid_source_types = ['scan']

    def finish(self):
        super().finish()
        # Assuming that all frequencies came consequently, calculate mean time
        time_marks = []
        for all_freqs_time_marks in zip(self.time_marks.values()):
            time_marks.append(central_time(all_freqs_time_marks))
        time_marks = np.array(time_marks, dtype=dtime.datetime)

        # Create complete arrays from all central frequencies
        # Discard frequencies and data located at regions where frequency bands overlap
        central_frequencies = sorted(self.central_frequencies)
        n_central = len(central_frequencies)

        frequencies = []  # Array of all bands, joined together
        if self.eval_spectra:
            spectra = {channel: [] for channel in self.channels}
        else:
            spectra = None

        if self.eval_coherence:
            coherence = []
        else:
            coherence = None

        ref_band = self.ref_band
        for central_freq_num in range(n_central):
            curr_central_freq = central_frequencies[central_freq_num]
            freq_band = ref_band + curr_central_freq
            band_mask = np.ones(self.n_fft, dtype=bool)

            # Overlap with previous band
            if central_freq_num != 0:
                prev_central_freq = central_frequencies[central_freq_num - 1]
                prev_mid_freq = (curr_central_freq - prev_central_freq) / 2
                band_mask &= (freq_band > prev_mid_freq)

            # Overlap with following band
            if central_freq_num != (n_central - 1):
                next_central_freq = central_frequencies[central_freq_num + 1]
                next_mid_freq = (next_central_freq - curr_central_freq) / 2
                band_mask &= (freq_band < next_mid_freq)

            frequencies.append(freq_band[band_mask])

            # Evaluate quantities, taking only values in band_mask region
            if self.eval_spectra:
                for ch in self.channels:
                    daily_spectrum = []
                    for time_mark_num, spectrum in enumerate(self.spectra[ch][curr_central_freq]):
                        daily_spectrum.append(spectrum[band_mask])

                    spectra[ch].append(np.concatenate(daily_spectrum))

            if self.eval_coherence:
                daily_coherence = []
                for time_mark_num, coh in enumerate(self.coherence[curr_central_freq]):
                    daily_coherence.append(coh[band_mask])

                coherence.append(np.concatenate(daily_coherence))

        # Convert to numpy arrays
        central_frequencies = Frequency(np.array(central_frequencies), 'Hz')
        frequencies = Frequency(np.array(frequencies), 'Hz')

        if spectra is not None:
            spectra = {ch: np.array(spectra[ch]).T for ch in spectra}
            del spectra

        if coherence is not None:
            coherence = np.array(coherence).T
            del coherence

        parameters = PassiveParameters(self.sampling_frequency, self.n_avg, self.n_fft,
                                       self.channels)
        return PassiveScanResult(self.date, parameters, time_marks,
                                 frequencies, central_frequencies,
                                 spectra, coherence)


class PassiveTrackHandler(PassiveHandler):
    valid_source_types = ['cyg', 'cass', 'crab', 'sun']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_name = None

    def init(self, params: SeriesParameters):
        super().init(params)
        self.source_name = get_source_type(params.file_info)

    def process(self, params: SeriesParameters, time_marks: np.ndarray, quadratures: np.ndarray):
        if not self.validate(params):
            raise ValueError('Given parameters are invalid')

        # Initialization
        if not self.initialized:
            self.init(params)

        if params.channel not in self.channels_set:
            self.channels_set.append(params.channel)

        if params.frequency not in self.central_frequencies:
            self.central_frequencies.add(params.frequency)

        if len(quadratures.shape) != 2:
            raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        if len(time_marks) != len(quadratures):
            raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        # Await channels_set
        if self.await_input is not None:
            prev_params, prev_time_marks, prev_quadratures = self.await_input
            prev_channel = prev_params.channel
            prev_frequency = prev_params.frequency

            # If adjacent channel did not appear
            if params.channel == prev_channel:


        if params.frequency in self.await_input:
            prev_params, prev_time_marks, prev_quadratures = self.await_input[params.frequency]
            prev_channel = prev_params.channel
            # If adjacent channel did not appear
            if params.channel == prev_channel:
                self._process_quads(prev_params.frequency, prev_time_marks,
                                    {prev_channel: prev_quadratures})
                self.await_input[params.frequency] = params, time_marks, quadratures
            # If adjacent channel appeared, process and reset await_input
            elif params.channel == ADJACENT_CHANNELS[prev_channel]:
                if not (time_marks == prev_time_marks).all():
                    raise RuntimeError('Invalid time marks of adjacent channel {}'
                                       ''.format(prev_channel))
                joint_quads = {params.channel: quadratures, prev_channel: prev_quadratures}
                self._process_quads(params.frequency, time_marks, joint_quads)
                del self.await_input[params.frequency]
            else:
                raise RuntimeError('Invalid input parameters (expect same or adjacent channels_set)')
        else:
            self.await_input[params.frequency] = params, time_marks, quadratures

    def finish(self):
        super().finish()
        time_marks = np.array(self.time_marks, dtype=dtime.datetime)
        central_frequencies = Frequency(np.array(self.central_frequencies))

        params = PassiveParameters(self.sampling_frequency, self.n_avg, self.n_fft, self.channels)

        # Gether

        # Convert to numpy arrays
        spectra = {ch: np.array(self.spectra[ch]) for ch in self.channels}
        coherence = np.array(self.coherence)

        return PassiveTrackResult(self.date,
                                  source_name=self.source_name,
                                  parameters=params,
                                  time_marks=time_marks,
                                  central_frequencies=central_frequencies,
                                  spectra=spectra,
                                  coherence=coherence)

class PassiveSupervisor(Supervisor):
    """Supervisor that manages passive processing"""
    def __init__(self, n_accumulation, n_fft, timeout, eval_spectra=True, eval_coherence=True):
        super().__init__(timeout=timeout)
        self.n_accumulation = n_accumulation
        self.n_fft = n_fft
        self.eval_spectra = eval_spectra
        self.eval_coherence = eval_coherence

    def init_handler(self, time_mark: Sequence[dtime.datetime], parameters: SeriesParameters):
        source_name = get_source_type(parameters.file_info)
        if source_name == 'scan':
            return PassiveScanHandler()
