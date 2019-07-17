import datetime as dt
from collections import defaultdict, namedtuple
from enum import Enum

import numpy as np
from typing import List, TextIO, Dict, IO, Iterator, Tuple, Generator, Any, BinaryIO

from iisr.iisr_io import FileInfo, TimeSeriesPackage, TimeSeries
from iisr.preprocessing.representation import HandlerResult, Handler, HandlerParameters, \
    Supervisor, timeout_filter, HandlerBatch
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit
from iisr.utils import central_time


class PassiveMode(Enum):
    scan = 1
    cyg = 2
    cass = 3
    sun = 4
    crab = 5


SOURCES_FILE_CODES = {
    1: PassiveMode.cyg,
    2: PassiveMode.crab,
    4: PassiveMode.cass,
    8: PassiveMode.sun,
}


def get_passive_mode(file_info: FileInfo) -> PassiveMode:
    """Return type of passive data given information in file name.

    Args:
        file_info: Information from file name.

    Returns:
        passive_type: Name of passive data type.
    """
    if file_info.field4 == 0:
        return PassiveMode.scan
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


class PassiveParameters(HandlerParameters):
    sampling_frequency = NotImplemented
    n_accumulation = NotImplemented
    n_fft = NotImplemented
    channels = NotImplemented
    band_type = NotImplemented

    @classmethod
    def load_txt(cls, file: TextIO):
        params = cls.read_params_from_txt(file)
        return cls(**params)


class PassiveScanParameters(PassiveParameters):
    params_to_save = ['sampling_frequency', 'n_accumulation', 'n_fft',
                      'central_frequencies', 'channels', 'band_type']

    def __init__(self, sampling_frequency: Frequency, n_accumulation: int,
                 n_fft: int, central_frequencies: Tuple[Frequency], channels: List[Channel],
                 band_type: str):
        self.sampling_frequency = sampling_frequency
        self.n_accumulation = n_accumulation
        self.n_fft = n_fft
        self.central_frequencies = tuple(central_frequencies)
        self.channels = tuple(sorted(channels))
        self.band_type = band_type


class PassiveTrackParameters(PassiveParameters):
    params_to_save = ['sampling_frequency', 'n_accumulation', 'n_fft', 'channels',
                      'band_type']

    def __init__(self, sampling_frequency: Frequency, n_accumulation: int,
                 n_fft: int, channels: List[Channel], band_type: str):
        self.sampling_frequency = sampling_frequency
        self.n_accumulation = n_accumulation
        self.n_fft = n_fft
        self.channels = tuple(sorted(channels))
        self.band_type = band_type


class PassiveResult(HandlerResult):
    def __init__(self, parameters: PassiveParameters):
        self.parameters = parameters

    @property
    def short_name(self):
        return '{}'.format(self.parameters.band_type)

    def save_txt(self, file: IO, save_date: dt.date = None):
        raise NotImplementedError

    @classmethod
    def load_txt(cls, file: List[IO]):
        raise NotImplementedError

    @property
    def n_fft(self):
        return self.parameters.n_fft

    @property
    def sampling_frequency(self):
        return self.parameters.sampling_frequency


class PassiveScanResult(PassiveResult):
    mode_name = 'passive_scan'

    def __init__(self, parameters: PassiveScanParameters, time_marks: np.ndarray,
                 frequencies: Frequency, spectra: Dict[Channel, np.ndarray],
                 coherence: np.ndarray = None):
        if spectra is not None:
            if sorted(spectra.keys()) != parameters.channels:
                raise AssertionError('Channels in spectra dictionary must be identical '
                                     'to channels_set in parameters object')

            for spectrum in spectra.values():
                if spectrum.shape != (len(time_marks), frequencies.size):
                    raise ValueError('Expected shape for spectrum: [n_times x n_central]')

        if coherence is not None:
            if coherence.shape != (len(time_marks), frequencies.size):
                raise ValueError('Expected shape for coherence: [n_times x n_central]')

            if not np.iscomplexobj(coherence):
                raise ValueError('Expect coherence to be complex valued')

        super().__init__(parameters)
        self.time_marks = np.array(time_marks, dtype=dt.datetime)
        self.frequencies = frequencies
        self.spectra = spectra
        self.coherence = coherence

    def save_pickle(self, file: IO, save_date: dt.date = None):
        raise NotImplementedError

    @classmethod
    def load_pickle(cls, file: List[BinaryIO]) -> 'HandlerResult':
        raise NotImplementedError


class PassiveTrackResult(PassiveResult):
    mode_name = 'passive_track'

    def __init__(self, parameters: PassiveTrackParameters,
                 time_marks: np.ndarray, central_frequencies: Frequency,
                 spectra: Dict[Channel, np.ndarray], coherence: np.ndarray = None):
        if central_frequencies.size != time_marks.size:
            raise ValueError('Expect central frequencies to be 1-d array with value for each '
                             'time mark')

        if spectra is not None:
            if sorted(spectra.keys()) != parameters.channels:
                raise AssertionError('Channels in spectra dictionary must be identical '
                                     'to channels_set in parameters object')

            for spectrum in spectra.values():
                if spectrum.shape != (len(time_marks), parameters.n_fft):
                    raise ValueError('Expected shape for spectrum: [n_times x n_central]')

        if coherence is not None:
            if coherence.shape != (len(time_marks), parameters.n_fft):
                raise ValueError('Expected shape for coherence: [n_times x n_central]')

            if not np.iscomplexobj(coherence):
                raise ValueError('Expect coherence to be complex valued')

        super().__init__(parameters)
        self.time_marks = np.array(time_marks, dtype=dt.datetime)
        self._frequencies = None
        self.central_frequencies = central_frequencies
        self.spectra = spectra
        self.coherence = coherence

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

    def save_pickle(self, file: BinaryIO, save_date: dt.date = None):
        pass

    @classmethod
    def load_pickle(cls, file: List[BinaryIO]) -> 'HandlerResult':
        pass


class PassiveBatch(HandlerBatch):
    def __init__(self, time_marks: Any, batch_params: Any, quadratures: Any):
        self.time_marks = time_marks
        self.batch_params = batch_params
        self.quadratures = quadratures


class PassiveScanBatch(PassiveBatch):
    def __init__(self,
                 time_marks: List[List[dt.datetime]],
                 batch_params: Dict,
                 quadratures: Dict[Channel, List[np.ndarray]]):
        super().__init__(time_marks, batch_params, quadratures)


class PassiveTrackBatch(PassiveBatch):
    def __init__(self,
                 time_marks: List[dt.datetime],
                 batch_params: Dict,
                 quadratures: Dict[Channel, np.ndarray]):
        super().__init__(time_marks, batch_params, quadratures)


class PassiveHandler(Handler):
    def __init__(self, parameters: PassiveParameters, eval_coherence=True):
        self.parameters = parameters

        self.eval_coherence = eval_coherence

        self.time_marks = []
        self.spectra = defaultdict(list)
        self.coherence = [] if eval_coherence else None

        # Masks for resulting spectra (to create non-overlapping bands)
        self.band_masks = NotImplemented

    @property
    def channels(self):
        """Sorted list of channels"""
        return self.parameters.channels

    @property
    def n_fft(self):
        return self.parameters.n_fft

    @property
    def n_avg(self):
        return self.parameters.n_accumulation

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

    def finish(self):
        return NotImplemented


class PassiveScanHandler(PassiveHandler):
    def __init__(self, parameters: PassiveScanParameters, eval_coherence=True):
        super().__init__(parameters, eval_coherence=eval_coherence)
        self.central_frequencies = parameters.central_frequencies
        self.n_central = len(self.central_frequencies)
        self.frequencies, self.band_masks = self._get_non_overlapping_masks()

    def _get_non_overlapping_masks(self):
        frequencies = []
        band_masks = []

        ref_band = calc_ref_band(self.n_fft, self.parameters.sampling_frequency)['Hz']

        for central_freq_num in range(self.n_central):
            curr_central_freq = self.central_frequencies[central_freq_num]['Hz']
            freq_band = ref_band + curr_central_freq
            band_mask = np.ones(self.n_fft, dtype=bool)

            # Overlap with previous band
            if central_freq_num != 0:
                prev_central_freq = self.central_frequencies[central_freq_num - 1]['Hz']
                prev_mid_freq = (curr_central_freq - prev_central_freq) / 2
                band_mask &= (freq_band > prev_mid_freq)

            # Overlap with following band
            if central_freq_num != (self.n_central - 1):
                next_central_freq = self.central_frequencies[central_freq_num + 1]['Hz']
                next_mid_freq = (next_central_freq - curr_central_freq) / 2
                band_mask &= (freq_band < next_mid_freq)

            frequencies.append(freq_band[band_mask])
            band_masks.append(band_mask)
        return frequencies, band_masks

    def _assert_correct_batch(self, batch: PassiveScanBatch):
        time_marks = batch.time_marks
        quadratures = batch.quadratures

        input_channels = tuple(sorted(quadratures.keys()))
        if input_channels != self.parameters.channels:
            raise ValueError(f'Invalid channels from scan batch: {input_channels}')

        for channel_quads in quadratures.values():
            if len(channel_quads) != self.n_central:
                raise ValueError('Expect separate quadratures array for each sampling frequency')

            if len(time_marks) != self.n_central:
                raise ValueError('Expect separate time_marks array for each sampling frequency')

            for q_arr, dt_arr in zip(channel_quads, time_marks):
                if q_arr.shape != (len(dt_arr), self.n_fft):
                    raise ValueError('Incorrect shape of quadratures arrays')

    def handle(self, batch: PassiveScanBatch):
        self._assert_correct_batch(batch)

        # Whole input arrays (each channel and each frequency)
        # will be represented by single time mark
        self.time_marks.append(central_time([central_time(marks) for marks in batch.time_marks]))

        # Calculate fft of quadratures for each channel for each central frequency
        channels_fft = defaultdict(list)
        channels_power_spectra = defaultdict(list)
        for ch, quads_per_freq in batch.quadratures.items():
            for quads in quads_per_freq:
                fft = self.calc_fft(quads, axis=1)
                fft = np.fft.fftshift(fft)
                channels_fft[ch].append(fft)
                channels_power_spectra[ch].append(self.calc_power_spectra(fft))

        # Concatenate estimated quantities from difference frequency bands
        # Spectra
        for ch in self.channels:
            full_band_spectra = []
            for pwr_sp, mask in zip(channels_power_spectra[ch], self.band_masks):
                full_band_spectra.append(pwr_sp[mask])
            self.spectra[ch].append(np.concatenate(full_band_spectra))

        # Coherence
        if self.eval_coherence:
            full_band_coherence = []
            for cfreq_num, mask in enumerate(self.band_masks):
                # calc_spectral_coherence require dictionary of [channel: quadratures array]
                cfreq_fft = {ch: fft[cfreq_num] for ch, fft in channels_fft.items()}
                cfreq_power_spectra = {ch: pwr_sp[cfreq_num] for ch, pwr_sp
                                       in channels_power_spectra.items()}
                coherence = self.calc_spectral_coherence(cfreq_fft, cfreq_power_spectra)
                full_band_coherence.append(coherence[mask])
            self.coherence.append(np.concatenate(full_band_coherence))

    def finish(self):
        time_marks = np.array(self.time_marks, dtype=dt.datetime)

        # Convert to numpy arrays
        frequencies = Frequency(self.frequencies, 'Hz')

        spectra = {ch: np.array(self.spectra[ch]).T for ch in self.spectra}
        del spectra

        if self.eval_coherence:
            coherence = np.array(self.coherence).T
        else:
            coherence = None

        parameters = self.parameters  # type: PassiveScanParameters
        return PassiveScanResult(parameters, time_marks, frequencies, spectra, coherence)


class PassiveTrackHandler(PassiveHandler):
    def __init__(self, parameters: PassiveTrackParameters, eval_coherence=True):
        super().__init__(parameters, eval_coherence=eval_coherence)
        self.central_frequencies_hz = []

    def _assert_correct_batch(self, batch: PassiveScanBatch):
        time_marks = batch.time_marks
        pars = batch.batch_params
        quadratures = batch.quadratures

        assert 'central_frequency' in pars

        input_channels = tuple(sorted(quadratures.keys()))
        if input_channels != self.parameters.channels:
            raise ValueError(f'Invalid channels from scan batch: {input_channels}')

        for channel_quads in quadratures.values():
            for q_arr in channel_quads:
                if q_arr.shape != (len(time_marks), self.n_fft):
                    raise ValueError('Incorrect shape of quadratures arrays')

    def handle(self, batch: PassiveScanBatch):
        self._assert_correct_batch(batch)

        self.time_marks.append(central_time(batch.time_marks))

        central_frequency = batch.batch_params['central_frequency']
        self.central_frequencies_hz.append(central_frequency['Hz'])

        # Calculate fft and power spectra
        channels_fft = defaultdict(list)  # cache for coherence
        for ch, quads in batch.quadratures.items():
            fft = self.calc_fft(quads, axis=1)
            fft = np.fft.fftshift(fft)
            channels_fft[ch].append(fft)
            self.spectra[ch].append(self.calc_power_spectra(fft))

        # Coherence
        if self.eval_coherence:
            cfreq_fft = {ch: fft for ch, fft in channels_fft.items()}
            cfreq_power_spectra = {ch: pwr_sp for ch, pwr_sp in self.spectra.items()}
            coherence = self.calc_spectral_coherence(cfreq_fft, cfreq_power_spectra)
            self.coherence.append(coherence)

    def finish(self):
        time_marks = np.array(self.time_marks, dtype=dt.datetime)
        central_frequencies = Frequency(np.array(self.central_frequencies_hz), 'Hz')

        # Convert to numpy arrays
        spectra = {ch: np.array(self.spectra[ch]) for ch in self.channels}

        if self.eval_coherence:
            coherence = np.array(self.coherence)
        else:
            coherence = None

        parameters = self.parameters  # type: PassiveTrackParameters
        return PassiveTrackResult(parameters, time_marks, central_frequencies, spectra, coherence)


class PassiveSupervisor(Supervisor):
    """Supervisor that manages passive processing"""
    UniqueParams = namedtuple('UniqueParams', ['mode', 'band_type'])
    AggYieldType = Tuple[UniqueParams, PassiveBatch]

    def __init__(self, n_accumulation: int, n_fft: int, timeout: dt.timedelta,
                 eval_spectra: bool = True, eval_coherence: bool = True):
        self.n_accumulation = n_accumulation
        self.n_fft = n_fft
        self.timeout = timeout

        self.n_batch_samples = n_fft * n_accumulation

        self.eval_spectra = eval_spectra
        self.eval_coherence = eval_coherence

    def groupby_band(self, packages: Iterator[TimeSeriesPackage]):
        timeout_coroutine = timeout_filter(self.timeout)
        next(timeout_coroutine)  # Init coroutine

        prev_mode = None
        prev_date = None

        acc_series = defaultdict(lambda: defaultdict(list))

        for package in packages:
            sample_series = package.time_series_list[0]
            frequency = sample_series.parameters.frequency
            mode = get_passive_mode(sample_series.parameters.file_info)
            n_samples = sample_series.parameters.n_samples

            # Check, if necessary number of samples can be formed by stacking quadratures in series
            if self.n_batch_samples % n_samples != 0:
                raise ValueError('n_accumulated_samples = (n_accumulation x n_fft)'
                                 ' should be divisible by the number of samples in series '
                                 '(n_accumulated_samples = {}, n_samples = {})'
                                 ''.format(self.n_batch_samples, n_samples))
            else:
                max_acc_series = self.n_batch_samples // n_samples

            band_types = defaultdict(list)
            for series in package:
                assert series.parameters.frequency == frequency
                assert get_passive_mode(series.parameters.file_info) == mode

                # Check timeout
                if timeout_coroutine.send(package):
                    acc_series = {}
                    continue

                # Check if mode has changed
                if prev_mode is not None and prev_mode != mode:
                    acc_series = {}
                    continue
                prev_mode = mode

                # Check if date has changed
                date = package.time_mark.date()
                if prev_date is not None and prev_date != date:
                    acc_series = {}
                    continue
                prev_date = date

                band_types[series.parameters.channel.band_type].append(series)

            for band_type, series_list in band_types.items():
                series_dict = acc_series[(band_type, frequency)]

                current_length = None
                for series in series_list:
                    channel = series.parameters.channel
                    series_dict[channel].append(series)
                    current_length = len(series_dict[channel])

                if current_length is None:
                    raise AssertionError('Series list is empty')

                if current_length == max_acc_series:
                    for ch in series_dict:
                        assert current_length == len(series_dict[ch])

                    del acc_series[(band_type, frequency)]
                    yield (band_type, frequency), series_dict

    def _get_passive_params(self, mode, date, sampling_frequency, channels, central_frequencies,
                            band_type):
        if mode is PassiveMode.scan:
            batch_params = {}
            passive_params = PassiveScanParameters(sampling_frequency=sampling_frequency,
                                                   n_accumulation=self.n_accumulation,
                                                   n_fft=self.n_fft,
                                                   central_frequencies=central_frequencies,
                                                   channels=channels,
                                                   band_type=band_type)
        elif mode in PassiveMode:
            assert len(central_frequencies) == 1
            batch_params = {'central_frequency': central_frequencies[0]}
            passive_params = PassiveTrackParameters(sampling_frequency=sampling_frequency,
                                                    n_accumulation=self.n_accumulation,
                                                    n_fft=self.n_fft,
                                                    channels=channels,
                                                    band_type=band_type)
        else:
            raise AssertionError('Incorrect mode')

        return batch_params, passive_params

    def _get_fft_split_count(self, series_length):
        # Check, if necessary number of samples can be formed by stacking quadratures in series
        if self.n_batch_samples % series_length == 0:
            return self.n_batch_samples // series_length
        else:
            raise ValueError(
                f'n_batch_samples = (n_accumulation x n_fft) should be divisible '
                f'by the number of samples in series '
                f'(n_batch_samples = {self.n_batch_samples}, series_length = {series_length})'
            )

    def aggregator(self, packages: Iterator[TimeSeriesPackage]
                   ) -> Generator[AggYieldType, Any, Any]:
        """Aggregate series from packages into arrays, grouping them by parameters.
        Since parameters of incoming series are unknown, grouping logic should be formulated on fly.
        Quadratures in passive mode grouped by frequencies and channel bands.

        Args:
            packages:

        Returns:

        """
        UniqueParams = self.UniqueParams

        def new_buffer() -> Dict[UniqueParams, Dict[Channel, Tuple]]:
            return defaultdict(lambda: defaultdict(lambda: ([], [])))

        timeout_coroutine = timeout_filter(self.timeout)
        next(timeout_coroutine)  # Init coroutine

        buffer = new_buffer()
        track_curr_frequency = None
        curr_mode = None
        for package in packages:
            # Get passive mode from first series of the package
            sample_series = package.time_series_list[0]
            mode = get_passive_mode(sample_series.parameters.file_info)
            frequency = sample_series.parameters.frequency
            n_samples = sample_series.parameters.n_samples

            # Reset if timeout or mode has changed
            if timeout_coroutine.send(package) or curr_mode != mode:
                buffer = new_buffer()
                curr_mode = mode

            for series in package.time_series_list:
                channel = series.parameters.channel
                band_type = series.parameters.band_type

                unique_params = UniqueParams(mode, band_type)

                if mode == PassiveMode.scan:
                    pass
                else:
                    # Track mode
                    if frequency != track_curr_frequency:
                        # Reset buffer
                        buffer = new_buffer()
                        track_curr_frequency = frequency

                    band_buf = buffer[unique_params]

                    acc_marks, acc_quads = band_buf[channel]

                    # Append new record
                    acc_marks.append(series.time_mark)
                    acc_quads.append(series.quadratures)

                    if len(acc_marks) != self.n_batch_samples:
                        pass

    def aggregator_(self, packages: Iterator[TimeSeriesPackage], drop_timeout_series: bool = True
                    ) -> Generator[AggYieldType, Any, Any]:
        """Aggregate series from packages into arrays, grouping them by parameters.

        This function contain complicated logic to iterate over series.
        Since parameters of incoming series are unknown, grouping logic should be formulated on fly.

        Quadratures in passive mode grouped by observation frequencies and channel bands.

        Args:
            packages:
            drop_timeout_series:

        Returns:

        """
        start_params = None
        band_dict = defaultdict(list)
        central_frequencies_dict = defaultdict(list)
        for (band_type, frequency), series_dict in self.groupby_band(packages):
            # If full frequency cycle was passed, yield
            if start_params == (band_type, frequency):
                central_frequencies = central_frequencies_dict[band_type]

                # n_central_frequencies accumulated series:
                acc_series = band_dict[band_type]  # type: List[Dict[Channel, List[TimeSeries]]]

                # Channels are identical for each dictionary in the list
                channels = list(acc_series[0].keys())  # type: List[Channel]

                # Sampling frequency, date and mode are also identical
                sample_series = acc_series[0][channels[0]][0]  # type: TimeSeries
                mode = get_passive_mode(sample_series.parameters.file_info)
                sampling_frequency = sample_series.parameters.sampling_frequency
                date = sample_series.time_mark.date()

                # Now, for each central frequency, append accumulated time marks to a list
                # Quadratures should be appended to each channel
                time_marks = []
                quadratures = {ch: [] for ch in channels}
                for freq_num in range(len(central_frequencies)):
                    single_freq_acc_series_dict = acc_series[freq_num]

                    # Time marks are identical for each channel. It is guarantied by previous logic.
                    # Oversample time marks (we cut series n_samples by n_fft)
                    n_samples = single_freq_acc_series_dict[channels[0]][0].parameters.n_samples
                    n_oversample = n_samples // self.n_fft
                    acc_time_marks = []
                    for series in single_freq_acc_series_dict[channels[0]]:
                        acc_time_marks.extend([series.time_mark] * n_oversample)
                    time_marks.append(np.array(acc_time_marks, dtype=dt.datetime))

                    for ch in channels:
                        # n_acc x n_samples
                        acc_quadratures = [series.quadratures for series
                                           in single_freq_acc_series_dict[ch]]
                        new_shape = self.n_accumulation, -1
                        quadratures[ch].append(np.stack(acc_quadratures).reshape(new_shape))

                # Get parameters
                batch_params, passive_params = self._get_passive_params(
                    mode, date, sampling_frequency, channels, central_frequencies, band_type
                )
                yield passive_params, PassiveBatch(time_marks, batch_params, quadratures)

            if start_params is None:
                start_params = band_type, frequency

            band_dict[band_type].append(series_dict)
            central_frequencies_dict[band_type].append(frequency)

    def init_handler(self, parameters: PassiveParameters):
        if isinstance(parameters, PassiveScanParameters):
            return PassiveScanHandler(parameters, eval_coherence=self.eval_coherence)
        elif isinstance(parameters, PassiveTrackParameters):
            return PassiveTrackHandler(parameters, eval_coherence=self.eval_coherence)
        else:
            raise AssertionError('Unexpected type of input parameters: {}'.format(type(parameters)))
