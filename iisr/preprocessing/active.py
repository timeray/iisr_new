import itertools as it
from collections import defaultdict
from datetime import datetime, date, timedelta

import numpy as np
from scipy import signal
from typing import List, TextIO, Dict, Sequence, Tuple, Generator, Iterator, Any

from iisr.io import SeriesParameters, ExperimentParameters, TimeSeriesPackage
from iisr.preprocessing.representation import HandlerResult, Handler, ResultParameters, Supervisor
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit, Distance
from iisr.utils import TIME_FMT, DATE_FMT, central_time

__all__ = ['calc_delays', 'calc_distance', 'ActiveParameters', 'ActiveResult', 'PowerParams',
           'CoherenceParams', 'LongPulseActiveHandler', 'ShortPulseActiveHandler',
           'EvalCoherenceError', 'ReadingError', 'square_barker']


class ReadingError(Exception):
    pass


class EvalCoherenceError(Exception):
    pass


def square_barker(n_points: int, barker_len: int) -> np.ndarray:
    """Return square Barker code with specified number of elements.

    Waveform is fit into number of given points, meaning that it starts at 0 and ends at n_points-1.

    Args:
        n_points: Number of discrete points.
        barker_len: Number of elements in sequence (2, 3, 4, 5, 11, 13).

    Returns:
        waveform: Discrete Barker sequence (1 or -1).
    """
    if barker_len == 2:
        arr = np.array([1, -1])
    elif barker_len == 3:
        arr = np.array([1, 1, -1])
    elif barker_len == 4:
        arr = np.array([1, -1, 1, 1])  # or 1, -1, -1, -1
    elif barker_len == 5:
        arr = np.array([1, 1, 1, -1, 1])
    elif barker_len == 11:
        arr = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
    elif barker_len == 13:
        arr = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    else:
        raise ValueError('Wrong Barker element number')

    discrete = barker_len / (n_points - 1)
    res = []
    for i in range(n_points - 1):
        res.append(arr[int(np.floor(i * discrete))])
    res.append(arr[-1])
    return np.array(res, dtype=float)


def calc_delays(sampling_frequency: Frequency, total_delay: TimeUnit, n_samples: int) -> TimeUnit:
    """Calculate delays between transmission and reception time.

    Args:
        sampling_frequency: Sampling frequency.
        total_delay: Delay between transmission and start of reception.
        n_samples: Number of registered samples.

    Returns:
        delays: Delays.
    """
    # Time between samples, us
    period = TimeUnit(1 / sampling_frequency['MHz'], 'us')

    stop_time = n_samples * period['us'] + total_delay['us']
    delays = TimeUnit(np.arange(total_delay['us'], stop_time, period['us']), 'us')
    return delays


def calc_distance(delays: TimeUnit) -> Distance:
    """Calculate distance using delays.

    Args:
        delays: Samples delays between transmission and reception.

    Returns:
        distances: Distance between radar and target.
    """
    return Distance(delays['us'] * 0.15, 'km')


class ActiveParameters(ResultParameters):
    params_to_save = ['sampling_frequency', 'total_delay', 'n_samples', 'channels_set',
                      'pulse_type', 'frequency', 'pulse_length', 'phase_code']

    def __init__(self, global_parameters: ExperimentParameters, channels: List[Channel],
                 pulse_type: str, frequency: Frequency, pulse_length: TimeUnit, phase_code: int):
        self.global_parameters = global_parameters

        if not channels:
            raise ValueError('Empty channels_set')

        for i, ch in enumerate(channels):
            if isinstance(ch, int):
                channels[i] = Channel(ch)

        self.channels = sorted(channels)
        self.pulse_type = pulse_type
        self.frequency = frequency
        self.pulse_length = pulse_length
        self.phase_code = phase_code

        self._delays = None

    @property
    def sampling_frequency(self):
        return self.global_parameters.sampling_frequency

    @property
    def n_samples(self):
        return self.global_parameters.n_samples

    @property
    def total_delay(self):
        return self.global_parameters.total_delay

    @property
    def delays(self) -> TimeUnit:
        if self._delays is None:
            self._delays = calc_delays(self.sampling_frequency, self.total_delay, self.n_samples)
        return self._delays

    @property
    def distance(self) -> Distance:
        return calc_distance(self.delays)

    @classmethod
    def load_txt(cls, file: TextIO):
        params = cls.read_params_from_txt(file)
        global_params = ExperimentParameters(params.pop('sampling_frequency'),
                                             params.pop('n_samples'),
                                             params.pop('total_delay'))
        return cls(global_params, **params)


class QuantityParameters:
    parameter_names = NotImplemented

    def __eq__(self, other: 'PowerParams'):
        for param in self.parameter_names:
            if getattr(self, param) != getattr(other, param):
                return False
        return True

    def __hash__(self):
        return hash(tuple(getattr(self, param) for param in self.parameter_names))

    def __str__(self):
        params = (str(getattr(self, param)) for param in self.parameter_names)
        return '{}({})'.format(self.__class__.__name__, ', '.join(params))

    def __iter__(self):
        return iter((getattr(self, param) for param in self.parameter_names))


class PowerParams(QuantityParameters):
    def __init__(self, channel: int, frequency: Frequency, pulse_length: TimeUnit):
        self.channel = channel
        self.frequency = frequency
        self.pulse_length = pulse_length

        self.parameter_names = ['channel', 'frequency', 'pulse_length']

    @classmethod
    def from_series_params(cls, parameters: SeriesParameters):
        return cls(channel=parameters.channel, frequency=parameters.frequency,
                   pulse_length=parameters.pulse_length)


class CoherenceParams(QuantityParameters):
    def __init__(self, frequency: Frequency, pulse_length: TimeUnit):
        self.frequency = frequency
        self.pulse_length = pulse_length

        self.parameter_names = ['frequency', 'pulse_length']

    @classmethod
    def from_series_params(cls, parameters: SeriesParameters):
        return cls(parameters.frequency,  parameters.pulse_length)


def _parse_header(header: str, keys: List[str], types: List[type]) -> List:
    header = header.split(',')[0].split('_')  # split and remove units (e.g. *_len700,abs.units)
    assert len(header) == (len(keys) + 1)
    res = []
    for i, (key, value_type) in enumerate(zip(keys, types)):
        value_str = header[i+1]
        if not value_str.startswith(key):
            raise ValueError('Key: {} not present or has incorrect position in'
                             ' input header: {}'.format(key, header))
        res.append(value_type(value_str[len(key):]))
    return res


class ActiveResult(HandlerResult):
    quantity_headers = {'power': 'Power', 'coherence': 'Coherence'}
    mode_name = 'active'

    def __init__(self, parameters: ActiveParameters,
                 time_marks: Sequence[datetime],
                 power: Dict[Channel, np.ndarray] = None,
                 coherence: np.ndarray = None):
        """Result of processing of active experiment. It is expected, that power and spectrum are
        calculated for each (frequency, pulse_len, channel) and time marks are aligned.

        Size of all estimated values must be equal to size of time_marks.

        Args:
            parameters: Parameters for processed experiment.
            time_marks: Sorted array of datetimes.
            power: Dictionary (key for each channel) of power profiles.
                Each profiles should have shape [len(time_marks), n_samples].
            coherence: Coherence between channels_set. Complex values.
                Shape [len(time_marks), n_samples].
        """
        if power is not None:
            if sorted(power.keys()) != parameters.channels:
                raise AssertionError('Channels in power dictionary must be identical '
                                     'to channels_set in parameters object')

            for pwr in power.values():
                if len(time_marks) != len(pwr):
                    raise ValueError('Length of time_marks and all power arrays must be equal')

        if coherence is not None:
            if len(time_marks) != len(coherence):
                raise ValueError('Length of time_marks and all coherence arrays must be equal')

            if not np.iscomplexobj(coherence):
                raise ValueError('Expect coherence to be complex valued')

        self.parameters = parameters
        self.time_marks = np.array(time_marks, dtype=datetime)
        self.power = power
        self.coherence = coherence

        # Calculate all involved dates
        self.dates = sorted(set((date(dt.year, dt.month, dt.day) for dt in self.time_marks)))

    @property
    def short_name(self) -> str:
        return 'ch{}_freq{:.2f}_len{}'.format(self.channels,
                                              self.frequency['MHz'],
                                              self.pulse_length['us'])

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif hasattr(self.parameters, item):
            return getattr(self.parameters, item)
        else:
            raise AttributeError(item)

    def _write_results(self, file: TextIO, save_date: date = None, sep=' ', dist_units='km',
                       precision=5):
        # Save data for given date
        # Write header
        cols = ['Date,UT', 'Time,UT', 'Distance,{}'.format(dist_units)]
        if self.power is not None:
            for ch in self.power:
                cols.append('{}_ch{},abs.units'.format(self.quantity_headers['power'], ch))
        if self.coherence is not None:
            cols.append(self.quantity_headers['coherence'])
        file.write(sep.join(cols))
        file.write('\n')

        # Write arrays
        if save_date is not None:
            start_date = datetime(save_date.year, save_date.month, save_date.day)
            end_date = start_date + timedelta(days=1)
            date_mask = (self.time_marks >= start_date) & (self.time_marks < end_date)
        else:
            date_mask = np.ones(len(self.time_marks), dtype=bool)

        # Convert all repeatable arrays to strings
        float_fmt = '{{:.{}f}}'.format(precision)

        distance_str = [float_fmt.format(dist[dist_units]) for dist in self.distance]
        time_str = [time_mark.strftime(TIME_FMT) for time_mark in self.time_marks[date_mask]]
        date_str = [time_mark.strftime(DATE_FMT) for time_mark in self.time_marks[date_mask]]

        # Form long column iterators: distance changes every row, time and date change after all
        # distances
        columns = [
            it.chain.from_iterable(it.repeat(d, len(distance_str)) for d in date_str),
            it.chain.from_iterable(it.repeat(t, len(distance_str)) for t in time_str),
            it.chain.from_iterable(it.repeat(distance_str, len(time_str))),
        ]
        if self.power is not None:
            for ch in self.power:
                columns.append((float_fmt.format(val)
                                for val in self.power[ch][date_mask].ravel()))

        if self.coherence is not None:
            columns.append((float_fmt.format(val) for val in self.coherence[date_mask].ravel()))

        for row in zip(*columns):
            file.write(sep.join(row) + '\n')

    def save_txt(self, file: TextIO, save_date: date = None, save_params: bool = True,
                 sep: str = ' ', precision=5):
        """Save results to specific directory. If date was passed, save only results corresponding
        to this date.

        Args:
            file: File.
            save_date: Date to save.
            save_params: If params should be saved as a header.
            sep: Separator between columns in the file.
            precision: Precision of saved float numbers.
        """
        if save_params:
            self.parameters.save_txt(file)

        if save_date is not None and save_date not in self.dates:
            raise ValueError('Not results for given date {}'.format(save_date))

        self._write_results(file, save_date=save_date, sep=sep, precision=precision)

    @classmethod
    def load_txt(cls, files: List[TextIO], sep: str = ' ') -> 'ActiveResult':
        """Load processing results of active experiment.

        Args:
            files: List of files. Files must be sorted by date.
            sep: Separator between columns in the files.

        Returns:
            results: Results.
        """
        # Parse parameters
        params = ActiveParameters.load_txt(files[0])
        if params is None:
            raise NotImplementedError('Reading from files without parameters')

        for file in files[1:]:
            if params != ActiveParameters.load_txt(file):
                raise ReadingError('Parameters in all input files should match')

        # Check that all headers are equal
        header = files[0].readline()

        for file in files[1:]:
            if header != file.readline():
                raise ReadingError('Result headers of files should match')

        # Parse header
        header = header.split()
        if len(header) < 4:
            # Only parameters are present
            return cls(params, [])

        is_power_present = False
        is_coherence_present = False

        quantities_types = []
        pwr_counter = 0
        for pos in range(3, len(header)):
            if header[pos].startswith(cls.quantity_headers['power']):
                is_power_present = True
                quantities_types.append(float)
                ch = _parse_header(header[pos], ['ch'], [int])[0]
                assert params.channels[pwr_counter] == Channel(ch)
                pwr_counter += 1

            elif header[pos].startswith(cls.quantity_headers['coherence']):
                is_coherence_present = True
                quantities_types.append(complex)

            else:
                raise ReadingError('Unknown quantity: {}'.format(header[pos]))

        # Read arrays
        time_marks_set = set()
        time_marks = []
        prev_time_mark = datetime.min
        arrays = [[] for _ in range(len(quantities_types))]
        dtime_fmt = DATE_FMT + ' ' + TIME_FMT

        for file_num, file in enumerate(files):
            for line in file:
                line = line.split(sep)

                # Parse datetimes
                dtime_tuple = (line[0], line[1])
                if dtime_tuple not in time_marks_set:
                    time_marks_set.add(dtime_tuple)
                    time_mark = datetime.strptime(dtime_tuple[0] + ' ' + dtime_tuple[1], dtime_fmt)

                    if prev_time_mark > time_mark:
                        raise ReadingError(
                            'Some time mark in file (num = {}) is less then previous '
                            '(Maybe files was not sorted by date)'.format(file_num + 1)
                        )

                    time_marks.append(time_mark)
                    prev_time_mark = time_mark

                # line[2] is distance, we skip it here (it can be calculated from parameters)
                for val_num, val_type in enumerate(quantities_types):
                    arrays[val_num].append(val_type(line[3 + val_num]))

        # Match arrays to results
        arr_num = 0
        expected_arr_shape = (len(time_marks), params.n_samples)
        if is_power_present:
            power = {}
            for ch in params.channels:
                power[ch] = np.array(arrays[arr_num], dtype=float).reshape(expected_arr_shape)
                arr_num += 1
        else:
            power = None

        if is_coherence_present:
            coherence = np.array(arrays[arr_num], dtype=complex).reshape(expected_arr_shape)
            arr_num += 1
        else:
            coherence = None

        return cls(params, sorted(time_marks), power=power, coherence=coherence)


class ActiveHandler(Handler):
    """Abstract class for active processing"""
    valid_pulse_type = NotImplemented

    def preproc_quads(self, quadratures: np.ndarray) -> np.ndarray:
        return NotImplemented

    def __init__(self, global_parameters, channels, pulse_length, frequency, phase_code,
                 n_fft=None, h_step=None, eval_power=True, eval_coherence=False):
        self.global_parameters = global_parameters

        self.channels = sorted(channels)
        self.pulse_length = pulse_length
        self.frequency = frequency
        self.phase_code = phase_code

        self.nfft = n_fft
        self.h_step = h_step

        self.eval_power = eval_power
        self.eval_coherence = eval_coherence

        self.time_marks = []
        self.power = defaultdict(list) if self.eval_power else None
        self.coherence = [] if self.eval_coherence else None

    def process(self, time_marks: np.ndarray, quadratures: Dict[Channel, np.ndarray]):
        for quads in quadratures.values():
            if len(quads.shape) != 2:
                raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        if len(time_marks) != len(quadratures):
            raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        # Preprocess quadratures
        quadratures = {ch: self.preproc_quads(quadratures[ch]) for ch in self.channels}

        # Evaluate quantities
        self.time_marks.append(central_time(time_marks))

        if self.eval_power:
            for ch in self.channels:
                self.power[ch].append(self.calc_power(quadratures[ch]))

        if self.eval_coherence:
            quadratures_channels = sorted(quadratures.keys())
            if self.channels != quadratures_channels:
                raise EvalCoherenceError('Cannot evaluate coherence: incorrect channels '
                                         '(expected {} got {})'
                                         ''.format(self.channels, quadratures_channels))

            coherence = self.calc_coherence_coef(quadratures[self.channels[0]],
                                                 quadratures[self.channels[1]])
            self.coherence.append(coherence)

    def finish(self) -> ActiveResult:
        """Output results"""
        # Gather all parameters
        global_params = self.global_parameters
        channels = self.channels

        phase_code = self.phase_code
        frequency = self.frequency
        pulse_length = self.pulse_length

        active_params = ActiveParameters(global_params, channels, self.valid_pulse_type,
                                         frequency, pulse_length, phase_code)

        # Convert evaluated quantities to 2-d arrays
        if self.eval_power:
            assert isinstance(self.power, dict)

            power = {}
            for ch in self.power:
                power[ch] = np.stack(self.power[ch])
        else:
            power = None

        if self.eval_coherence:
            assert isinstance(self.coherence, list)

            coherence = np.stack(self.coherence)
        else:
            coherence = None
        result = ActiveResult(active_params, self.time_marks, power, coherence)
        return result


class LongPulseActiveHandler(ActiveHandler):
    """Class for processing of narrowband series (default channels_set 0, 2)"""
    valid_pulse_type = 'long'

    def __init__(self, *args, filter_half_band=25000, **kwargs):
        super().__init__(*args, **kwargs)
        self._filter = None
        self.filter_half_band = filter_half_band

    @property
    def filter(self) -> Dict[str, np.ndarray]:
        if self._filter is None:
            nyq = 0.5 * self.global_parameters.sampling_frequency['Hz']
            crit_norm_freq = self.filter_half_band / nyq

            filt = signal.butter(7, crit_norm_freq)  # type: Tuple[np.ndarray, np.ndarray]
            self._filter = {'numerator': filt[0], 'denominator': filt[1]}
        return self._filter

    def preproc_quads(self, quadratures: np.ndarray, axis=1) -> np.ndarray:
        return signal.lfilter(
            self.filter['numerator'], self.filter['denominator'], quadratures, axis=axis
        )


class ShortPulseActiveHandler(ActiveHandler):
    """Class for processing of wideband series (default channels_set 1, 3)"""
    valid_pulse_type = 'short'

    def __init__(self, global_parameters, channels, pulse_length, frequency, phase_code,
                 n_fft=None, h_step=None, eval_power=True, eval_coherence=False):
        super().__init__(global_parameters, channels, pulse_length, frequency, phase_code,
                         n_fft, h_step, eval_power, eval_coherence)
        self._code = None
        self._is_noise = None

    @property
    def is_noise(self):
        if self._is_noise is None:
            self._is_noise = (self.phase_code == 0) and (int(self.pulse_length['us']) == 0)
        return self._is_noise

    @property
    def code(self):
        if self._code is None:
            dlength = self.pulse_length['us'] * self.global_parameters.sampling_frequency['MHz']
            self._code = square_barker(int(dlength), self.phase_code)
        return self._code

    def preproc_quads(self, quadratures: np.ndarray, axis=1) -> np.ndarray:
        if self.is_noise:
            return quadratures
        else:
            return np.apply_along_axis(signal.correlate, axis=axis, arr=quadratures,
                                       in2=self.code, mode='same')


class ActiveSupervisor(Supervisor):
    """Supervisor that manages active processing"""
    def __init__(self, n_accumulation: int, timeout: timedelta,
                 n_fft: int = None, h_step: float = None,
                 eval_power: bool = True, eval_coherence: bool = False):
        """

        Args:
            n_accumulation:
            timeout:
        """
        super().__init__(timeout=timeout)
        self.n_accumulation = n_accumulation
        self.timeout = timeout
        self.n_fft = n_fft
        self.h_step = h_step

        self.eval_power = eval_power
        self.eval_coherence = eval_coherence

    def aggregator(self, packages: Iterator[TimeSeriesPackage], drop_timeout_series: bool = True
                   ) -> Generator[Tuple[SeriesParameters, np.ndarray, np.ndarray], Any, Any]:
        """Aggregate series with equal parameters from packages.
         Quadratures are accumulated to form 2-d arrays.

        Args:
            packages: Series packages iterator.
            drop_timeout_series: Defines behaviour when timeout occur.
                If True, drop already accumulated
            series. If False, yield them.

        Yields:
            params: Unique parameters that corresponds to accumulated series.
            time_marks: All consecutive time marks.
            quadratures: 2-D Array of accumulated quadratures. Shape [n_accumulation x n_samples]
        """
        def to_arrays(marks, quads):
            return np.array(marks, dtype=datetime), np.stack(quads)

        timeout_filter = self.timeout_filter()
        next(timeout_filter)  # Init coroutine

        buffer = defaultdict(lambda: ([], []))
        for package in packages:
            if timeout_filter.send(package):
                if not drop_timeout_series:
                    # Yield what was accumulated
                    for params, acc_marks, acc_quads in buffer:
                        time_marks, quadratures = to_arrays(acc_marks, acc_quads)
                        yield params, time_marks, quadratures

                # Reset buffer
                buffer = defaultdict(lambda: ([], []))

            for time_series in package.time_series_list:
                params = time_series.parameters
                acc_marks, acc_quads = buffer[params]

                # Append new record. If full, yield and reset buffer.
                acc_marks.append(time_series.time_mark)
                acc_quads.append(time_series.quadratures)

                if len(acc_marks) >= self.n_accumulation:
                    time_marks, quadratures = to_arrays(acc_marks, acc_quads)
                    del buffer[params]
                    yield params, time_marks, quadratures

    def agg_params(self) -> Generator[ActiveParameters, SeriesParameters, None]:
        unique_active_params = set()
        while True:
            params = yield ()


    def init_handler(self, parameters: ActiveParameters):
        ActiveParameters()
        global_parameters
        channels
        pulse_length
        frequency
        phase_code
        n_fft
        h_step
        eval_power
        eval_coherence

        common_kwargs = {
            'global_parameters': parameters.global_parameters,
            'eval_power': self.eval_power,
            'eval_coherence': self.eval_coherence
        }
        if parameters.pulse_type == 'short':
            return ShortPulseActiveHandler()
        elif parameters.pulse_type == 'long':
            return LongPulseActiveHandler(**common_kwargs)
        else:
            raise ValueError('Unknown pulse type')

    def get_identifier(self, params):
        """

        Args:
            params:

        Returns:

        """
