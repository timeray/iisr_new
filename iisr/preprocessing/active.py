import itertools as it
import json
import copy
import warnings
from collections import defaultdict
from datetime import datetime, date, timedelta

import numpy as np
from typing import List, TextIO, Dict, Sequence

from iisr.representation import CHANNELS, Channel
from iisr.preprocessing.representation import HandlerResult, Handler
from iisr.io import SeriesParameters, ExperimentParameters
from iisr.representation import ReprJSONEncoder, ReprJSONDecoder
from iisr.units import Frequency, TimeUnit, Distance
from iisr.utils import TIME_FMT, DATE_FMT, central_time

__all__ = ['calc_delays', 'calc_distance', 'ActiveParameters', 'ActiveResult', 'PowerParams',
           'CoherenceParams', 'LongPulseActiveHandler', 'ShortPulseActiveHandler',
           'EvalCoherenceError', 'ReadingError']


class ReadingError(Exception):
    pass


class EvalCoherenceError(Exception):
    pass


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


class ResultParameters:
    pass


class ActiveParameters(ResultParameters):
    params_to_save = ['sampling_frequency', 'total_delay', 'n_samples', 'channels', 'pulse_type',
                      'frequency', 'pulse_length', 'phase_code']
    header_n_params_key = 'params_json_length'

    def __init__(self, global_parameters: ExperimentParameters, channels: List[Channel],
                 pulse_type: str, frequency: Frequency, pulse_length: TimeUnit, phase_code: int):
        self.global_parameters = global_parameters

        if not channels:
            raise ValueError('Empty channels')

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

    def __eq__(self, other: 'ActiveParameters'):
        for name in self.params_to_save:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    @property
    def delays(self) -> TimeUnit:
        if self._delays is None:
            self._delays = calc_delays(self.sampling_frequency, self.total_delay, self.n_samples)
        return self._delays

    @property
    def distance(self) -> Distance:
        return calc_distance(self.delays)

    def save_txt(self, file: TextIO):
        params = {}
        for name in self.params_to_save:
            if not hasattr(self, name):
                raise ValueError('Unexpected parameter {}'.format(name))
            params[name] = getattr(self, name)

        dumped_params = json.dumps(params, indent=True, cls=ReprJSONEncoder)
        file.write('{} {}\n'.format(self.header_n_params_key, len(dumped_params)))
        file.write(dumped_params)
        file.write('\n')

    @classmethod
    def load_txt(cls, file: TextIO):
        start_pos = file.tell()
        header = file.readline()
        if not header.startswith(cls.header_n_params_key):
            warnings.warn('File has no header parameters')
            file.seek(start_pos)
            return

        json_length = int(header.split()[1]) + 1  # 1 for last '\n'
        params = json.loads(file.read(json_length), cls=ReprJSONDecoder)
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
            coherence: Coherence between channels. Complex values.
                Shape [len(time_marks), n_samples].
        """
        if power is not None:
            if sorted(power.keys()) != parameters.channels:
                raise AssertionError('Channels in power dictionary must be identical '
                                     'to channels in parameters object')

            for pwr in power.values():
                if len(time_marks) != len(pwr):
                    raise ValueError('Length of time_marks and all power arrays must be equal')

        if coherence is not None:
            if len(time_marks) != len(coherence):
                raise ValueError('Length of time_marks and all power arrays must be equal')

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
        distance_str = ['{}'.format(dist[dist_units]) for dist in self.distance]
        time_str = [time_mark.strftime(TIME_FMT) for time_mark in self.time_marks[date_mask]]
        date_str = [time_mark.strftime(DATE_FMT) for time_mark in self.time_marks[date_mask]]

        # Form long column iterators: distance changes every row, time and date change after all
        # distances
        float_fmt = '{{:.{}f}}'.format(precision)
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
    valid_band_type = NotImplemented

    def __init__(self, n_fft=None, h_step=None, eval_power=True, eval_coherence=False):
        self.nfft = n_fft
        self.h_step = h_step

        self.eval_power = eval_power
        self.eval_coherence = eval_coherence

        # Channels that correspond to the pulse type
        valid_channels = sorted([ch for ch in CHANNELS if ch.pulse_type == self.valid_pulse_type])
        if len(valid_channels) != 2:
            raise AssertionError('Expect 2 channels, but {} are given'.format(len(valid_channels)))
        self.adjacent_channels = {valid_channels[0]: valid_channels[1],
                                  valid_channels[1]: valid_channels[0]}

        self.valid_input_params = None
        self.actual_input_params = None
        self.time_marks = None
        self.power = None
        self.coherence = None

        self.await_input = None

        self.reset_buffers()

    def reset_buffers(self):
        self.valid_input_params = set()
        self.actual_input_params = set()
        self.time_marks = []
        self.power = defaultdict(list) if self.eval_power else None
        self.coherence = [] if self.eval_coherence else None

        self.await_input = None

    def validate(self, params: SeriesParameters) -> bool:
        """Check if parameters correspond to the handler.

        Args:
            params: Parameters to validate.

        Returns:
            valid: True if parameters match.
        """
        # If not initialized
        if not self.valid_input_params and params.pulse_type == self.valid_pulse_type:
            return True
        # If initialized
        elif params in self.valid_input_params:
            return True
        else:
            return False

    def _get_adjacent_params(self, params: SeriesParameters) -> SeriesParameters:
        """Return params with adjacent channel

        Args:
            params: Input parameter.

        Returns:
            adjacent_params: Adjacent parameters.

        """
        adjacent_params = copy.deepcopy(params)
        adjacent_params.channel = self.adjacent_channels[params.channel]
        return adjacent_params

    def _process(self, time_marks: np.ndarray, quadratures: Dict[Channel, np.ndarray]):
        """Function to evaluate power and coherence when all channels arrived"""
        self.time_marks.append(central_time(time_marks))
        channels = sorted(quadratures.keys())

        if self.eval_power:
            for ch in channels:
                self.power[ch].append(self.calc_power(quadratures[ch]))

        if self.eval_coherence:
            if sorted(channels) != sorted(self.adjacent_channels):
                raise EvalCoherenceError('Cannot evaluate coherence: incorrect channels '
                                         '(expected {} got {})'
                                         ''.format(self.adjacent_channels, channels))

            coherence = self.calc_coherence(quadratures[channels[0]], quadratures[channels[1]])
            self.coherence.append(coherence)

    def process(self, params: SeriesParameters, time_marks: np.ndarray, quadratures: np.ndarray):
        # Check that parameters are valid
        if not self.validate(params):
            raise ValueError('Given parameters are invalid.')

        # If empty - initialize
        if not self.actual_input_params:
            self.valid_input_params.add(params)
            self.valid_input_params.add(self._get_adjacent_params(params))

        if params not in self.actual_input_params:
            self.actual_input_params.add(params)

        if len(quadratures.shape) != 2:
            raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        if len(time_marks) != len(quadratures):
            raise ValueError('Input quadratures should have shape (n_time_marks, n_samples)')

        # Join adjacent channels or compute previous channel
        channel = params.channel
        if self.await_input is not None:
            prev_params, prev_time_marks, prev_quadratures = self.await_input
            prev_channel = prev_params.channel
            # If adjacent channel did not appear
            if channel == prev_channel:
                self._process(prev_time_marks, {prev_channel: prev_quadratures})
                self.await_input = params, time_marks, quadratures
            # If adjacent channel appeared, process and reset await_input
            elif channel == self.adjacent_channels[prev_channel]:
                if not all(time_marks == prev_time_marks):
                    raise RuntimeError('Invalid time marks of adjacent channel {}'
                                       ''.format(prev_channel))
                joint_quads = {channel: quadratures, prev_channel: prev_quadratures}
                self._process(time_marks, joint_quads)
                self.await_input = None
            else:
                raise RuntimeError('Invalid input parameters (expect same or adjacent channels)')
        else:
            self.await_input = params, time_marks, quadratures

    def finish(self) -> ActiveResult:
        """Output results and free memory."""
        # Finalize all computations (that may arise, because we always wait for adjacent channel)
        if self.await_input is not None:
            last_params, last_time_marks, last_quadratures = self.await_input
            self._process(last_time_marks, {last_params.channel: last_quadratures})

        # Gather all parameters
        # Expect 2 adjacent channels of same band_type or 1 channel of band_type
        self.actual_input_params = list(self.actual_input_params)
        assert len(self.actual_input_params) == 2 or len(self.actual_input_params) == 1

        if len(self.actual_input_params) == 2:
            test_params = SeriesParameters.REFINED_PARAMETERS - {'channel'}
            for param_name in test_params:
                assert getattr(self.actual_input_params[0], param_name) \
                       == getattr(self.actual_input_params[1], param_name), param_name

        global_params = self.actual_input_params[0].global_parameters
        channels = []
        for params in self.actual_input_params:
            channels.append(params.channel)
        channels = sorted(channels)

        phase_code = self.actual_input_params[0].phase_code
        frequency = self.actual_input_params[0].frequency
        pulse_length = self.actual_input_params[0].pulse_length

        active_params = ActiveParameters(global_params, channels, self.valid_pulse_type,
                                         frequency, pulse_length, phase_code)

        # Convert evaluated quantities to 2-d arrays
        if self.eval_power:
            power = {}
            for ch in self.power:
                power[ch] = np.stack(self.power[ch])
        else:
            power = None

        if self.eval_coherence:
            coherence = np.stack(self.coherence)
        else:
            coherence = None
        result = ActiveResult(active_params, self.time_marks, power, coherence)

        self.reset_buffers()
        return result


class LongPulseActiveHandler(ActiveHandler):
    """Class for processing of narrowband series (default channels 0, 2)"""
    valid_pulse_type = 'long'


class ShortPulseActiveHandler(ActiveHandler):
    """Class for processing of wideband series (default channels 1, 3)"""
    valid_pulse_type = 'short'
