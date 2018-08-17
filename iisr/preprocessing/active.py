from collections import defaultdict
from datetime import datetime, date

import numpy as np
from typing import List, TextIO, Dict

from iisr.preprocessing.representation import HandlerResult, Handler
from iisr.representation import Parameters
from iisr.utils import TIME_FMT, DATE_FMT, central_time
from iisr.units import Frequency, TimeUnit, Distance


__all__ = ['calc_delays', 'calc_distance', 'ActiveParameters', 'ActiveResult', 'ActiveHandler']


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
    def __init__(self, sampling_frequency: Frequency, delays: TimeUnit, channels: List,
                 frequency: Frequency, pulse_length: TimeUnit, phase_codes: List):
        self.sampling_frequency = sampling_frequency
        self.delays = delays

        if len(channels) != len(phase_codes):
            raise ValueError('Length of channels and phase_codes must be equal')

        # Sort by channels
        channel_argsort = sorted(range(len(channels)), key=lambda i: channels[i])
        self.channels = [channels[i] for i in channel_argsort]
        self.phase_codes = [phase_codes[i] for i in channel_argsort]

        self.frequency = frequency
        self.pulse_length = pulse_length


class ActiveResult(HandlerResult):
    def __init__(self, parameters: ActiveParameters, time_marks: List[datetime],
                 power: Dict[int, np.ndarray], coherence: Dict[str, np.ndarray]):
        """Result of processing of active experiment. It is expected, that power and spectrum are
        calculated for each (frequency, pulse_len, channel) and time marks are aligned.

        Size of all estimated values must be equal to size of time_marks.

        Args:
            parameters: Parameters for processed experiment.
            time_marks: Sorted array of datetimes.
            power: Dictionary (key for each channel) of power profiles.
                Each profiles should have shape [len(time_marks), n_samples].
            coherence: Dictionary (key for 'upper' and 'lower' horns) of coherence between channels.
                Each profiles should have shape [len(time_marks), n_samples].
        """
        if sorted(power.keys()) != parameters.channels:
            raise ValueError('Channels of power dictionary must be identical '
                             'to channels of parameters object')

        for pwr in power.values():
            if len(time_marks) != len(pwr):
                raise ValueError('Length of time_marks and all power arrays must be equal')

        if coherence is not None:
            for coh in coherence.values():
                if len(time_marks) != len(coh):
                    raise ValueError('Length of time_marks and all power arrays must be equal')

        self.parameters = parameters
        self.time_marks = time_marks
        self.power = power
        self.coherence = coherence

        # Calculate all involved dates
        self.dates = sorted(set((date(dt.year, dt.month, dt.day) for dt in self.time_marks)))

        # # Gather set of experiment parameters and processing options
        # # that uniquely identify the results
        # self.results_specification = {
        #     'parameters': parameters,
        #     'options': options,
        # }

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif hasattr(self.parameters, item):
            return getattr(self.parameters, item)
        else:
            raise AttributeError(item)

    def save_txt(self, file: TextIO, save_date: date = None):
        """Save results to specific directory. If date was passed, save only results corresponding
        to this date.

        Args:
            path_to_dir: Path to save directory.
            save_date: Date to save.
        """


        # Save data for given date
        def write_line(f, t_mark, *values):
            time_str = t_mark.strftime(TIME_FMT)
            date_str = t_mark.strftime(DATE_FMT)
            columns = [date_str, time_str]

            for value in values:
                columns.append(str(value))

            f.write(' '.join(columns))
            f.write('\n')

        if save_date is not None:
            if save_date not in self.dates:
                raise ValueError('Not results for given date {}'.format(save_date))

            for time_mark, power in zip(self.time_marks, self.power):
                if time_mark.date() != save_date:
                    continue

                write_line(file, time_mark, power)

        # Save data for all dates to single file
        else:
            for time_mark, power in zip(self.time_marks, self.power):
                write_line(file, time_mark, power)

    @classmethod
    def load_txt(cls, files: List[TextIO]):
        """"""
        time_marks = []
        power = []
        dtime_fmt = DATE_FMT + ' ' + TIME_FMT
        for file in files:
            for line in file:
                line = line.split()
                dtime_str = line[0] + ' ' + line[1]
                time_marks.append(datetime.strptime(dtime_str, dtime_fmt))
                power.append(float(line[2]))
        power = np.array(power)
        return cls(time_marks, params, power)


class ActiveHandler(Handler):
    def __init__(self, n_fft=None, h_step=None):
        self.nfft = n_fft
        self.h_step = h_step

        self.all_params = None
        self.time_marks = None
        self.power = None

        self.reset_buffers()

    def reset_buffers(self):
        self.all_params = set()
        self.time_marks = defaultdict(list)
        self.power = defaultdict(list)

    def process(self, params: Parameters, time_marks: np.ndarray, quadratures: np.ndarray):
        # Calculate power
        self.all_params.add(params)
        self.time_marks[params].append(central_time(time_marks))
        self.power[params].append(self.calc_power(quadratures))

    def finish(self):
        """Output results and free memory.

        Returns: List[ActiveResults]

        """
        results = []
        for params in self.all_params:
            results.append(ActiveResult(self.time_marks[params], params, self.power[params]))

        self.reset_buffers()
        return results