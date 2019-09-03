import json
import warnings
import logging
from abc import ABCMeta, abstractmethod
import datetime as dt

import numpy as np
from scipy import fftpack
from typing import IO, List, TextIO, Union, Generator, Any, Iterator, Tuple, BinaryIO, Dict

from iisr.data_manager import DataManager
from iisr.iisr_io import TimeSeriesPackage
from iisr.representation import ReprJSONEncoder, ReprJSONDecoder

from iisr.utils import merge, infinite_defaultdict
from iisr import StdFile, AnnotatedData

DATE_FMT = '%Y-%m-%d'


def _merge_stdfiles(file1: StdFile, file2: StdFile) -> StdFile:
    def key_fn(data: AnnotatedData):
        return data.header.start_time

    new_power = merge(file1.power, file2.power, key=key_fn)
    new_spectra = merge(file1.spectra, file2.spectra, key=key_fn)
    return StdFile(new_power, new_spectra)


def _merge_and_save_stdfiles(results, manager):
    assert all(results[0].dates == res.dates for res in results)
    dates = results[0].dates
    for date in dates:
        grouped_files = infinite_defaultdict()
        for result in results:
            stdfiles = result.to_std(date)
            for ch, stdfile in stdfiles.items():
                horn = ch.horn
                pulse_type = ch.pulse_type
                stdfile = stdfiles[ch]
                freq = stdfile.power[0].header.frequency_hz / 1e6
                pulse_len = stdfile.power[0].header.pulse_length_us

                if pulse_type == 'short':
                    if pulse_len == 0:
                        # Noise channel, ignore
                        continue

                    # Shift grouping frequency to long channel equivalent
                    freq -= 0.3

                grouped_files[horn, freq][pulse_type][pulse_len] = stdfile

        for (horn, freq), files in grouped_files.items():
            files: Dict[str, Dict[int, StdFile]]
            if len(files) != 2:
                raise ValueError('Expect short and long pulses to be present in files')

            if len(files['short']) != 1:
                raise ValueError('Expect single short pulse')

            short_stdfile = list(files['short'].values())[0]

            if len(files['long']) > 2:
                raise ValueError('Expect at most 2 long pulses (700 and 900)')

            for pulse_len, long_stdfile in files['long'].items():
                stdfile = _merge_stdfiles(short_stdfile, long_stdfile)

                filename = '{}_{}_f{:.2f}_len{}.std' \
                           ''.format(date.strftime('%Y%m%d'), horn, freq, int(pulse_len))
                manager.save_stdfile(stdfile, filename)


class HandlerResult(metaclass=ABCMeta):
    dates = NotImplemented  # type: Union[List[dt.date], dt.date]
    mode_name = NotImplemented  # type: str

    @property
    @abstractmethod
    def short_name(self) -> str:
        pass

    @abstractmethod
    def save_txt(self, data_manager: DataManager):
        """Save results to file. If save_date is passed, save specific date."""

    @classmethod
    @abstractmethod
    def load_txt(cls, file: List[IO]) -> 'HandlerResult':
        """Load results from list of files."""

    @abstractmethod
    def save_pickle(self, data_manager: DataManager):
        """Pickle results to file."""

    @classmethod
    @abstractmethod
    def load_pickle(cls, file: List[BinaryIO]) -> 'HandlerResult':
        """Unpickle results."""


class HandlerBatch(metaclass=ABCMeta):
    pass


class Handler(metaclass=ABCMeta):
    """Parent class for various types of first-stage processing."""
    results = NotImplemented

    @abstractmethod
    def handle(self, batch: HandlerBatch):
        """Processing algorithm that returns intermediate result"""

    @abstractmethod
    def finish(self):
        """Finish processing and return the results"""

    @staticmethod
    def calc_power(q: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate signal power.

        Args:
            q: Array of complex numbers.
            axis: Averaging axis. Defaults to 0.

        Returns:
            power: Array of floats.
        """
        return (q.real ** 2 + q.imag ** 2).mean(axis=axis)

    @staticmethod
    def calc_coherence_coef(q1: np.ndarray, q2: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate coherence coefficient between two signals.

        Input array must have same shape.

        Args:
            q1: Array of complex numbers.
            q2: Array of complex numbers.
            axis: Averaging axis. Defaults to 0.

        Returns:
            coherence: Array of floats.
        """
        return (q1 * q2.conj()).mean(axis=axis)

    @staticmethod
    def calc_fft(quadratures: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate fft for quadratures. No shifting is applied, i.e. first is zero frequency,
        then positive frequencies, then negative frequencies.

        Args:
            quadratures: Array of complex numbers.
            axis: Axis along which operation is applied.

        Returns:
            fft: Fast Fourier Transform of the quadratures
        """
        fft = fftpack.fft(quadratures, axis=axis)
        return fft


class HandlerParameters(metaclass=ABCMeta):
    params_to_save = NotImplemented
    header_n_params_key = 'params_json_length'

    def __eq__(self, other: 'HandlerParameters'):
        for name in self.params_to_save:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        return hash(tuple(getattr(self, name) for name in self.params_to_save))

    @classmethod
    def read_params_from_txt(cls, file: TextIO):
        start_pos = file.tell()
        header = file.readline()
        if not header.startswith(cls.header_n_params_key):
            warnings.warn('File has no header parameters')
            file.seek(start_pos)
            return

        json_length = int(header.split()[1]) + 1  # 1 for last '\n'
        params = json.loads(file.read(json_length), cls=ReprJSONDecoder)
        return params

    @classmethod
    @abstractmethod
    def load_txt(cls, file: TextIO):
        """Load parameters from text file"""

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


def timeout_filter(timeout, invalid_time_mark_policy: str = 'yield_timeout'
                   ) -> Generator[bool, TimeSeriesPackage, Any]:
    """Coroutine to check if time difference between consequent packages exceeds timeout.

    Args:
        invalid_time_mark_policy: Determine policy when new time mark is earlier than previous:
            'yield_timeout' - yield timeout flag; error message will be logged.
            'raise_exception' - raise exception


    Yields:
        is_timeout: If timeout occur at given package.
    """
    assert invalid_time_mark_policy in ['yield_timeout', 'raise_exception']

    prev_time_mark = None
    is_timeout = False

    while True:
        package = yield is_timeout

        if prev_time_mark is None:
            time_diff = dt.timedelta(microseconds=1)
        else:
            time_diff = package.time_mark - prev_time_mark

        if time_diff > timeout:
            is_timeout = True

        elif time_diff <= dt.timedelta(0):
            # Known issues
            test_time_mark = package.time_mark.replace(second=0, microsecond=0)
            if test_time_mark == dt.datetime(2015, 6, 5, 1, 36):
                is_timeout = True
            elif test_time_mark == dt.datetime(2015, 6, 5, 13, 7):
                is_timeout = True
            else:
                err_msg = 'New time mark is earlier than previous (new {}, prev {})'\
                          ''.format(package.time_mark, prev_time_mark)
                if invalid_time_mark_policy == 'yield_timeout':
                    is_timeout = True
                    logging.info(err_msg)
                else:
                    raise RuntimeError(err_msg)
        else:
            is_timeout = False

        prev_time_mark = package.time_mark


class Supervisor(metaclass=ABCMeta):
    AggregatorReturnType = Tuple[dt.date, HandlerParameters, HandlerBatch]

    """Supervisor manages data processing for specific mode of operation"""
    @abstractmethod
    def aggregator(self, packages: Iterator[TimeSeriesPackage],
                   ) -> Generator[AggregatorReturnType, Any, Any]:
        """Aggregate input packages by parameters to create arrays of quadratures"""

    @abstractmethod
    def init_handler(self, *args, **kwargs) -> Handler:
        """Initialize new handler"""

    @staticmethod
    def save_complete_results(handlers: Dict[HandlerParameters, Handler],
                              output_formats, data_manager, subfolders):
        results = []
        for handler in handlers.values():
            results.append(handler.finish())

        for out_fmt in output_formats:
            if out_fmt == 'pkl':
                for result in results:
                    result.save_pickle(data_manager, subfolders)
            elif out_fmt == 'std':
                _merge_and_save_stdfiles(results, data_manager)
            else:
                logging.warning('Unexpected format from config: {}'.format(out_fmt))

    def process_packages(self, packages: Iterator[TimeSeriesPackage],
                         data_manager: DataManager = None,
                         output_formats: List[str] = None,
                         subfolders: List[str] = None):
        """Process all packages from the generator to get list of results"""

        handlers = {}
        curr_date = None
        save_results = output_formats is not None and output_formats
        save_intermediate_txt = 'txt' in output_formats if save_results else None

        for date, key_params, batch in self.aggregator(packages):
            # Split results by date
            # When new date appears, finish all
            if curr_date is None:
                logging.info(f'Process date {date.strftime(DATE_FMT)}')
                curr_date = date
            elif curr_date < date:
                # Save the results
                logging.info(f'Save results for date {date.strftime(DATE_FMT)}')
                self.save_complete_results(handlers, output_formats, data_manager, subfolders)

                curr_date = date
                logging.info(f'Process date {date.strftime(DATE_FMT)}')
                handlers = {}  # erase all handlers
            elif curr_date == date:
                pass
            else:
                raise RuntimeError('New date is earlier than current date - cannot perform '
                                   'split by date')

            # If no there is no handler for given parameters, create it
            if key_params in handlers:
                handler = handlers[key_params]
            else:
                handler = self.init_handler(key_params)
                handlers[key_params] = handler

            # Process grouped series using handler
            intermediate_result = handler.handle(batch)
            if save_intermediate_txt:
                intermediate_result.append_to_txt(data_manager, subfolders)

        self.save_complete_results(handlers, output_formats, data_manager, subfolders)
