import json
import warnings
import logging
from abc import ABCMeta, abstractmethod
import datetime as dt

import numpy as np
from scipy import fftpack
from typing import IO, List, TextIO, Union, Generator, Any, Iterator, Tuple, BinaryIO

from iisr.iisr_io import TimeSeriesPackage
from iisr.representation import ReprJSONEncoder, ReprJSONDecoder


class HandlerResult(metaclass=ABCMeta):
    dates = NotImplemented  # type: Union[List[dt.date], dt.date]
    mode_name = NotImplemented  # type: str

    @property
    @abstractmethod
    def short_name(self) -> str:
        pass

    @abstractmethod
    def save_txt(self, file: IO, save_date: dt.date = None):
        """Save results to file. If save_date is passed, save specific date."""

    @classmethod
    @abstractmethod
    def load_txt(cls, file: List[IO]) -> 'HandlerResult':
        """Load results from list of files."""

    @abstractmethod
    def save_pickle(self, file: BinaryIO, save_date: dt.date = None):
        """Pickle results to file."""

    @classmethod
    @abstractmethod
    def load_pickle(cls, file: List[BinaryIO]) -> 'HandlerResult':
        """Unpickle results."""


class HandlerBatch(metaclass=ABCMeta):
    pass


class Handler(metaclass=ABCMeta):
    """Parent class for various types of first-stage processing."""
    @abstractmethod
    def handle(self, batch: HandlerBatch):
        """Processing algorithm"""

    @abstractmethod
    def finish(self) -> HandlerResult:
        """Returns results of processing and reset buffers.

        Returns:
            results: Processing results.
        """

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
    """Supervisor manages data processing for specific mode of operation"""
    @abstractmethod
    def aggregator(self, packages: Iterator[TimeSeriesPackage],
                   ) -> Generator[Tuple[HandlerParameters, HandlerBatch], Any, Any]:
        """Aggregate input packages by parameters to create arrays of quadratures"""

    @abstractmethod
    def init_handler(self, *args, **kwargs) -> Handler:
        """Initialize new handler"""

    def process_packages(self, packages: Iterator[TimeSeriesPackage]) -> List[HandlerResult]:
        """Process all packages from the generator to get list of results"""
        # Group series by parameters to n_acc, check for timeouts

        handlers = {}
        for key_params, batch in self.aggregator(packages):
            # If no there is no handler for given parameters, create it
            if key_params in handlers:
                handler = handlers[key_params]
            else:
                handler = self.init_handler(key_params)
                handlers[key_params] = handler

            # Process grouped series using handler
            handler.handle(batch)

        # Get results
        results = []
        for handler in handlers.values():
            results.append(handler.finish())

        return results
