import json
import warnings
from abc import ABCMeta, abstractmethod, abstractclassmethod
from datetime import date, timedelta, datetime
from scipy import fftpack

import numpy as np
from typing import IO, List, TextIO, Union, Generator, Any, Iterator, Tuple, Sequence

from iisr.io import SeriesParameters, TimeSeriesPackage
from iisr.representation import ReprJSONEncoder, ReprJSONDecoder


class HandlerResult(metaclass=ABCMeta):
    dates = NotImplemented  # type: Union[List[date], date]
    mode_name = NotImplemented  # type: str

    @property
    @abstractmethod
    def short_name(self) -> str:
        pass

    @abstractmethod
    def save_txt(self, file: IO, save_date: date = None):
        """Save results to file. If save_date is passed, save specific date."""

    @abstractclassmethod
    def load_txt(self, file: List[IO]) -> 'HandlerResult':
        """Load results from list of files."""


class Handler(metaclass=ABCMeta):
    """Parent class for various types of first-stage processing."""
    @abstractmethod
    def process(self, time_marks: np.ndarray, quadratures: Any):
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


class ResultParameters(metaclass=ABCMeta):
    params_to_save = NotImplemented
    header_n_params_key = 'params_json_length'

    def __eq__(self, other: 'ResultParameters'):
        for name in self.params_to_save:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

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

    @abstractclassmethod
    def load_txt(self, file: TextIO):
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


class Supervisor(metaclass=ABCMeta):
    """Supervisors are classes to manage data processing"""
    def __init__(self, timeout: timedelta):
        self.timeout = timeout

    def timeout_filter(self) -> Generator[bool, TimeSeriesPackage, Any]:
        """Coroutine to check if time difference between consequent packages exceeds timeout.

        Yields:
            is_timeout: If timeout occur at given package.
        """
        prev_time_mark = None
        is_timeout = False

        while True:
            package = yield is_timeout

            if prev_time_mark is None:
                time_diff = timedelta(0)
            else:
                time_diff = package.time_mark - prev_time_mark

            if time_diff > self.timeout:
                is_timeout = True

            elif time_diff < timedelta(0):
                raise RuntimeError(
                    'New time mark is earlier than previous (new {}, prev {})'
                    ''.format(package.time_mark, prev_time_mark)
                )

            prev_time_mark = package.time_mark

    @abstractmethod
    def aggregator(self, packages: Iterator[TimeSeriesPackage], drop_timeout_series: bool = True
                   ) -> Generator[Tuple[SeriesParameters, np.ndarray, np.ndarray], Any, Any]:
        """Aggregate input packages to form numpy arrays"""

    @abstractmethod
    def init_handler(self, *args, **kwargs):
        """Initialize new handler"""

    def process_packages(self, packages: Generator) -> List[HandlerResult]:
        """Process all packages from the generator to get list of results"""
        # Group series by parameters to n_acc, check for timeouts

        handlers = {}
        for params, time_marks, quadratures in self.aggregator(packages):
            # If no there is no handler for given parameters, create it
            if params not in handlers:
                handler = self.init_handler(params)
                handlers[params] = handler
            else:
                handler = handlers[params]

            # Process grouped series using handler
            handler.handle(time_marks, quadratures)

        # Get results
        results = []
        for handler in handlers:
            results.append(handler.finish())

        return results
