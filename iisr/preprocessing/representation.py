from abc import abstractmethod, ABCMeta
from datetime import date

import numpy as np
from typing import IO, List

from iisr.representation import SeriesParameters


class HandlerResult:
    def save_txt(self, file: IO, save_date: date = None):
        """Save results to file. If save_date is passed, save specific date."""

    def load_txt(self, file: List[IO]):
        """Load results from list of files."""


class Handler(metaclass=ABCMeta):
    """Parent class for various types of first-stage processing."""
    @abstractmethod
    def process(self, params: SeriesParameters, time_marks: np.ndarray, quadratures: np.ndarray):
        """Processing algorithm"""

    @abstractmethod
    def validate(self, params):
        """Check if parameters correspond to the handler.

        Args:
            params: Parameters to validate.

        Returns:
            valid: True if parameters match.
        """
        pass

    @abstractmethod
    def finish(self) -> HandlerResult:
        """Returns results of processing and reset buffers.

        Returns:
            results: Processing results.
        """

    def calc_power(self, q: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate signal power.

        Args:
            q: Array of complex numbers.
            axis: Averaging axis. Defaults to 0.

        Returns:
            power: Array of floats.
        """
        return (q.real ** 2 + q.imag ** 2).mean(axis=axis)

    def calc_coherence(self, q1: np.ndarray, q2: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate coherence between two signals.

        Input array must have same shape.

        Args:
            q1: Array of complex numbers.
            q2: Array of complex numbers.
            axis: Averaging axis. Defaults to 0.

        Returns:
            coherence: Array of floats.
        """
        return (q1 * q2.conj()).mean(axis=axis)