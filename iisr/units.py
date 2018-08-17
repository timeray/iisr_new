"""
Classes for physical units.
"""
from typing import Union
import numpy as np


class Unit:
    available_units = {}

    @property
    def value(self):
        return self._value

    def __init__(self, value, unit):
        if unit not in self.available_units:
            raise KeyError(unit)

        if isinstance(value, np.ndarray):
            value = value.copy()  # Copy for later in-place operations

        self._value = value
        self._cur_unit = unit

        if isinstance(value, (int, float, complex)):
            self.size = 1
        else:
            self.size = len(value)

    def __getitem__(self, unit: str) -> Union[float, np.ndarray]:
        """Return values in the given unit.

        Only values in last used unit are stored to optimize access and memory.

        Args:
            unit: Units.

        Returns:
            values: Values in given units.
        """
        if self._cur_unit != unit:
            ratio = self.available_units[self._cur_unit] / self.available_units[unit]
            self._value *= ratio
            self._cur_unit = unit
        return self._value

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return repr(self._value)


class Frequency(Unit):
    available_units = {'Hz': 1., 'kHz': 1e3, 'MHz': 1e6}


class TimeUnit(Unit):
    available_units = {'s': 1., 'ms': 1e-3, 'us': 1e-6}


class Distance(Unit):
    available_units = {'mm': 1e-3, 'm': 1., 'km': 1e3}