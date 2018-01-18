"""
Classes for physical units.
"""


class Unit:
    available_keys = {}

    @property
    def value(self):
        return self._value

    def __init__(self, value, unit):
        self._value = value * self.available_keys[unit]

    def __getitem__(self, unit: str):
        return self._value / self.available_keys[unit]

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return repr(self._value)


class Frequency(Unit):
    available_keys = {'Hz': 1., 'kHz': 1e3, 'MHz': 1e6}


class Time(Unit):
    available_keys = {'s': 1., 'ms': 1e-3, 'us': 1e-6}
