"""
Classes for physical units.
"""
import json
from typing import Union
import numpy as np


JSON_UNIT_TYPE_STR = '_unit_type'
JSON_UNIT_VALUE_STR = 'value'

_unit_types_registry = {}


def register_unit(cls):
    _unit_types_registry[cls.__name__] = cls
    return cls


class Unit:
    available_units = {}

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
        if self.size == 1 and isinstance(self._value, np.ndarray):
            value = self._value.item()
        else:
            value = self._value
        return str(value) + ' ' + str(self._cur_unit)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, repr(self._value), repr(self._cur_unit))

    def __eq__(self, other: 'Unit'):
        if self.__class__.__name__ != other.__class__.__name__:
            raise TypeError('Wrong unit {} (expected {})'.format(
                repr(other), self.__class__.__name__)
            )
        ratio = self.available_units[self._cur_unit] / other.available_units[other._cur_unit]
        if self._value * ratio == other._value:
            return True
        else:
            return False


@register_unit
class Frequency(Unit):
    available_units = {'Hz': 1., 'kHz': 1e3, 'MHz': 1e6}


@register_unit
class TimeUnit(Unit):
    available_units = {'s': 1., 'ms': 1e-3, 'us': 1e-6}


@register_unit
class Distance(Unit):
    available_units = {'mm': 1e-3, 'm': 1., 'km': 1e3}


def get_unit_by_name(name):
    return _unit_types_registry[name]


class UnitsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Unit):
            if obj.size != 1:
                raise TypeError('Only 1-sized Units may be json-serialized')
            return {JSON_UNIT_TYPE_STR: obj.__class__.__name__, JSON_UNIT_VALUE_STR: obj.__str__()}

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class UnitJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if JSON_UNIT_TYPE_STR not in obj:
            return obj
        type = get_unit_by_name(obj[JSON_UNIT_TYPE_STR])
        value_str = obj[JSON_UNIT_VALUE_STR].split()
        value = float(value_str[0])
        unit = (value_str[1])
        return type(value, unit)
