"""
Classes for physical units.
"""
import json
from functools import total_ordering
from typing import Union
import numpy as np


JSON_UNIT_TYPE_STR = '_unit_type'
JSON_UNIT_VALUE_STR = 'value'

_unit_types_registry = {}


def register_unit(cls):
    _unit_types_registry[cls.__name__] = cls
    return cls


@total_ordering
class Unit:
    """
    Provide types to store physical values.
    Only transition between units and equality comparison is supported.
    Since the class is used to only store units, arithmetic operations and array operations
    are not allowed.
    """
    available_units = {}

    def __init__(self, value, unit):
        if unit not in self.available_units:
            raise KeyError(unit)

        if isinstance(value, int):
            value = float(value)

        if isinstance(value, np.ndarray):
            value = value.copy().squeeze()  # Copy for later in-place operations
            value.dtype = np.float
            self.size = value.size
            self.shape = value.shape
        elif isinstance(value, (float, complex)):
            self.size = 1
            self.shape = ()
        else:
            self.size = len(value)
            self.shape = (self.size, )

        self._value = value
        self._unit = unit
        self._ref_unit = next(iter(self.available_units.keys()))

    def __getitem__(self, unit: str) -> Union[float, np.ndarray]:
        """Return values in the given unit.

        Only values in last used unit are stored to optimize access and memory.

        Args:
            unit: Units.

        Returns:
            values: Values in given units.
        """
        if isinstance(unit, int):
            raise ValueError("__getitem__ is used as index. Use __iter__ to iterate over values.")

        ratio = self.available_units[self._unit] / self.available_units[unit]
        return ratio * self._value

    def __iter__(self):
        if self.size == 1:
            raise ValueError('Cannot iterate over 1-sized {}'.format(self.__class__.__name__))
        else:
            for val in self._value:
                yield self.__class__(val, self._unit)

    def __str__(self):
        if self.size == 1 and isinstance(self._value, np.ndarray):
            value = self._value.item()
        else:
            value = self._value
        return str(value) + ' ' + str(self._unit)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, repr(self._value), repr(self._unit))

    def __eq__(self, other: 'Unit'):
        if self.__class__.__name__ != other.__class__.__name__:
            raise TypeError('Wrong type {} (expected {})'.format(
                repr(other), self.__class__.__name__)
            )
        if self.shape != ():
            raise NotImplementedError('Comparision for arrays is ambiguous')

        ratio = self.available_units[self._unit] / other.available_units[other._unit]
        return self._value * ratio == other._value

    def __le__(self, other: 'Unit'):
        if self.__class__.__name__ != other.__class__.__name__:
            raise TypeError('Wrong type {} (expected {})'.format(
                repr(other), self.__class__.__name__)
            )

        if self.shape != ():
            raise NotImplementedError('Comparision for arrays is ambiguous')

        ratio = self.available_units[self._unit] / other.available_units[other._unit]
        return self._value * ratio <= other._value

    def __hash__(self):
        # Raise error if underlying value is a numpy array
        return hash(self[self._ref_unit])

    def __add__(self, other: 'Unit'):
        ref_unit = self._ref_unit
        new_value = self[ref_unit] + other[ref_unit]
        return self.__class__(new_value, ref_unit)

    def __sub__(self, other: 'Unit'):
        ref_unit = self._ref_unit
        new_value = self[ref_unit] - other[ref_unit]
        return self.__class__(new_value, ref_unit)


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


class UnitsJSONDecoder(json.JSONDecoder):
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
