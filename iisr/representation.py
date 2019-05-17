"""
Collect classes for IISR data representation.
"""
from typing import Sequence

__all__ = ['CHANNELS_INFO', 'Channel', 'ReprJSONDecoder', 'ReprJSONEncoder']

from iisr.units import UnitsJSONDecoder, UnitsJSONEncoder

CHANNELS_INFO = {
    0: {'type': 'long', 'horn': 'upper', 'band_type': 'narrow'},
    1: {'type': 'short', 'horn': 'upper', 'band_type': 'wide'},
    2: {'type': 'long', 'horn': 'lower', 'band_type': 'narrow'},
    3: {'type': 'short', 'horn': 'lower', 'band_type': 'wide'},
    20: {'type': 'long', 'horn': 'both', 'band_type': 'narrow'},
    31: {'type': 'short', 'horn': 'both', 'band_type': 'wide'},
}


class Channel:
    __slots__ = ['value', 'pulse_type', 'horn', 'band_type']

    def __init__(self, value):
        _valid_channels = [0, 1, 2, 3, 20, 31]  # 20 and 31 are sum channels
        if value not in _valid_channels:
            raise ValueError('Channel can be one of {}'.format(_valid_channels))
        self.value = value
        self.pulse_type = CHANNELS_INFO[value]['type']
        self.horn = CHANNELS_INFO[value]['horn']
        self.band_type = CHANNELS_INFO[value]['band_type']

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other: 'Channel'):
        if not isinstance(other, Channel):
            raise TypeError('Types {} and {} are not comparable'.format(Channel, int))
        return self.value == other.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)

    def __le__(self, other: 'Channel'):
        return self.value.__le__(other.value)

    def __lt__(self, other: 'Channel'):
        return self.value.__lt__(other.value)

    def __ge__(self, other: 'Channel'):
        return self.value.__ge__(other.value)

    def __gt__(self, other: 'Channel'):
        return self.value.__gt__(other.value)


def get_sum_channel(channels: Sequence[Channel]):
    assert len(channels) > 0
    pulse_type = channels[0].pulse_type
    assert all(ch.pulse_type == pulse_type for ch in channels)
    if pulse_type == 'long':
        return Channel(20)
    elif pulse_type == 'short':
        return Channel(31)
    else:
        raise ValueError('Unexpected pulse type: {}'.format(pulse_type))


ADJACENT_CHANNELS = {Channel(0): Channel(2), Channel(1): Channel(3),
                     Channel(2): Channel(0), Channel(3): Channel(1)}

JSON_REPR_TYPE_STR = '_repr_type'


class ReprJSONEncoder(UnitsJSONEncoder):
    def default(self, obj):
        if isinstance(obj, Channel):
            return {JSON_REPR_TYPE_STR: Channel.__name__, 'value': obj.value}
        return super().default(obj)


class ReprJSONDecoder(UnitsJSONDecoder):
    def object_hook(self, obj):
        if JSON_REPR_TYPE_STR not in obj:
            return super().object_hook(obj)

        if obj[JSON_REPR_TYPE_STR] == Channel.__name__:
            return Channel(obj['value'])
        else:
            raise ValueError('Unexpected name {}'.format(obj[JSON_REPR_TYPE_STR]))


