"""
Useful methods.
"""
from collections import defaultdict
import datetime as dt

import numpy as np
from typing import Sequence, List, Callable, Union
from scipy.integrate import trapz, simps

DATE_FMT = '%Y-%m-%d'
TIME_FMT = '%H:%M:%S'

LOCAL_OFFSET = dt.timedelta(hours=8)
ZERO_OFFSET = dt.timedelta(0)


# UTC time zone
class UTC(dt.tzinfo):
    """UTC"""
    def utcoffset(self, dt):
        return ZERO_OFFSET

    def tzname(self, dt):
        return 'UT'

    def dst(self, dt):
        return ZERO_OFFSET


# Local time zone
class LocalTime(dt.tzinfo):
    """Local"""
    def utcoffset(self, dt):
        return LOCAL_OFFSET

    def dst(self, dt):
        return ZERO_OFFSET

    def tzname(self, dt):
        return 'LT'


utc_tz = UTC()
local_tz = LocalTime()

tz_dict = {'UT': utc_tz, 'LT': local_tz}


def uneven_mean(x: np.ndarray, y: np.ndarray, axis: int = -1, method: str = 'trapz'
                ) -> Union[np.ndarray, np.float]:
    """Compute mean of function with uneven sampling.

    Args:
        x: Sampling values.
        y: Function values.
        axis: Axis of averaging.
        method: Method 'trapz' or 'simps' for trapezoid and Simpson rules. 'trapz' performs faster
                but is less accurate.

    Returns:
        mean: Mean.

    """
    if x.size == 1:
        return y.item()

    if method == 'trapz':
        integral = trapz(y, x, axis=axis)
    elif method == 'simps':
        integral = simps(y, x, axis=axis)
    else:
        raise ValueError(f'Unknown method: {method}')
    return integral / (x[-1] - x[0])


def central_time(dtimes: Sequence[dt.datetime]) -> dt.datetime:
    """Return central time of datetime array.

    Args:
        dtimes: Sequence of datetimes.

    Returns:
        central_time: Central time of given sequence.
    """
    sorted_dtimes = sorted(dtimes)
    if len(sorted_dtimes) < 2:
        return sorted_dtimes[0]
    if not isinstance(sorted_dtimes[0], dt.datetime) or not isinstance(sorted_dtimes[-1], dt.datetime):
        raise ValueError('Input sequence should be sequence of datetimes')
    return sorted_dtimes[0] + (sorted_dtimes[-1] - sorted_dtimes[0]) / 2


def datetime2hour(dtimes: np.ndarray):
    """Convert array of datetimes to array of float hours. Date information will be lost.

    Args:
        dtimes: Array of datetimes.

    Returns:
        hours: Array of float hours.
    """
    return np.array([dtime.hour + dtime.minute / 60 + dtime.second / 3600 for dtime in dtimes])


def normalize2unity(arr: np.ndarray, axis: int = None):
    """Normalizes input array to [0, 1] range"""
    return (arr - arr.min(axis=axis, keepdims=True)) / arr.ptp(axis=axis, keepdims=True)


def infinite_defaultdict():
    return defaultdict(infinite_defaultdict)


def merge(l1: List, l2: List, key: Callable = None):
    if key is None:
        key = lambda x: x

    new_list = []
    n1 = n2 = 0
    while n1 < len(l1) and n2 < len(l2):
        val1 = l1[n1]
        val2 = l2[n2]

        if key(val1) < key(val2):
            new_list.append(val1)
            n1 += 1
        else:
            new_list.append(val2)
            n2 += 1

    if n1 < len(l1):
        new_list.extend(l1[n1:])

    if n2 < len(l2):
        new_list.extend(l2[n2:])

    return new_list
