"""
Useful methods.
"""
from collections import defaultdict
from datetime import datetime

import numpy as np
from typing import Sequence, List, Callable

DATE_FMT = '%Y-%m-%d'
TIME_FMT = '%H:%M:%S'


def central_time(dtimes: Sequence[datetime]) -> datetime:
    """Return central time of datetime array.

    Args:
        dtimes: Sequence of datetimes.

    Returns:
        central_time: Central time of given sequence.
    """
    sorted_dtimes = sorted(dtimes)
    if len(sorted_dtimes) < 2:
        return sorted_dtimes[0]
    if not isinstance(sorted_dtimes[0], datetime) or not isinstance(sorted_dtimes[-1], datetime):
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