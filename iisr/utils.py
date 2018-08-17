"""
Useful methods.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Sequence, Union


DATE_FMT = '%Y-%m-%d'
TIME_FMT = '%H:%M:%S'


def central_time(sorted_dtimes: Sequence[datetime]) -> datetime:
    """Return central time of datetime array.

    Args:
        sorted_dtimes: Sorted sequence of datetimes.

    Returns:
        central_time: Central time of given sequence.
    """
    if len(sorted_dtimes) < 2:
        raise ValueError('Input datetime array must have at least 2 elements')
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