"""
Useful methods.
"""
from datetime import datetime

import numpy as np
from typing import Sequence

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
