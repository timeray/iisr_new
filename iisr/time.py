__all__ = ['time_zone_correction', 'utc_tz', 'local_tz', 'datetime2hour']

from datetime import datetime, timedelta, tzinfo

import numpy as np

LOCAL_OFFSET = timedelta(hours=8)
ZERO_OFFSET = timedelta(0)


# UTC time zone
class UTC(tzinfo):
    """UTC"""
    def utcoffset(self, dt):
        return ZERO_OFFSET

    def tzname(self, dt):
        return 'UT'

    def dst(self, dt):
        return ZERO_OFFSET


# Local time zone
class LocalTime(tzinfo):
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


@np.vectorize
def time_zone_correction(dtime, time_zone='UT'):
    """Convert dtime time zone to local IISR time zone.

    Args:
        dtime: datetime
            Datetime to convert.
        time_zone: 'UT' or 'LT'
            Desired timezone.
    """
    if dtime.tzinfo is None:
        dtime = dtime.replace(tzinfo=utc_tz)

    if time_zone == 'LT':
        return dtime.astimezone(tz_dict[time_zone])
    elif time_zone == 'UT':
        return dtime.astimezone(tz_dict[time_zone])
    else:
        raise ValueError("time_zone should be 'LT' or 'UT'")


def central_datetime(dtimes, axis=-1):
    dtimes = np.asarray(dtimes, dtype=datetime)

    if dtimes.shape[axis] < 2:
        return dtimes

    diff = np.diff(dtimes.take([0, -1], axis=axis)).squeeze()
    result = dtimes.take(0, axis=axis) + diff / 2
    return result


def datetime2hour(dtime):
    hour = (dtime.hour
            + dtime.minute / 60
            + dtime.second / 3600
            + dtime.microsecond / 1e6 / 3600)
    return hour


def dtime_linspace(start, end, num, endpoint=True):
    """
    Create numpy array containing num datetimes
    from start datetime to end datetime. Do not include end.
    With best regards, StackOverflow.
    """
    if not endpoint or num == 1:
        delta = (end - start) / num
    else:
        delta = (end - start) / (num - 1)

    increments = range(0, num) * np.array([delta] * num)
    return start + increments


if __name__ == '__main__':
    import numpy as np
    from numpy.testing import assert_equal, assert_raises

    def test_central_datetime_array():
        test_arr = np.array([datetime(2015, i, 1) for i in range(1, 13)])
        assert central_datetime(test_arr) == datetime(2015, 6, 17)

        days = range(1, 10)
        test_arr = np.array([[datetime(2015, 1, j, i) for i in range(0, 24)]
                             for j in days])
        result_arr = np.array([datetime(2015, 1, i, 11, 30) for i in days])

        assert_equal(central_datetime(test_arr), result_arr)

        assert central_datetime([datetime(2015, 1, 1)]) == [datetime(2015, 1, 1)]
        assert_equal(central_datetime(test_arr[:, :, None]),
                     test_arr[:, :, None])

    test_central_datetime_array()
    print('Tests passed')
