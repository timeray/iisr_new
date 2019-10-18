"""
Module for interpolation of pre-calculated sky noise temperature for
Irkutsk Incoherent Scatter Radar (IISR). Pre-calculated values are contained in
file DNR_AllFreqs.dat and take into account radar antenna pattern.

Create SkyNoiseIISR instance and use get_sky_temperature(dtime, freq) method.
"""
__all__ = ['SkyNoiseIISR']
# reference date: 1.1.2015
import os
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import RectBivariateSpline

DIRPATH = os.path.dirname(__file__)
DATA_PATH_DICT = {
    'upper':
        {'gsm': os.path.join(DIRPATH, 'IISR_noise_upper.dat'),
         'gsm2016': os.path.join(DIRPATH, 'IISR_noise_upper_gsm2016.dat')},
    'lower':
        {'gsm': os.path.join(DIRPATH, 'IISR_noise_lower.dat'),
         'gsm2016': os.path.join(DIRPATH, 'IISR_noise_lower.dat')}
}


def read_noise(filename):
    """
    Read modelled IISR sky noise from pre-calculated file.
    """
    with open(filename, 'r') as file:
        time, freq, temp = np.loadtxt(file, skiprows=1, unpack=True)

    time = np.unique(time)
    freq = np.unique(freq)
    temp = temp.reshape(time.size, freq.size)
    return time, freq, temp


class SkyNoiseIISR:
    """
    Interpolate sky noise temperatures for IISR.
    
    Note: returned temperatures take into account single polarization
    received at the IISR, so there no additional need to consider it.
    """

    _start_date = datetime(2015, 1, 1)

    def __init__(self, horn_type, model='gsm'):
        try:
            time, freq, temp = read_noise(DATA_PATH_DICT[horn_type][model])
        except ValueError:
            raise ValueError('horn_type should be "upper" or "lower",'
                             'model should be "gsm", "gsm2016"')

        self.sky_noise = RectBivariateSpline(
            time, freq, temp
        )
    
    def get_sky_temperature(self, dtime, freq):
        """
        Return sky noise temperature at IISR receiver input.
        
        @param dtime: datetime or int
            Date and time when sky noise should be evaluated. Original map
            was calculated for specific date so offset should be added to
            estimate specific date.
            If passed as int, dtime would be interpreted as offset in hours.
        
        @param freq: float or int
            Frequency at which sky noise should be evaluated, MHz.
        
        Note: returned temperatures take into account single polarization
        received at the IISR, so there no additional need to consider it.
        """
        if isinstance(dtime, datetime):
            diff = dtime - self._start_date
            dtime = dtime + timedelta(hours=(24 / 365.2425) * diff.days)
            offset = (dtime.hour + dtime.minute / 60
                      + dtime.second / 3600
                      + dtime.microsecond / 1e6 / 3600)
            return self.sky_noise(offset, freq).reshape(-1)[0]
        else:
            return self.sky_noise(dtime, freq).reshape(-1)[0]

    def get_avg_sky_temperature(self, time_intervals, freq, resolution):
        """
        Return sky noise averaged between starting and ending point
        of time interval. It will be better estimate of sky noise if
        measured noise is averaged.

        Parameters
        ----------
        time_intervals: 2-element sequence of datetimes
            Starting and ending points for averaging.
        freq: number
            IISR frequency, MHz.
        resolution: number
            Time resolution in seconds for averaging grid.

        Returns
        -------
        avg_sky_noise, float
            Averaged sky noise.
        
        Note: returned temperatures take into account single polarization
        received at the IISR, so there no additional need to consider it.
        """
        dtimes = np.arange(time_intervals[0], time_intervals[1],
                           timedelta(seconds=resolution), dtype=datetime)
        diff = dtimes - self._start_date
        dtimes += [timedelta(hours=(24 / 365.2425) * d.days) for d in diff]
        offset = np.array([dt.hour + dt.minute/60 + dt.second/3600
                           + dt.microsecond/1e6/3600
                           for dt in dtimes])
        offset.sort()  # RectBivariateSpline access only sorted points!
        return self.sky_noise(offset, freq).reshape(-1).mean()
