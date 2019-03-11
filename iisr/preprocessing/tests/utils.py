from itertools import product

from typing import List
from iisr import iisr_io
import datetime as dt
from iisr.tests.test_io import get_test_signal_time_series, get_test_parameters


def package_generator(dtimes: List[dt.datetime], params_list: List[iisr_io.SeriesParameters]):
    for dtime in dtimes:
        series_list = []
        cur_freq = None
        for params in params_list:
            if cur_freq is not None and cur_freq != params.frequency:
                yield iisr_io.TimeSeriesPackage(dtime, series_list)
                series_list = []

            cur_freq = params.frequency
            series_list.append(get_test_signal_time_series(dtime, test_params=params))

        yield iisr_io.TimeSeriesPackage(dtime, series_list)


def get_test_param_list(freqs=(155.5, 158.0), channels=(0, 1, 2, 3), n_samples=128):
    param_list = []
    for fr, ch in product(freqs, channels):
        params = {'freq': fr, 'channel': ch, 'n_samples': n_samples}
        param_list.append(get_test_parameters(**params))
    return param_list
