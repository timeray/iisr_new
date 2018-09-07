import numpy as np
from iisr.representation import CHANNELS_INFO
from iisr.io import SeriesParameters, SignalTimeSeries, ExperimentGlobalParameters
from datetime import datetime, timedelta


def get_test_raw_parameters(freq=155000, stel='st1', channel=0, year=2015, month=6,
                            day=14, hour=5, minute=55, second=59, millisecond=850):
    pulse_type = CHANNELS_INFO[channel]['type']
    test_raw_parameters = {
        'reserved': 0,
        'mode': 1,
        'step': 2,
        'number_all': 2048,
        'number_after': 1024,
        'first_delay': 1000,
        'channel': channel,
        'date_year': year,
        'date_mon_day': (month << 8) + day,
        'time_h_m': (minute << 8) + hour,
        'time_sec': second,
        'time_msec': millisecond,
        '{}_{}_fr_lo'.format(stel, pulse_type): freq & 0xFFFF,
        '{}_{}_fr_hi'.format(stel, pulse_type): freq >> 16,
        '{}_{}_len'.format(stel, pulse_type): 700,
        'phase_code': 0,
        'average': 32,
        'offset_st1': 80,
        'sample_freq': 1000,
        'version': 3
    }
    return test_raw_parameters


def get_test_parameters(n_samples=2048, freq=155.5, pulse_len=700,
                        sampling_freq=1000., channel=0, phase_code=0, total_delay=1000,
                        pulse_type='long', rest_raw_parameters=None):
    """
    Returns
    -------
    options: SeriesParameters
    """
    global_params = ExperimentGlobalParameters(sampling_freq, n_samples, total_delay)
    test_parameters = SeriesParameters(global_params, channel, freq, pulse_len, phase_code,
                                       pulse_type)
    if rest_raw_parameters is not None:
        test_parameters.rest_raw_parameters = rest_raw_parameters
    return test_parameters


def get_test_signal_time_series():
    """
    Returns
    -------
    time_series: SignalTimeSeries
    """
    time_mark = datetime(2015, 6, 7, 8, 9, 10, 11)
    test_params = get_test_parameters()
    quad_i = np.random.randint(-100, 100, test_params.n_samples)
    quad_q = np.random.randint(-100, 100, test_params.n_samples)
    quadratures = quad_i + 1j * quad_q
    time_series = SignalTimeSeries(time_mark, test_params, quadratures)
    return time_series