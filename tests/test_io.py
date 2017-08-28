from unittest import TestCase, main
from iisr.io import read, refined2raw_parameters, write
from iisr.representation import Parameters, SignalTimeSeries, SignalBlock
from datetime import datetime, timedelta

import numpy as np
from nose.tools import *


def setup():
    """Module level setup"""


def teardown():
    """Module level teardown"""


class TestWriteRead(TestCase):
    def test_write_read(self):
        test_file_path = r'test_iisr_file.ISE'
        high_level_parameters = Parameters()
        high_level_parameters.n_samples = 2048
        high_level_parameters.frequency_MHz = 155.5
        high_level_parameters.pulse_type = 'long'
        high_level_parameters.pulse_length_us = 700
        high_level_parameters.sampling_frequency = 1000
        high_level_parameters.channel = 0
        high_level_parameters.phase_code = 0
        high_level_parameters.total_delay = 1000

        time_mark = datetime(2015, 6, 6, 10, 15, 37, 43752)

        raw_parameters, data_block_len = refined2raw_parameters(time_mark,
                                                                high_level_parameters)
        test_quad_i = np.random.randint(-2 ** 15 + 1, 2 ** 15, data_block_len // 4)
        test_quad_q = np.random.randint(-2 ** 15 + 1, 2 ** 15, data_block_len // 4)
        test_quadratures = test_quad_i + 1j * test_quad_q

        write(test_file_path, raw_parameters, data_block_len, test_quadratures)

        series = next(read(test_file_path))
        print(series, data_block_len)
        self.assertEqual(test_quadratures.size, series.quadratures.size)
        np.testing.assert_equal(test_quadratures, series.quadratures)


class TestRead(TestCase):
    def test_read(self):
        path_to_test_data = r'D:\Projects\iisr\test_data' \
                            r'\20170603_0700_008_0000_002_003.ISE.gz'
        series = next(read(path_to_test_data))
        self.assertIsInstance(series.time_mark, datetime)
        self.assertIsInstance(series.parameters, Parameters)
        print(series)


if __name__ == '__main__':
    main()
