from datetime import datetime, timedelta
from unittest import TestCase, main

import os
import random
import itertools as it
import numpy as np
import contextlib

from iisr import io
from iisr.representation import *
from tests.tools import get_test_raw_parameters, get_test_parameters


def setup():
    """Module level setup"""


def teardown():
    """Module level teardown"""


TEST_FILE_NAME = 'test_iisr_file.ISE'


@contextlib.contextmanager
def make_test_file(n_unique_series=2, n_time_marks=2):
    file_path = TEST_FILE_NAME

    def gen_unique_parameters():
        freqs = [154., 155.5, 156.8, 158., 159.5]
        random.shuffle(freqs)
        channels = [0, 1, 2, 3]
        random.shuffle(channels)
        pulse_len = [200, 500, 700, 900]
        random.shuffle(pulse_len)
        phase_code = [0, 5]
        random.shuffle(phase_code)
        for freq, ch, length, code in it.product(freqs, channels, pulse_len, phase_code):
            pulse_type = CHANNELS_NUMBER_INFO[ch]['type']
            yield get_test_parameters(freq=freq, channel=ch, pulse_len=length,
                                      phase_code=code, pulse_type=pulse_type)

    generator = gen_unique_parameters()
    parameter_sets = []
    for i in range(n_unique_series):
        parameter_sets.append(generator.__next__())

    default_datetime = datetime(2015, 6, 7, 10, 55, 37, 437000)
    series_list = []
    for i in range(n_time_marks):
        time_mark = default_datetime + timedelta(milliseconds=41*i)
        for parameters in parameter_sets:
            n_samples = parameters.n_samples
            quad_i = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
            quad_q = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
            quadratures = quad_i + 1j * quad_q
            series = SignalTimeSeries(time_mark, parameters, quadratures)
            series_list.append(series)

    with io.open_data_file(file_path, 'w') as writer:
        for series in series_list:
            writer.write(series)

    yield file_path, series_list


class TestWriteRead(TestCase):
    def test_write_read_reciprocity(self):
        test_file_path = TEST_FILE_NAME

        # Create test parameters
        test_parameters = get_test_parameters()
        time_mark = datetime(2015, 6, 7, 10, 55, 37, 437000)
        n_samples = test_parameters.n_samples

        test_quad_i = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
        test_quad_q = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
        test_quadratures = test_quad_i + 1j * test_quad_q

        test_series = SignalTimeSeries(time_mark, test_parameters, test_quadratures)
        with io.open_data_file(test_file_path, 'w') as writer:
            writer.write(test_series)

        with io.open_data_file(test_file_path, 'r') as reader:
            series = next(reader.read_series())

        # Time
        self.assertEqual(time_mark, series.time_mark)

        # Quadratures
        self.assertEqual(test_quadratures.size, series.quadratures.size)
        np.testing.assert_equal(test_quadratures, series.quadratures)

        # Check test parameters against read parameters
        self.assertEqual(test_parameters.n_samples, series.parameters.n_samples)
        self.assertEqual(test_parameters.frequency_MHz, series.parameters.frequency_MHz)
        self.assertEqual(test_parameters.pulse_type, series.parameters.pulse_type)
        self.assertEqual(test_parameters.pulse_length_us,
                         series.parameters.pulse_length_us)
        self.assertEqual(test_parameters.sampling_frequency,
                         series.parameters.sampling_frequency)
        self.assertEqual(test_parameters.channel, series.parameters.channel)
        self.assertEqual(test_parameters.phase_code, series.parameters.phase_code)
        self.assertEqual(test_parameters.total_delay, series.parameters.total_delay)


class TestRead(TestCase):
    def test_read(self):
        with make_test_file() as (test_file_path, test_series):
            with io.open_data_file(test_file_path) as reader:
                series = next(reader.read_series())
            self.assertIsInstance(series.time_mark, datetime)
            self.assertIsInstance(series.parameters, Parameters)
            self.assertIsInstance(series.quadratures, np.ndarray)
            print(series)

    def test_read_by_series(self):
        with make_test_file() as (test_file_path, test_series_list):
            output = io.read_files_by_series(test_file_path)
            for series, test_series in zip(output, test_series_list):
                self.assertIsInstance(series, SignalTimeSeries)
                self.assertEqual(series.time_mark, test_series.time_mark)
                self.assertEqual(series.parameters.__dict__,
                                 test_series.parameters.__dict__)
                self.assertEqual(series.parameters, test_series.parameters)
                np.testing.assert_array_equal(series.quadratures, test_series.quadratures)

    def test_read_by_blocks(self):
        with make_test_file() as (test_file_path, test_series):
            output = io.read_files_by_packages(test_file_path)
            package = next(output)
            self.assertIsInstance(package, TimeSeriesPackage)
            for series in package:
                self.assertIsInstance(series, SignalTimeSeries)
                self.assertEqual(package.time_mark, series.time_mark)


class TestParametersTransformation(TestCase):
    def test_reciprocity(self):
        test_byte_length = 4096
        test_raw_parameters = get_test_raw_parameters()

        # Need to copy parameters, because transformation change input dictionary
        result = io.raw2refined_parameters(test_raw_parameters.copy(), test_byte_length)
        raw_parameters, byte_length = io.refined2raw_parameters(*result)
        self.assertEqual(test_byte_length, byte_length)
        self.assertEqual(test_raw_parameters, raw_parameters)


if __name__ == '__main__':
    main()
