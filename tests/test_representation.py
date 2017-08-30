from unittest import TestCase, main
from iisr.representation import *
from . import tools


class TestParameters(TestCase):
    """"""
    def test_parameters(self):
        test_params = tools.get_test_parameters()
        self.assertIsNotNone(test_params.n_samples)
        self.assertIsNotNone(test_params.sampling_frequency)
        self.assertIsNotNone(test_params.channel)
        self.assertIsNotNone(test_params.pulse_type)
        self.assertIsNotNone(test_params.frequency_MHz)
        self.assertIsNotNone(test_params.total_delay)
        self.assertIsNotNone(test_params.phase_code)
        self.assertIsNotNone(test_params.pulse_length_us)
        print(test_params)


class TestSignalTimeSeries(TestCase):
    def test_initialization(self):
        test_time_series = tools.get_test_signal_time_series()
        self.assertIsNotNone(test_time_series.time_mark)
        self.assertIsNotNone(test_time_series.quadratures)
        self.assertIsNotNone(test_time_series.parameters)

    def test_print(self):
        test_time_series = tools.get_test_signal_time_series()
        print(test_time_series)

    def test_size(self):
        test_time_series = tools.get_test_signal_time_series()
        expected_size = test_time_series.quadratures.size
        self.assertEqual(expected_size, test_time_series.size)

if __name__ == '__main__':
    main()
