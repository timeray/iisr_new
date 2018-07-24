from unittest import TestCase, main
from iisr.representation import *
from datetime import datetime, timedelta
from . import tools


class TestParameters(TestCase):
    """"""
    def test_parameters(self):
        test_params = tools.get_test_parameters()
        self.assertIsNotNone(test_params.n_samples)
        self.assertIsNotNone(test_params.sampling_frequency)
        self.assertIsNotNone(test_params.channel)
        self.assertIsNotNone(test_params.pulse_type)
        self.assertIsNotNone(test_params.frequency)
        self.assertIsNotNone(test_params.total_delay)
        self.assertIsNotNone(test_params.phase_code)
        self.assertIsNotNone(test_params.pulse_length)
        print(test_params)

    def test_equality(self):
        test_params1 = tools.get_test_parameters(channel=0)
        test_params2 = tools.get_test_parameters(channel=1)
        test_params3 = tools.get_test_parameters(channel=0)
        self.assertEqual(test_params1, test_params3)
        self.assertNotEqual(test_params1, test_params2)


class TestSignalTimeSeries(TestCase):
    def test_initialization(self):
        test_time_series = tools.get_test_signal_time_series()
        self.assertIsNotNone(test_time_series.time_mark)
        self.assertIsNotNone(test_time_series.quadratures)
        self.assertIsNotNone(test_time_series.parameters)

    def test_print(self):
        test_time_series = tools.get_test_signal_time_series()
        test_time_series.parameters.rest_raw_parameters = {'version': 3}
        print(test_time_series)

    def test_size(self):
        test_time_series = tools.get_test_signal_time_series()
        expected_size = test_time_series.quadratures.size
        self.assertEqual(expected_size, test_time_series.size)

        # uninitialized options
        empty_params = Parameters()
        series = SignalTimeSeries(datetime(2000, 1, 1), empty_params, None)
        with self.assertRaises(ValueError):
            print(series.size)


class TestTimeSeriesPackage(TestCase):
    def test_init(self):
        test_time_series1 = tools.get_test_signal_time_series()
        test_time_series2 = tools.get_test_signal_time_series()
        package = TimeSeriesPackage(test_time_series1.time_mark,
                                    [test_time_series1, test_time_series2])

        for series, test_series in zip(package, [test_time_series1, test_time_series2]):
            self.assertEqual(series, test_series)

        self.assertEqual(package.time_series_list[0], test_time_series1)
        self.assertEqual(package.time_series_list[1], test_time_series2)
        self.assertEqual(package.time_mark, test_time_series2.time_mark)

    def test_incorrect_series(self):
        test_time_series1 = tools.get_test_signal_time_series()
        test_time_series2 = tools.get_test_signal_time_series()
        test_time_series2.time_mark = datetime(2016, 3, 4, 6, 2, 55, 3123)
        with self.assertRaises(ValueError):
            package = TimeSeriesPackage(test_time_series1.time_mark,
                                        [test_time_series1, test_time_series2])

    def test_empty(self):
        with self.assertRaises(ValueError):
            TimeSeriesPackage(datetime(2000, 1, 1), [])


if __name__ == '__main__':
    main()
