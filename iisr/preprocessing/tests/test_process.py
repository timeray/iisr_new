from datetime import datetime, timedelta
from itertools import product, cycle
from unittest import TestCase, main

from typing import List

from iisr import io
from iisr.preprocessing.run import aggregate_packages
from iisr.tests.test_io import get_test_signal_time_series, get_test_parameters


def package_generator(dtimes: List[datetime], params_list: List[io.SeriesParameters]):
    for dtime in dtimes:
        series_list = []
        for params in params_list:
            series = get_test_signal_time_series(dtime, test_params=params)
            series_list.append(series)

        yield io.TimeSeriesPackage(dtime, series_list)


class TestAggregatePackages(TestCase):
    @staticmethod
    def get_test_param_list():
        param_list = []
        for fr, ch in product([155.5, 158.0], [0, 1, 2, 3]):
            n_samples = 128 if ch in [0, 2] else 258
            params = {'freq': fr, 'channel': ch, 'n_samples': n_samples}
            param_list.append(get_test_parameters(**params))
        return param_list

    def test_regular(self):
        n_dtimes = 10
        n_accumulation = 3
        dtimes = [datetime(2014, 3, 4) + timedelta(milliseconds=41*i) for i in range(n_dtimes)]
        param_list = self.get_test_param_list()

        agg_series = aggregate_packages(package_generator(dtimes, param_list),
                                        n_accumulation=n_accumulation)

        n_series = 0
        for test_params, (params, time_marks, quads) in zip(cycle(param_list), agg_series):
            self.assertEqual(len(time_marks), n_accumulation)
            self.assertEqual(quads.shape, (n_accumulation, test_params.n_samples))
            self.assertEqual(params, test_params)
            n_series += 1

        self.assertEqual(n_series, (n_dtimes // n_accumulation) * len(param_list))

    def test_timeout(self):
        n_dtimes = 10
        n_accumulation = 3
        timeout = timedelta(minutes=5)
        param_list = self.get_test_param_list()

        def _calc_n_series(*, insert_idx, drop):
            dtimes = []
            for i in range(n_dtimes):
                if i <= insert_idx:
                    dtimes.append(datetime(2014, 3, 4) + timedelta(milliseconds=41 * i))
                else:
                    dtimes.append(datetime(2014, 3, 4) + timedelta(milliseconds=41 * i) + timeout)

            agg_series = aggregate_packages(package_generator(dtimes, param_list),
                                            n_accumulation=n_accumulation,
                                            timeout=timeout,
                                            drop_timeout_series=drop)

            n_series = 0
            for test_params, (params, time_marks, quads) in zip(cycle(param_list), agg_series):
                self.assertEqual(len(time_marks), n_accumulation)
                self.assertEqual(quads.shape, (n_accumulation, test_params.n_samples))
                self.assertEqual(params, test_params)
                n_series += 1
            return n_series

        # Drop 1 series and it is still enough to get 3 series for each unique parameters
        n_series = _calc_n_series(insert_idx=3, drop=True)
        self.assertEqual(n_series, ((n_dtimes - 1) // n_accumulation) * len(param_list))

        # Drop 2 series and rest series can form only 2 series for each unique parameters
        n_series = _calc_n_series(insert_idx=4, drop=True)
        self.assertEqual(n_series, ((n_dtimes - 2) // n_accumulation) * len(param_list))


if __name__ == '__main__':
    main()