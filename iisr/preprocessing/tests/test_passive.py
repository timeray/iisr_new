from unittest import TestCase, main
from iisr.preprocessing.passive import PassiveSupervisor
import datetime as dt
from iisr.preprocessing.tests.utils import package_generator, get_test_param_list


class TestPassiveSupervisor(TestCase):
    def test_aggregator(self):
        freqs = [155.5, 156.7, 157.9]
        channels = [1, 3]
        n_samples = 64
        parameters_list = get_test_param_list(freqs=freqs, channels=channels, n_samples=n_samples)

        n_dtimes = 10
        dtimes = [dt.datetime(2042, 4, 2) + dt.timedelta(milliseconds=41 * i)
                  for i in range(n_dtimes)]

        packages = package_generator(dtimes, parameters_list)

        n_accumulation = 16
        n_fft = 8
        timeout = dt.timedelta(minutes=1)

        supervisor = PassiveSupervisor(n_accumulation, n_fft, timeout)

        for params, time_marks, batch_params, quadratures in supervisor.aggregator(packages):
            self.assertEqual(len(time_marks), len(freqs))
            self.assertEqual(len(time_marks[0]), n_accumulation)

            self.assertEqual(len(quadratures), len(channels))
            for ch, quads_list in quadratures.items():
                self.assertEqual(len(quads_list), len(freqs))
                for quads in quads_list:
                    self.assertEqual(quads.shape, (n_accumulation, n_fft))


if __name__ == '__main__':
    main()
