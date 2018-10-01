from unittest import TestCase, main
from iisr.preprocessing.passive import PassiveSupervisor
import datetime as dtime


class TestPassiveSupervisor(TestCase):
    def test_aggregator(self):
        packages = []
        n_accumulation = 5
        n_fft = 8
        timeout = dtime.timedelta(minutes=1)

        supervisor = PassiveSupervisor(n_accumulation, n_fft, timeout)

        for params, time_marks, batch_params, quadratures in supervisor.aggregator(packages):
            self.fail()


if __name__ == '__main__':
    main()
