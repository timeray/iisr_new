from unittest import main, TestCase
from iisr.preprocessing.active import *
from iisr.units import Frequency, TimeUnit
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
import numpy as np
import os


class TestCalcs(TestCase):
    def test_calc_delays(self):
        sampling_freq = Frequency(1, 'MHz')
        n_samples = 4096
        total_delay = TimeUnit(1e-3, 's')

        test_delays = np.linspace(1e-3, 1e-3 + 4095 * 1e-6, 4096)
        delays = calc_delays(sampling_freq, total_delay, n_samples)
        np.testing.assert_almost_equal(test_delays, delays['s'])

    def test_calc_distance(self):
        n = 10000
        delays = TimeUnit(np.linspace(1000, 5000, n, endpoint=False), 'ms')
        test_distances = delays['s'] * 3e8 / 2
        distances = calc_distance(delays)
        np.testing.assert_almost_equal(test_distances, distances['m'])


def get_active_params():
    sample_freq = Frequency(1, 'MHz')
    freq = Frequency(158, 'MHz')
    pulse_len = TimeUnit(700, 'us')
    delays = calc_delays(sample_freq, TimeUnit(3000, 'us'), 2048)
    channels = [2, 1]
    phase_codes = [4, 5]

    return ActiveParameters(sampling_frequency=sample_freq, delays=delays, channels=channels,
                            frequency=freq, pulse_length=pulse_len, phase_codes=phase_codes)


class TestActiveParameters(TestCase):
    def test_params(self):
        sample_freq = Frequency(1, 'MHz')
        freq = Frequency(158, 'MHz')
        pulse_len = TimeUnit(700, 'us')
        delays = calc_delays(sample_freq, TimeUnit(3000, 'us'), 2048)
        channels = [2, 1]
        phase_codes = [4, 5]

        params = ActiveParameters(sampling_frequency=sample_freq, delays=delays, channels=channels,
                                  frequency=freq, pulse_length=pulse_len, phase_codes=phase_codes)
        self.assertEqual(params.sampling_frequency, sample_freq)
        self.assertEqual(params.frequency, freq)
        self.assertEqual(params.pulse_length, pulse_len)
        self.assertIs(delays, params.delays)
        self.assertEqual(params.channels, [1, 2])
        self.assertEqual(params.phase_codes, [5, 4])


class TestActiveResult(TestCase):
    def test_basics(self):
        self.fail()

    def test_save(self):
        n = 15
        n_samples = 2048
        params = get_active_params()

        time_marks = [datetime(2015, 1, 1, 15) + timedelta(hours=h) for h in range(n)]
        power = {ch: np.random.randn(n, n_samples) for ch in params.channels}
        coherence = {}
        result = ActiveResult(parameters=params, time_marks=time_marks,
                              power=power, coherence=coherence)

        # Check parameters access (delegation to parameters object)
        self.assertEqual(result.channels, params.channels)
        self.assertEqual(result.sampling_frequency, params.sampling_frequency)
        self.assertEqual(result.frequency, params.frequency)
        self.assertEqual(result.__str__(), str(result))

        # Check that something is saved
        with NamedTemporaryFile('w') as file:
            result.save_txt(file=file)
            file.flush()
            self.assertGreater(os.path.getsize(file.name), 0)

    def test_save_load(self):
        n = 15
        n_samples = 2048
        params = get_active_params()

        time_marks = [datetime(2015, 1, 1, 15) + timedelta(hours=h) for h in range(n)]
        power = {ch: np.random.randn(n, n_samples) for ch in params.channels}
        coherence = {}
        result = ActiveResult(parameters=params, time_marks=time_marks,
                              power=power, coherence=coherence)

        with NamedTemporaryFile('r+') as file:
            result.save_txt(file=file)

            file.seek(0)
            load_results = ActiveResult.load_txt([file])

            self.assertEqual(result.parameters, load_results.parameters)
            self.assertEqual(result.time_marks, load_results.time_marks)
            for ch in params.channels:
                np.testing.assert_almost_equal(result.power[ch], load_results.power[ch])


if __name__ == '__main__':
    main()
