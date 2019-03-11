import os
from datetime import datetime, timedelta
from itertools import cycle
from tempfile import NamedTemporaryFile
from unittest import main, TestCase

import numpy as np

from iisr.iisr_io import ExperimentParameters
from iisr.preprocessing.active import *
from iisr.preprocessing.active import ActiveHandler, ActiveBatch, ActiveSupervisor
from iisr.preprocessing.tests.utils import get_test_param_list, package_generator
from iisr.representation import Channel
from iisr.units import Frequency, TimeUnit


def get_global_params(n_samples=2048, sampling_frequency=Frequency(1, 'MHz'),
                      total_delay=TimeUnit(3000, 'us')):
    return ExperimentParameters(sampling_frequency, n_samples, total_delay)


def get_active_params(n_samples=2048):
    global_parameters = get_global_params(n_samples=n_samples)

    params_dict = {
        'global_parameters': global_parameters,
        'pulse_type': 'short',
        'pulse_length': TimeUnit(700, 'us'),
        'frequency': Frequency(158, 'MHz'),
        'channels': [Channel(1), Channel(3)],
        'phase_code': 5,
    }
    return params_dict, ActiveParameters(**params_dict)


def get_active_results():
    n = 50
    params_dict, params = get_active_params(n_samples=512)

    time_marks = [datetime(2015, 1, 1, 15) + timedelta(hours=h) for h in range(n)]

    power = {}
    for ch in params.channels:
        power[ch] = np.random.randn(n, params.n_samples)

    coherence = np.random.randn(n, params.n_samples) + 1j * np.random.randn(n, params.n_samples)

    result = ActiveResult(parameters=params, time_marks=time_marks,
                          power=power, coherence=coherence)
    return params, time_marks, power, coherence, result


class TestSquareBarker(TestCase):
    def test(self):
        length = 5
        barker = 5
        test_pulse = np.array([1., 1., 1., -1., 1.])
        pulse = square_barker(length, barker)
        self.assertIsInstance(pulse, np.ndarray)
        self.assertEqual(len(pulse), length)
        np.testing.assert_almost_equal(test_pulse, pulse)

        length = 20
        barker = 5
        pulse = square_barker(length, barker)
        self.assertIsInstance(pulse, np.ndarray)
        self.assertEqual(len(pulse), length)
        np.testing.assert_almost_equal(np.repeat(test_pulse, length // barker), pulse)

        length = 7
        barker = 5
        test_pulse = np.array([1., 1., 1., 1., -1., 1., 1.])
        pulse = square_barker(length, barker)
        self.assertIsInstance(pulse, np.ndarray)
        self.assertEqual(len(pulse), length)
        np.testing.assert_almost_equal(test_pulse, pulse)

        length = 7
        barker = 3
        test_pulse = np.array([1., 1., 1., 1., -1., -1., -1.])
        pulse = square_barker(length, barker)
        self.assertIsInstance(pulse, np.ndarray)
        self.assertEqual(len(pulse), length)
        np.testing.assert_almost_equal(test_pulse, pulse)

        length = 12
        barker = 5
        test_pulse = np.array([1., 1., 1., 1., 1., 1., 1., -1., -1., 1., 1., 1.])
        pulse = square_barker(length, barker)
        self.assertIsInstance(pulse, np.ndarray)
        self.assertEqual(len(pulse), length)
        np.testing.assert_almost_equal(test_pulse, pulse)

        length = 5
        barker = 11
        test_pulse = np.array([1., 1., -1., -1., -1.])
        pulse = square_barker(length, barker)
        self.assertIsInstance(pulse, np.ndarray)
        self.assertEqual(len(pulse), length)
        np.testing.assert_almost_equal(test_pulse, pulse)


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
        distances = delays2distance(delays)
        np.testing.assert_almost_equal(test_distances, distances['m'])


class TestActiveParameters(TestCase):
    def test_params(self):
        param_dict, params = get_active_params()
        self.assertEqual(params.sampling_frequency,
                         param_dict['global_parameters'].sampling_frequency)
        self.assertEqual(params.total_delay, param_dict['global_parameters'].total_delay)
        self.assertEqual(params.n_samples, param_dict['global_parameters'].n_samples)
        self.assertEqual(params.frequency, param_dict['frequency'])
        self.assertEqual(params.pulse_length, param_dict['pulse_length'])
        self.assertEqual(params.channels, params.channels)
        self.assertEqual(params.phase_code, params.phase_code)

    def test_save_load(self):
        params_dict, params = get_active_params()
        with NamedTemporaryFile('r+') as file:
            params.save_txt(file)
            file.seek(0)
            load_params = ActiveParameters.load_txt(file)
            for name in ActiveParameters.params_to_save:
                self.assertEqual(getattr(params, name), getattr(load_params, name))


class TestActiveResult(TestCase):
    def test_basics(self):
        params, time_marks, power, coherence, result = get_active_results()

        # Parameters attributes are accessible from result instance
        self.assertEqual(params.channels, result.channels)
        self.assertEqual(params.frequency, result.frequency)
        self.assertEqual(result.sampling_frequency, params.sampling_frequency)
        self.assertEqual(result.__str__(), str(result))

    def test_save(self):
        params, time_marks, power, coherence, result = get_active_results()

        # Check that something is saved
        with NamedTemporaryFile('w') as file:
            result.save_txt(file=file)
            file.flush()
            self.assertGreater(os.path.getsize(file.name), 0)

    def test_save_load(self):
        params, time_marks, power, coherence, result = get_active_results()
        precision = 5

        # Regular write-read
        with NamedTemporaryFile('r+') as file:
            result.save_txt(file=file, precision=precision)

            file.seek(0)
            load_results = ActiveResult.load_txt([file])

            self.assertEqual(result.parameters, load_results.parameters)
            np.testing.assert_equal(result.time_marks, load_results.time_marks)
            for par in result.power:
                np.testing.assert_almost_equal(result.power[par], load_results.power[par],
                                               decimal=precision)

        # With save_params=False, loading is not implemented
        with NamedTemporaryFile('r+') as file:
            result.save_txt(file=file, save_params=False)

            file.seek(0)

            self.assertTrue(file.readline().startswith('Date'))

            file.seek(0)

            with self.assertWarns(UserWarning), self.assertRaises(NotImplementedError):
                ActiveResult.load_txt([file])

        # Empty results
        empty_results = ActiveResult(parameters=params, time_marks=[], power=None, coherence=None)
        self.assertEqual(params, empty_results.parameters)
        np.testing.assert_equal(empty_results.time_marks, np.array([], dtype=datetime))
        self.assertIsNone(empty_results.power)
        self.assertIsNone(empty_results.coherence)


class TestActiveHandler(TestCase):
    def test_processing(self):
        global_params = get_global_params()
        freq = Frequency(155.5, 'MHz')
        pulse_len = TimeUnit(700, 'us')
        pulse_type = 'long'
        channels = [Channel(0), Channel(2)]
        phase_code = 0

        params = ActiveParameters(global_params, channels, pulse_type, frequency=freq,
                                  pulse_length=pulse_len, phase_code=phase_code)

        handler = ActiveHandler(active_parameters=params, eval_power=True, eval_coherence=True)

        n_series = 4
        n_acc = 3
        n_samples = global_params.n_samples
        ref_dt = datetime(2015, 1, 2, 14)
        time_marks = []
        experiment_quadratures = []

        for i in range(n_series):
            time_marks.append(
                np.array([ref_dt + timedelta(minutes=m, days=i) for m in range(n_acc)])
            )

            quadratures = {}
            for ch in channels:
                quadratures[ch] = \
                    np.random.randn(n_acc, global_params.n_samples) \
                    + 1j * np.random.randn(n_acc, global_params.n_samples)
            experiment_quadratures.append(quadratures)

        # 2 channels, calculate coherence and power
        for i in range(n_series):
            handler.handle(ActiveBatch(time_marks[i], experiment_quadratures[i]))

        results = handler.finish()

        self.assertIsInstance(results, ActiveResult)
        self.assertEqual(results.parameters, params)
        self.assertEqual(len(results.time_marks), n_series)

        self.assertIsNotNone(results.power)
        self.assertEqual(len(results.power), len(channels))

        for ch in channels:
            self.assertIn(ch, results.power)
            self.assertIsInstance(results.power[ch], np.ndarray)
            self.assertEqual(results.power[ch].shape, (n_series, n_samples))

        self.assertIsNotNone(results.coherence)
        self.assertIsInstance(results.coherence, np.ndarray)
        self.assertTrue(np.iscomplexobj(results.coherence))
        self.assertEqual(results.coherence.shape, (n_series, n_samples))

        # 1 channel, calculate power
        channels = [Channel(0)]
        # Create new, single channel quadratures
        experiment_quadratures = [{ch: q} for quads in experiment_quadratures
                                  for ch, q in quads.items()
                                  if ch == channels[0]]
        params = ActiveParameters(global_params, channels, pulse_type, frequency=freq,
                                  pulse_length=pulse_len, phase_code=phase_code)

        handler = ActiveHandler(params, eval_power=True, eval_coherence=False)
        for i in range(n_series):
            handler.handle(ActiveBatch(time_marks[i], experiment_quadratures[i]))
        results = handler.finish()
        self.assertIsNone(results.coherence)
        self.assertIsNotNone(results.power)

        # 1 channel, calculate coherence raises error
        handler = ActiveHandler(params, eval_power=True, eval_coherence=True)
        with self.assertRaises(EvalCoherenceError):
            handler.handle(ActiveBatch(time_marks[0], experiment_quadratures[0]))


class TestSupervisor(TestCase):
    def test_aggregator(self):
        n_dtimes = 10
        n_accumulation = 3
        dtimes = [datetime(2014, 3, 4) + timedelta(milliseconds=41*i) for i in range(n_dtimes)]
        channels = 0, 1, 2, 3
        freqs = (155.5, 159.5)
        n_samples = 64
        param_list = get_test_param_list(freqs=freqs, channels=channels, n_samples=n_samples)
        packages = package_generator(dtimes, param_list)

        supervisor = ActiveSupervisor(n_accumulation, timeout=timedelta(minutes=5))

        expected_channels = [(Channel(0), Channel(2)), (Channel(1), Channel(3))]
        batch_counter = 0
        for active_params, batch in supervisor.aggregator(packages):
            self.assertIsInstance(active_params, ActiveParameters)
            self.assertIn(active_params.channels, expected_channels)
            self.assertIn(active_params.frequency['MHz'], freqs)
            self.assertEqual(len(batch.time_marks), n_accumulation)
            for quads in batch.quadratures.values():
                self.assertEqual(quads.shape, (n_accumulation, n_samples))

            batch_counter += 1

        expected_n_batches = len(freqs) * len(expected_channels) * (n_dtimes // n_accumulation)
        self.assertEqual(batch_counter, expected_n_batches)

    def test_timeout(self):
        n_dtimes = 10
        n_accumulation = 3
        channels = 0, 1, 2, 3
        expected_channels = [(Channel(0), Channel(2)), (Channel(1), Channel(3))]
        freqs = (155.5, 159.5)
        n_samples = 64
        timeout = timedelta(minutes=5)
        param_list = get_test_param_list(freqs=freqs, channels=channels, n_samples=n_samples)

        def _run_supervisor(*, sv, timeout_idx, drop):
            dtimes = []
            for i in range(n_dtimes):
                if i <= timeout_idx:
                    dtimes.append(datetime(2014, 3, 4) + timedelta(milliseconds=41 * i))
                else:
                    dtimes.append(datetime(2014, 3, 4) + timedelta(milliseconds=41 * i) + timeout)

            batch_generator = sv.aggregator(
                package_generator(dtimes, param_list),
                drop_timeout_series=drop
            )

            n_batches = 0
            for params, batch in batch_generator:
                self.assertIsInstance(params, ActiveParameters)
                self.assertIn(params.channels, expected_channels)
                self.assertEqual(len(batch.time_marks), n_accumulation)
                for quads in batch.quadratures.values():
                    self.assertEqual(quads.shape, (n_accumulation, n_samples))
                n_batches += 1
            return n_batches

        supervisor = ActiveSupervisor(n_accumulation, timeout)

        # Drop single series and it is still enough to get 3 series for each unique parameters
        n_series = _run_supervisor(sv=supervisor, timeout_idx=3, drop=True)
        expected_n_batches = ((n_dtimes - 1) // n_accumulation) \
                             * len(expected_channels) * len(freqs)
        self.assertEqual(n_series, expected_n_batches)

        # Drop two series and rest series can form only 2 series for each unique parameters
        n_series = _run_supervisor(sv=supervisor, timeout_idx=4, drop=True)
        expected_n_batches = ((n_dtimes - 2) // n_accumulation) \
                             * len(expected_channels) * len(freqs)
        self.assertEqual(n_series, expected_n_batches)


if __name__ == '__main__':
    main()
