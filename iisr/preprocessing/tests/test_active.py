from unittest import main, TestCase
from iisr.preprocessing.active import *
from iisr.representation import ExperimentGlobalParameters, SeriesParameters, Channel
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


def get_global_params(n_samples=2048, sampling_frequency=Frequency(1, 'MHz'),
                      total_delay=TimeUnit(3000, 'us')):
    return ExperimentGlobalParameters(sampling_frequency, n_samples, total_delay)


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

            with self.assertRaises(NotImplementedError):
                load_results = ActiveResult.load_txt([file])

        # Empty results
        empty_results = ActiveResult(parameters=params, time_marks=[], power=None, coherence=None)
        self.assertEqual(params, empty_results.parameters)
        np.testing.assert_equal(empty_results.time_marks, np.array([], dtype=datetime))
        self.assertEqual(empty_results.power, None)
        self.assertEqual(empty_results.coherence, None)


class TestNarrowbandActiveHander(TestCase):
    def test_processing(self):
        handler = LongPulseActiveHandler(eval_power=True, eval_coherence=True)

        global_params = get_global_params()
        freq = Frequency(155.5, 'MHz')
        pulse_len = TimeUnit(700, 'us')
        pulse_type = 'long'
        channels = [Channel(0), Channel(2)]
        phase_code = 0

        params = [
            SeriesParameters(global_params, channel=ch, frequency=freq, pulse_length=pulse_len,
                             phase_code=phase_code, pulse_type=pulse_type)
            for ch in channels
        ]

        n_series = 4
        n_acc = 3
        n_samples = global_params.n_samples
        ref_dt = datetime(2015, 1, 2, 14)
        time_marks = []

        for i in range(n_series):
            time_marks.append(
                np.array([ref_dt + timedelta(minutes=m, days=i) for m in range(n_acc)])
            )

        quadratures = {ch: [] for ch in channels}
        for ch in channels:
            for i in range(n_series):
                quadratures[ch].append(
                    np.random.randn(n_acc, global_params.n_samples)
                    + 1j * np.random.randn(n_acc, global_params.n_samples),
                )

        # Regular operation: 2 channels, calculate coherence and power
        for i in range(n_series):
            handler.process(params[0], time_marks[i], quadratures[channels[0]][i])
            handler.process(params[1], time_marks[i], quadratures[channels[1]][i])

        results = handler.finish()

        test_params = ActiveParameters(global_params, channels, pulse_type, frequency=freq,
                                       pulse_length=pulse_len, phase_code=phase_code)
        self.assertIsInstance(results, ActiveResult)
        self.assertEqual(results.parameters, test_params)
        self.assertEqual(len(results.time_marks), n_series)

        self.assertNotEqual(results.power, None)
        self.assertEqual(len(results.power), len(channels))

        for ch in channels:
            self.assertIn(ch, results.power)
            self.assertIsInstance(results.power[ch], np.ndarray)
            self.assertEqual(results.power[ch].shape, (n_series, n_samples))

        self.assertNotEqual(results.coherence, None)
        self.assertIsInstance(results.coherence, np.ndarray)
        self.assertTrue(np.iscomplexobj(results.coherence))
        self.assertEqual(results.coherence.shape, (n_series, n_samples))

        # Regular operation: 1 channel, calculate power
        handler = LongPulseActiveHandler(eval_power=True, eval_coherence=False)
        for i in range(n_series):
            handler.process(params[0], time_marks[i], quadratures[channels[0]][i])
        results = handler.finish()
        self.assertEqual(results.coherence, None)
        self.assertNotEqual(results.power, None)

        # Irregular operation: 1 channel, calculate power and coherence
        handler = LongPulseActiveHandler(eval_power=True, eval_coherence=True)
        handler.process(params[0], time_marks[0], quadratures[channels[0]][0])
        with self.assertRaises(EvalCoherenceError):
            handler.process(params[0], time_marks[1], quadratures[channels[0]][1])

        # Irregular operation: 4 channels, calculate power and coherence


if __name__ == '__main__':
    main()
