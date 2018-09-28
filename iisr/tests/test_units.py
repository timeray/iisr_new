import json
from unittest import TestCase, main

import numpy as np

from iisr import units


def setup():
    """Module level setup"""


def teardown():
    """Module level teardown"""


class TestUnits(TestCase):
    def test_frequency(self):
        test_freq = 1000

        freq_unit = units.Frequency(test_freq, 'Hz')
        self.assertEqual(freq_unit['Hz'], test_freq)
        self.assertEqual(freq_unit['kHz'], test_freq / 1000.)
        self.assertEqual(freq_unit['MHz'], test_freq / 1e6)
        self.assertEqual(freq_unit.size, 1)

        freq_unit = units.Frequency(test_freq, 'kHz')
        self.assertEqual(freq_unit['Hz'], test_freq * 1000.)
        self.assertEqual(freq_unit['kHz'], test_freq)
        self.assertEqual(freq_unit['MHz'], test_freq / 1000.)

        with self.assertRaises(KeyError):
            units.Frequency(test_freq, 'GHz')

        with self.assertRaises(TypeError):
            print(freq_unit * 2)

        # Test array
        size = 10
        test_freq = np.linspace(1000, 2000, 10)
        freq_unit = units.Frequency(test_freq, 'kHz')
        np.testing.assert_almost_equal(freq_unit['Hz'], test_freq * 1000.)
        np.testing.assert_almost_equal(freq_unit['kHz'], test_freq)
        np.testing.assert_almost_equal(freq_unit['MHz'], test_freq / 1000.)
        self.assertEqual(freq_unit.size, size)

        # Array should be copied, so inplace operation on original should not affect Units object
        test_freq *= 2
        self.assertTrue((freq_unit['kHz'] != test_freq).all())

        # Check floating point issue
        freq = units.TimeUnit(700, 'us')
        val1 = freq['ms']
        freq['us']
        val2 = freq['ms']
        self.assertEqual(val1, val2)

        # Same for arrays
        freq = units.TimeUnit(np.array([700, 900]), 'us')
        val1 = freq['ms']
        freq['us']
        val2 = freq['ms']
        np.testing.assert_equal(val1, val2)

    def test_iteration(self):
        # Iteration over 1-size array raises error
        with self.assertRaises(ValueError):
            next(iter(units.Frequency(1, 'kHz')))

        # It is supposed that iteration over array return Units objects with corresponding values
        test_arr = np.arange(10, dtype=float)
        freqs = units.Frequency(test_arr, 'kHz')
        for i, freq in enumerate(freqs):
            self.assertEqual(units.Frequency(test_arr[i], 'kHz'), freq)

        # If original object was changed during iteration (e.g. when someone access another items),
        # iteration should be over original array
        for i, freq in enumerate(freqs):
            self.assertEqual(units.Frequency(test_arr[i], 'kHz'), freq)
            freqs.__getitem__('MHz')

    def test_comparison(self):
        freq1 = units.Frequency(1000, 'kHz')
        freq2 = units.Frequency(1000, 'kHz')
        self.assertEqual(freq1, freq2)

        freq2 = units.Frequency(1, 'MHz')
        self.assertEqual(freq1, freq2)

        freq2 = units.Frequency(2000, 'kHz')
        self.assertNotEqual(freq1, freq2)

        t = units.TimeUnit(1, 's')

        with self.assertRaises(TypeError):
            freq1 == t

        # Array comparison is element-wise and binary mask is returned
        test_arr = np.arange(5, dtype=float)
        mask = np.array([1, 0, 1, 1, 0], dtype=bool)
        dist = units.Distance(test_arr, 'm')
        test_arr[mask] = 99  # original array should have been copied

        np.testing.assert_almost_equal(mask, dist['m'] != test_arr)

    def test_hashing(self):
        freq = units.Frequency(1000, 'kHz')
        self.assertEqual(hash(freq), hash(freq))
        self.assertEqual(hash(units.Frequency(1000, 'kHz')), hash(units.Frequency(1, 'MHz')))
        self.assertNotEqual(hash(units.Frequency(1000, 'kHz')), hash(units.Frequency(10, 'MHz')))

    def test_registry(self):
        self.assertIn('Frequency', units._unit_types_registry)
        self.assertIs(units._unit_types_registry['Frequency'], units.Frequency)
        self.assertIn('TimeUnit', units._unit_types_registry)
        self.assertIn('Distance', units._unit_types_registry)

    def test_json_serialization(self):
        test_freq = units.Frequency(1.5, 'kHz')
        freq = json.loads(
            json.dumps(test_freq, cls=units.UnitsJSONEncoder),
            cls=units.UnitsJSONDecoder,
        )
        self.assertEqual(freq, test_freq)

        # Intended version should work too
        test_dist = units.Distance(1.5, 'km')
        dist = json.loads(
            json.dumps(test_dist, indent=True, cls=units.UnitsJSONEncoder),
            cls=units.UnitsJSONDecoder,
        )
        self.assertEqual(dist, test_dist)

        # Array serialization (with size > 1) is not allowed
        arr_time = units.TimeUnit(np.array([1., 2., 3.]), 's')
        with self.assertRaises(TypeError):
            json.dumps(arr_time, cls=units.UnitsJSONEncoder)

        # Numpy arrays with size = 1 will be serialized and decoded later as floats
        arr_size1_time = units.TimeUnit(np.array([1.]), 's')
        decoded_time = json.loads(
            json.dumps(arr_size1_time, cls=units.UnitsJSONEncoder),
            cls=units.UnitsJSONDecoder
        )
        self.assertEqual(arr_size1_time, decoded_time)

    def test_arithmetic(self):
        freq1 = units.Frequency(1, 'Hz')
        freq2 = units.Frequency(1.002, 'kHz')
        freq3 = units.Frequency(np.array([1., 2.]), 'MHz')
        freq4 = units.Frequency(np.array([5., 6.]), 'kHz')

        self.assertAlmostEqual((freq1 + freq2)['Hz'], 1003.)
        self.assertAlmostEqual((freq1 - freq2)['Hz'], -1001.)

        np.testing.assert_almost_equal((freq3 + freq4)['kHz'], np.array([1005., 2006.]))
        np.testing.assert_almost_equal((freq3 - freq4)['kHz'], np.array([995., 1994.]))

        np.testing.assert_almost_equal((freq1 + freq4)['Hz'], np.array([5001., 6001.]))
        np.testing.assert_almost_equal((freq1 - freq4)['Hz'], np.array([-4999., -5999.]))


if __name__ == '__main__':
    main()
