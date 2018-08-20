from unittest import TestCase, main
from iisr import units
import numpy as np
import json


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

    def test_registry(self):
        self.assertIn('Frequency', units._unit_types_registry)
        self.assertIs(units._unit_types_registry['Frequency'], units.Frequency)
        self.assertIn('TimeUnit', units._unit_types_registry)
        self.assertIn('Distance', units._unit_types_registry)

    def test_json_serialization(self):
        test_freq = units.Frequency(1.5, 'kHz')
        freq = json.loads(
            json.dumps(test_freq, cls=units.UnitsJSONEncoder),
            cls=units.UnitJSONDecoder,
        )
        self.assertEqual(freq, test_freq)

        # Intended version should work too
        test_dist = units.Distance(1.5, 'km')
        dist = json.loads(
            json.dumps(test_dist, indent=True, cls=units.UnitsJSONEncoder),
            cls=units.UnitJSONDecoder,
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
            cls=units.UnitJSONDecoder
        )
        self.assertEqual(arr_size1_time, decoded_time)


if __name__ == '__main__':
    main()
