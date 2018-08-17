from unittest import TestCase, main
from iisr import units
import numpy as np


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


if __name__ == '__main__':
    main()
