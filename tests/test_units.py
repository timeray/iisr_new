from unittest import TestCase, main
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

        freq_unit = units.Frequency(test_freq, 'kHz')
        self.assertEqual(freq_unit['Hz'], test_freq * 1000.)
        self.assertEqual(freq_unit['kHz'], test_freq)
        self.assertEqual(freq_unit['MHz'], test_freq / 1000.)

        with self.assertRaises(KeyError):
            units.Frequency(test_freq, 'GHz')

        with self.assertRaises(TypeError):
            print(freq_unit * 2)


if __name__ == '__main__':
    main()
