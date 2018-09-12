from unittest import TestCase, main
from iisr.representation import *
import json


class TestChannels(TestCase):
    def test(self):
        channel = Channel(1)
        self.assertEqual(channel.value, 1)
        self.assertEqual(channel.pulse_type, 'short')
        self.assertEqual(channel.band_type, 'wide')
        self.assertEqual(channel.horn, 'upper')

        with self.assertRaises(ValueError):
            Channel(-1)

        with self.assertRaises(ValueError):
            Channel(4)

    def test_hashable(self):
        channel1 = Channel(1)
        channel2 = Channel(2)
        channel3 = Channel(1)

        self.assertEqual(channel1, channel3)
        self.assertNotEqual(channel1, channel2)

        self.assertEqual(hash(channel1), hash(channel1))
        self.assertEqual(hash(channel1), hash(channel3))
        self.assertNotEqual(hash(channel1), hash(channel2))


class TestReprJSONEncoder(TestCase):
    def test(self):
        test_channel = Channel(1)
        encoded_channel = json.dumps(test_channel, cls=ReprJSONEncoder)
        decoded_channel = json.loads(encoded_channel, cls=ReprJSONDecoder)
        self.assertEqual(test_channel, decoded_channel)


if __name__ == '__main__':
    main()
