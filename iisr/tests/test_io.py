import contextlib
import itertools as it
import random
import tempfile

from iisr.representation import CHANNELS_INFO, Channel
from iisr.units import Frequency, TimeUnit
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase, main, mock
from iisr import io
import numpy as np


DUMMY_FILE_NAME = '20150202_0000_000_0000_002_000.ISE'
TEST_REAL_FILEPATH = Path(__file__).parent / '20150606_0000_000_0000_002_000.ISE'
DEFAULT_DATETIME = datetime(2015, 6, 7, 10, 55, 37, 437000)


def get_file_info(field1=0, field2=0, version=0, field4=0):
    return io.FileInfo(field1, field2, version, field4)


def get_test_raw_parameters(freq=155000, pulse_len=700, stel='st1', channel=0, year=2015, month=6,
                            day=14, hour=5, minute=55, second=59, millisecond=850):
    pulse_type = CHANNELS_INFO[channel]['type']
    test_raw_parameters = {
        'reserved': 0,
        'mode': 1,
        'step': 2,
        'number_all': 2048,
        'number_after': 1024,
        'first_delay': 1000,
        'channel': channel,
        'date_year': year,
        'date_mon_day': (month << 8) + day,
        'time_h_m': (minute << 8) + hour,
        'time_sec': second,
        'time_msec': millisecond,
        '{}_{}_fr_lo'.format(stel, pulse_type): freq & 0xFFFF,
        '{}_{}_fr_hi'.format(stel, pulse_type): freq >> 16,
        '{}_{}_len'.format(stel, pulse_type): pulse_len,
        'phase_code': 0,
        'average': 32,
        'offset_st1': 80,
        'sample_freq': 1000,
        'version': 3
    }
    # Encode
    test_raw_parameters = {io.RAW_NAME_TO_CODE[name]: value
                           for name, value in test_raw_parameters.items()}
    return test_raw_parameters


def get_test_parameters(n_samples=2048, freq=155.5, pulse_len=700,
                        sampling_freq=1., channel=0, phase_code=0, total_delay=1000,
                        antenna_end='st1', file_info=None) -> io.SeriesParameters:
    if file_info is None:
        file_info = get_file_info()
    global_params = io.ExperimentParameters(Frequency(sampling_freq, 'MHz'), n_samples, total_delay)
    test_parameters = io.SeriesParameters(
        file_info, global_params, Channel(channel), Frequency(freq, 'MHz'),
        TimeUnit(pulse_len, 'us'), phase_code, antenna_end=antenna_end
    )
    return test_parameters


def get_test_signal_time_series(time_mark=datetime(2015, 6, 7, 8, 9, 10, 11), test_params=None,
                                **param_kwargs) -> io.SignalTimeSeries:
    if test_params is None:
        test_params = get_test_parameters(**param_kwargs)

    quad_i = np.random.randint(-100, 100, test_params.n_samples)
    quad_q = np.random.randint(-100, 100, test_params.n_samples)
    quadratures = quad_i + 1j * quad_q
    time_series = io.SignalTimeSeries(time_mark, test_params, quadratures)
    return time_series


@contextlib.contextmanager
def make_random_test_file(n_unique_series=2, n_time_marks=2, time_step_sec=1):
    """Create test_file with random series"""
    def gen_unique_parameters():
        freqs = [154., 155.5, 156.8, 158., 159.5]
        random.shuffle(freqs)
        channels = [0, 1, 2, 3]
        random.shuffle(channels)
        pulse_len = [200, 500, 700, 900]
        random.shuffle(pulse_len)
        phase_code = [0, 5]
        random.shuffle(phase_code)
        for freq, ch, length, code in it.product(freqs, channels, pulse_len, phase_code):
            yield get_test_parameters(freq=freq, channel=ch, pulse_len=length, phase_code=code)

    generator = gen_unique_parameters()
    parameter_sets = []
    for i in range(n_unique_series):
        parameter_sets.append(generator.__next__())

    series_list = []
    for i in range(n_time_marks):
        time_mark = DEFAULT_DATETIME + timedelta(seconds=time_step_sec * i)
        for parameters in parameter_sets:
            n_samples = parameters.n_samples
            quad_i = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
            quad_q = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
            quadratures = quad_i + 1j * quad_q
            series = io.SignalTimeSeries(time_mark, parameters, quadratures)
            series_list.append(series)

    with tempfile.TemporaryDirectory() as temp_dirname:
        file_path = Path(temp_dirname) / DUMMY_FILE_NAME
        with io.open_data_file(file_path, 'w') as writer:
            for series in series_list:
                writer.write(series)

        yield file_path, series_list


class TestSeriesParameters(TestCase):
    def test_parameters(self):
        test_params = get_test_parameters()
        self.assertIsNotNone(test_params.n_samples)
        self.assertIsNotNone(test_params.sampling_frequency)
        self.assertIsNotNone(test_params.channel)
        self.assertIsNotNone(test_params.pulse_type)
        self.assertIsNotNone(test_params.frequency)
        self.assertIsNotNone(test_params.total_delay)
        self.assertIsNotNone(test_params.phase_code)
        self.assertIsNotNone(test_params.pulse_length)

    def test_equality(self):
        test_params1 = get_test_parameters(channel=0)
        test_params2 = get_test_parameters(channel=1)
        test_params3 = get_test_parameters(channel=0)
        self.assertEqual(test_params1, test_params3)
        self.assertNotEqual(test_params1, test_params2)

    def test_hashable(self):
        test_params1 = get_test_parameters()
        test_params2 = get_test_parameters(freq=155.7)
        test_params3 = get_test_parameters()
        self.assertEqual(hash(test_params1), hash(test_params1))
        self.assertNotEqual(hash(test_params1), hash(test_params2))
        self.assertNotEqual(hash(test_params2), hash(test_params3))
        self.assertEqual(hash(test_params1), hash(test_params3))

        # Changing attribute changes hash
        test_params1.frequency = Frequency(155.7, 'MHz')
        self.assertEqual(hash(test_params1), hash(test_params2))
        self.assertNotEqual(hash(test_params1), hash(test_params3))


class TestSignalTimeSeries(TestCase):
    def test_initialization(self):
        test_time_series = get_test_signal_time_series()
        self.assertIsNotNone(test_time_series.time_mark)
        self.assertIsNotNone(test_time_series.quadratures)
        self.assertIsNotNone(test_time_series.parameters)

    def test_size(self):
        test_time_series = get_test_signal_time_series()
        expected_size = test_time_series.quadratures.size
        self.assertEqual(expected_size, test_time_series.size)


class TestTimeSeriesPackage(TestCase):
    def test_init(self):
        test_time_series1 = get_test_signal_time_series()
        test_time_series2 = get_test_signal_time_series()
        package = io.TimeSeriesPackage(test_time_series1.time_mark,
                                       [test_time_series1, test_time_series2])

        for series, test_series in zip(package, [test_time_series1, test_time_series2]):
            self.assertEqual(series, test_series)

        self.assertEqual(package.time_series_list[0], test_time_series1)
        self.assertEqual(package.time_series_list[1], test_time_series2)
        self.assertEqual(package.time_mark, test_time_series2.time_mark)

    def test_incorrect_series(self):
        test_time_series1 = get_test_signal_time_series()
        test_time_series2 = get_test_signal_time_series()
        test_time_series2.time_mark = datetime(2016, 3, 4, 6, 2, 55, 3123)
        with self.assertRaises(ValueError):
            io.TimeSeriesPackage(test_time_series1.time_mark,
                                 [test_time_series1, test_time_series2])

    def test_empty(self):
        with self.assertRaises(ValueError):
            io.TimeSeriesPackage(datetime(2000, 1, 1), [])


class TestGetAntennaEnd(TestCase):
    def test(self):
        for test_ant_end in ['st1', 'st2']:
            raw_params = {'{}_long_fr_lo'.format(test_ant_end): 1}
            ant_end = io._get_antenna_end(raw_params)
            self.assertEqual(test_ant_end, ant_end, test_ant_end)

        with self.assertRaises(ValueError):
            io._get_antenna_end({})

        raw_params = {'st1_long_fr_lo': 1, 'st2_long_fr_lo': 2}
        with self.assertRaises(ValueError):
            io._get_antenna_end(raw_params)


class TestParseFilename(TestCase):
    def test(self):
        dtime = datetime(2015, 2, 2, 14, 15)
        fields = [dtime.strftime('%Y%m%d_%H%M')]
        fields.extend(['0000', '001', '0002', '003'])
        test_file_info = io.FileInfo(0, 1, 2, 3)
        filename = '_'.join(fields)
        file_dtime, file_info = io.parse_filename(filename)
        self.assertEqual(dtime, file_dtime)
        self.assertEqual(test_file_info, file_info)

        wrong_filename = 'test_name.ise'
        with self.assertRaises(io.InvalidFilenameError):
            io.parse_filename(wrong_filename)


class TestParametersTransformation(TestCase):
    def test_millisecond_over_max(self):
        year = 2015
        month = 4
        day = 3
        hour = 5
        minute = 5
        second = 7
        millisecond = 1000

        test_dtime = datetime(year, month, day, hour, minute, second + 1)

        raw_params = get_test_raw_parameters(year=year, month=month, day=day, hour=hour,
                                             minute=minute, second=second, millisecond=millisecond)
        ref_params, _ = io._raw2refined_parameters(raw_params, 4096, get_file_info())
        self.assertEqual(test_dtime, ref_params)

    def test_reciprocity(self):
        test_byte_length = 4096
        test_raw_parameters = get_test_raw_parameters()
        test_file_info = get_file_info()

        # Need to copy options, because transformation change input dictionary
        result = io._raw2refined_parameters(test_raw_parameters.copy(), test_byte_length,
                                            test_file_info)
        raw_parameters, byte_length = io._refined2raw_parameters(*result)
        self.assertEqual(test_byte_length, byte_length)

        # Refined parameters contain only minimal amount of information from raw parameters
        # Some of parameters should be equal
        equal_params = ['number_all', 'first_delay', 'channel', 'sample_freq',
                        'date_year', 'date_mon_day', 'time_h_m', 'time_sec', 'time_msec',
                        'st1_long_fr_lo', 'st1_long_fr_hi', 'st1_long_len']
        for param in equal_params:
            code = io.RAW_NAME_TO_CODE[param]
            self.assertEqual(test_raw_parameters[code], raw_parameters[code])


class TestSeriesSelector(TestCase):
    def test(self):
        trivial_filter = io.SeriesSelector()

        valid_params = {
            'frequencies': [Frequency(155.5, 'MHz'), Frequency(159.5, 'MHz')],
            'channels': [Channel(0), Channel(2)],
            'pulse_lengths': [TimeUnit(200, 'us'), TimeUnit(700, 'us')],
            'pulse_types': None,
        }
        specific_filter = io.SeriesSelector(**valid_params)

        for fr, ch, len_ in it.product([155500, 159500], [0, 2], [200, 700]):
            raw_params = get_test_raw_parameters(freq=fr, pulse_len=len_, channel=ch)
            self.assertTrue(trivial_filter.validate_parameters(raw_params))
            self.assertTrue(specific_filter.validate_parameters(raw_params),
                            'fr={}, ch={}, len_={}'.format(fr, ch, len_))

        for fr, ch, len_ in it.product([154000, 159400], [1, 3], [900]):
            raw_params = get_test_raw_parameters(freq=fr, pulse_len=len_, channel=ch)
            self.assertTrue(trivial_filter.validate_parameters(raw_params))
            self.assertFalse(specific_filter.validate_parameters(raw_params),
                             'fr={}, ch={}, len_={}'.format(fr, ch, len_))

        # Channel and type
        valid_params = {
            'channels': [Channel(0), Channel(1)],
            'pulse_types': 'long',
        }
        specific_filter = io.SeriesSelector(**valid_params)
        for fr, ch, len_ in it.product([155500, 159500], [0, 2], [200, 700]):
            raw_params = get_test_raw_parameters(freq=fr, pulse_len=len_, channel=ch)

            self.assertEqual(ch == 0, specific_filter.validate_parameters(raw_params),
                             'fr={}, ch={}, len_={}'.format(fr, ch, len_))

    def test_time_validation(self):
        trivial_selector = io.SeriesSelector()
        start_selector = io.SeriesSelector(start_time=datetime(2015, 6, 6))
        stop_selector = io.SeriesSelector(stop_time=datetime(2015, 6, 8))
        start_stop_selector = io.SeriesSelector(start_time=datetime(2015, 6, 6),
                                                stop_time=datetime(2015, 6, 8))

        dtime = datetime(2015, 1, 1)
        self.assertTrue(trivial_selector.validate_time_mark(dtime))
        self.assertFalse(start_selector.validate_time_mark(dtime))
        self.assertTrue(stop_selector.validate_time_mark(dtime))
        self.assertFalse(start_stop_selector.validate_time_mark(dtime))

        dtime = datetime(2015, 6, 7)
        self.assertTrue(trivial_selector.validate_time_mark(dtime))
        self.assertTrue(start_selector.validate_time_mark(dtime))
        self.assertTrue(stop_selector.validate_time_mark(dtime))
        self.assertTrue(start_stop_selector.validate_time_mark(dtime))

        dtime = datetime(2015, 8, 6)
        self.assertTrue(trivial_selector.validate_time_mark(dtime))
        self.assertTrue(start_selector.validate_time_mark(dtime))
        self.assertFalse(stop_selector.validate_time_mark(dtime))
        self.assertFalse(start_stop_selector.validate_time_mark(dtime))


class TestRead(TestCase):
    def test_read(self):
        with make_random_test_file() as (test_file_path, test_series_list):
            with io.open_data_file(test_file_path) as reader:
                series = next(reader.read_series())
            self.assertIsInstance(series.time_mark, datetime)
            self.assertIsInstance(series.parameters, io.SeriesParameters)
            self.assertIsInstance(series.quadratures, np.ndarray)

            self.assertEqual(series.time_mark, test_series_list[0].time_mark)
            self.assertEqual(series.parameters, test_series_list[0].parameters)
            np.testing.assert_almost_equal(series.quadratures, test_series_list[0].quadratures)

    def test_read_with_selector(self):
        valid_params = {
            'frequencies': [Frequency(155.5, 'MHz')],
            'channels': [Channel(0), Channel(2)],
            'pulse_lengths': [TimeUnit(200, 'us')],
            'pulse_types': ['long'],
        }
        filter_ = io.SeriesSelector(**valid_params)

        test_parameters = []
        for fr, ch, len_ in it.product([155.5, 159.5], [0, 2], [200, 700]):
            test_parameters.append(get_test_parameters(freq=fr, pulse_len=len_, channel=ch))

        with tempfile.TemporaryDirectory() as dirname:
            test_filepath = Path(dirname) / DUMMY_FILE_NAME

            with io.open_data_file(test_filepath, 'w') as data_file:
                for param in test_parameters:
                    n_samples = param.n_samples
                    test_quad_i = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
                    test_quad_q = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
                    test_quadratures = test_quad_i + 1j * test_quad_q

                    series = io.SignalTimeSeries(DEFAULT_DATETIME, param, test_quadratures)
                    data_file.write(series)

            # Check if selector correct
            with io.open_data_file(test_filepath, series_selector=filter_) as reader:
                series_list = list(reader)

            self.assertEqual(len(series_list), 2)
            for series in series_list:
                self.assertEqual(series.parameters.frequency, valid_params['frequencies'][0])
                self.assertIn(series.parameters.channel, valid_params['channels'])
                self.assertEqual(series.parameters.pulse_length, valid_params['pulse_lengths'][0])
                self.assertEqual(series.parameters.pulse_type, valid_params['pulse_types'][0])

    def test_read_real_file(self):
        with io.open_data_file(TEST_REAL_FILEPATH) as data_file:
            series_list = list(data_file)
        self.assertEqual(len(series_list), 100)

        first_series = series_list[0]  # type: io.SignalTimeSeries
        self.assertEqual(first_series.time_mark.year, 2015)
        self.assertEqual(first_series.time_mark.month, 6)
        self.assertEqual(first_series.time_mark.day, 6)

        unique_freqs = set()
        unique_len = set()
        unique_channels = set()
        for series in series_list:
            unique_freqs.add(series.parameters.frequency)
            unique_len.add(series.parameters.pulse_length)
            unique_channels.add(series.parameters.channel)

        for freq in [155.5, 155.8, 159.5, 159.8]:
            freq = Frequency(freq, 'MHz')
            self.assertIn(freq, unique_freqs)

        for len_ in [0, 200, 700, 900]:
            len_ = TimeUnit(len_, 'us')
            self.assertIn(len_, unique_len)

        for ch in [0, 1, 2, 3]:
            ch = Channel(ch)
            self.assertIn(ch, unique_channels)


class TestWriteRead(TestCase):
    def test_write_read_reciprocity(self):
        with tempfile.TemporaryDirectory() as dirname:
            test_file_path = Path(dirname) / DUMMY_FILE_NAME

            # Create test options
            test_parameters = get_test_parameters()
            time_mark = DEFAULT_DATETIME
            n_samples = test_parameters.n_samples

            test_quad_i = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
            test_quad_q = np.random.randint(-2 ** 15 + 1, 2 ** 15, n_samples)
            test_quadratures = test_quad_i + 1j * test_quad_q

            test_series = io.SignalTimeSeries(time_mark, test_parameters, test_quadratures)
            with io.open_data_file(test_file_path, 'w') as writer:
                writer.write(test_series)

            with io.open_data_file(test_file_path, 'r') as reader:
                series = next(reader.read_series())

        # Time
        self.assertEqual(time_mark, series.time_mark)

        # Quadratures
        self.assertEqual(test_quadratures.size, series.quadratures.size)
        np.testing.assert_equal(test_quadratures, series.quadratures)

        # Check test options against read options
        self.assertEqual(test_parameters.n_samples, series.parameters.n_samples)
        self.assertEqual(test_parameters.frequency, series.parameters.frequency)
        self.assertEqual(test_parameters.pulse_type, series.parameters.pulse_type)
        self.assertEqual(test_parameters.pulse_length, series.parameters.pulse_length)
        self.assertEqual(test_parameters.sampling_frequency, series.parameters.sampling_frequency)
        self.assertEqual(test_parameters.channel, series.parameters.channel)
        self.assertEqual(test_parameters.phase_code, series.parameters.phase_code)
        self.assertEqual(test_parameters.total_delay, series.parameters.total_delay)


class TestReadFiles(TestCase):
    def test_read_by_series(self):
        with make_random_test_file() as (test_file_path, test_series_list):
            with io.read_files_by('series', test_file_path) as series_generator:
                for series, test_series in zip(series_generator, test_series_list):
                    self.assertIsInstance(series, io.SignalTimeSeries)
                    self.assertEqual(series.time_mark, test_series.time_mark)
                    self.assertEqual(series.parameters, test_series.parameters)
                    np.testing.assert_array_equal(series.quadratures, test_series.quadratures)

    def test_read_by_blocks(self):
        with make_random_test_file() as (test_file_path, test_series):
            with io.read_files_by('blocks', test_file_path) as packages_generator:
                package = next(packages_generator)
                self.assertIsInstance(package, io.TimeSeriesPackage)
                for series in package:
                    self.assertIsInstance(series, io.SignalTimeSeries)
                    self.assertEqual(package.time_mark, series.time_mark)


class TestOpenDataFile(TestCase):
    def test_input(self):
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as dirname:
                path = Path(dirname) / DUMMY_FILE_NAME
                with io.open_data_file(path, 'rb'):
                    pass

    def test_read(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / DUMMY_FILE_NAME

            with self.assertRaises(FileNotFoundError):
                with io.open_data_file(path, 'r'):
                    pass

            path.touch()

            with io.open_data_file(path, 'r') as reader:
                self.assertIsInstance(reader, io.DataFileReader)

    def test_write(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / DUMMY_FILE_NAME
            with io.open_data_file(path, 'w') as writer:
                self.assertTrue(path.exists())
                self.assertIsInstance(writer, io.DataFileWriter)

    @mock.patch('gzip.decompress')
    @mock.patch('tempfile.TemporaryFile')
    def test_read_compressed(self, mocked_gzip, mocked_tempfile):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / (DUMMY_FILE_NAME + io.ARCHIVE_EXTENSION)

            with self.assertRaises(FileNotFoundError):
                with io.open_data_file(path, 'r'):
                    pass

            path.touch()

            with io.open_data_file(path, 'r') as reader:
                self.assertIsInstance(reader, io.DataFileReader)
                self.assertTrue(mocked_gzip.called)
                self.assertTrue(mocked_tempfile.called)

    @mock.patch('gzip.compress')
    def test_write_compressed(self, mocked_gzip):
        mocked_gzip.return_value = ' '

        with tempfile.TemporaryDirectory() as dirname:
            path_with_extension = Path(dirname) / (DUMMY_FILE_NAME + io.ARCHIVE_EXTENSION)

            # When called without archive extension
            path = Path(dirname) / DUMMY_FILE_NAME
            with io.open_data_file(path, 'w', compress_on_write=True) as writer:
                self.assertIsInstance(writer, io.DataFileWriter)

            self.assertTrue(mocked_gzip.called)
            self.assertTrue(path_with_extension.exists())

            mocked_gzip.called = False

            # When called with archive extension
            with io.open_data_file(path_with_extension, 'w', compress_on_write=True) as writer:
                self.assertIsInstance(writer, io.DataFileWriter)

            self.assertTrue(mocked_gzip.called)
            self.assertTrue(path_with_extension.exists())


if __name__ == '__main__':
    main()
