from iisr.plots.helpers import PlotHelper
from matplotlib import pyplot as plt
from iisr.filtering import MedianAdAroundMedianFilter
from pathlib import Path
from typing import BinaryIO, Dict
from iisr.utils import uneven_mean
from iisr.preprocessing.active import square_barker
from configparser import ConfigParser
from collections import OrderedDict, namedtuple
from scipy.constants import Boltzmann
from scipy.signal.windows import hann
from scipy import signal
import datetime as dt
import numpy as np


DATE_FMT = '%Y%m%d'
DTIME_FMT = '%Y%m%d_%H%M%S'

ADC_MAX_VOLTAGE = 1.0
ADC_BITS = 16
ADC_VOLTS_PER_BIT = ADC_MAX_VOLTAGE / (2 ** (ADC_BITS - 1))
ADC_INPUT_RESISTANCE = 50

# Config parsing
cfg_parser = ConfigParser()
CONFIG_NAME = 'ddc4_tests_config.ini'

if not cfg_parser.read('ddc4_tests_config.ini'):
    raise FileNotFoundError(f'Cannot find config {CONFIG_NAME}')

COMMON_SECTION = 'Common'
assert COMMON_SECTION in cfg_parser
cfg_common = cfg_parser[COMMON_SECTION]

DATA_DIR = Path(cfg_common['data_dirpath'])
OUTPUT_DIR = Path(cfg_common['output_dirname'])
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

N_FFT = int(cfg_common['n_fft'])
N_ACCUMULATION = int(cfg_common['n_accumulated_samples'])
N_SPECTRA = int(cfg_common['n_displayed_spectra'])

HARD_THRESHOLD = float(cfg_common['hard_threshold'])
FILTER_THRESHOLD = float(cfg_common['filter_threshold'])
OULIER_MAX_RATE = float(cfg_common['outlier_max_rate'])

SKIP_N_SAMPLES = int(cfg_common['skip_n_samples'])


class FileEndReached(RuntimeError):
    pass


def read_n_samples(stream: BinaryIO, n_samples: int, n_channels: int) -> Dict[int, np.ndarray]:
    assert n_channels > 0
    types_list = []
    for ch in range(n_channels):
        types_list.append((f'quad_I_ch{ch}', '<i2', 1))
        types_list.append((f'quad_Q_ch{ch}', '<i2', 1))

    dtype = np.dtype(types_list)

    quadratures = np.fromfile(stream, dtype=dtype, count=n_samples)
    if quadratures.size != n_samples:
        raise FileEndReached()

    res_dict = {}
    for ch in range(n_channels):
        result = np.array(quadratures[f'quad_Q_ch{ch}']).astype(np.complex64)
        result *= 1j
        result += np.array(quadratures[f'quad_I_ch{ch}'])
        res_dict[ch] = result
    return res_dict


def calc_power_spectra(x, fft):
    n_fft = fft.shape[1]
    power_spectra = np.abs(fft)  # new array
    np.power(power_spectra, 2, out=power_spectra)

    power_spectra = uneven_mean(x, power_spectra.real, axis=0)
    power_spectra /= n_fft ** 2
    return power_spectra


def parse_parameter_file(filepath: Path) -> Dict:
    convertors = {
        'NUMBER_OF_CHANNELS': lambda x: int(x) // 2,
        'NUMBER_OF_SAMPLES': int,
        'SAMPLING_RATE': float,
        'SHIFT_FREQUENCY': float,
    }

    res = {}
    with open(str(filepath)) as file:
        for line in file:
            line = line.strip().split()
            if line:
                name, value = line
                name = name.rstrip('_')

                if name not in convertors:
                    continue

                if value.endswith(','):
                    value = value.split(',')[0]
                res[name] = convertors[name](value)
    return res


def plot_spectra_1d(save_name, freqs, spectra_ch1, spectra_ch3=None, log_scale=False,
                    ylim=None):
    n_axes = 1 if spectra_ch3 is None else 2
    # xlabel = 'Frequency, MHz'
    xlabel = 'Частота, МГц'

    if log_scale:
        # ylabel = 'Power, dB'
        ylabel = 'Мощность, дБ'
        spectra_ch1 = 10 * np.log10(spectra_ch1)  # type: np.ndarray
        if spectra_ch3 is not None:
            spectra_ch3 = 10 * np.log10(spectra_ch3)  # type: np.ndarray
        if ylim is None:
            ylim = PlotHelper.autolevel(spectra_ch1)
            ylim = (ylim[0], ylim[1] * 0.8)
    else:
        # ylabel = 'Power, rel.u.'
        ylabel = 'Мощность, отн.ед.'
        if ylim is None:
            ylim = (0, PlotHelper.autolevel(spectra_ch1)[1] * 1.2)

    freqs_megahertz = freqs / 1e6

    fig = plt.figure(figsize=(7, 2.5 if spectra_ch3 is None else 5))
    plt.subplot(n_axes, 1, 1)
    plt.plot(freqs_megahertz, spectra_ch1.T)
    plt.xlim(freqs_megahertz[0], freqs_megahertz[-1])
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.grid()

    if spectra_ch3 is None:
        plt.xlabel(xlabel)
    else:
        plt.xticks([])

        plt.subplot(212)
        plt.plot(freqs_megahertz, spectra_ch3.T)
        plt.xlim(freqs_megahertz[0], freqs_megahertz[-1])
        plt.ylim(*ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()

    plt.tight_layout()
    save_path = OUTPUT_DIR / save_name
    print(f'Saving at {save_path}')
    fig.savefig(str(save_path))


def plot_spectra_2d(dtimes, freqs, spectra_ch1, spectra_ch3, save_name):
    fig = plt.figure(figsize=(10, 7))
    level = PlotHelper.autolevel(np.concatenate([spectra_ch1.ravel(),
                                                 spectra_ch3.ravel()]))
    dtimes = np.array(dtimes, dtype=dt.datetime)
    dtimes = PlotHelper.pcolor_adjust_coordinates(dtimes)
    freqs = PlotHelper.pcolor_adjust_coordinates(freqs)

    plt.subplot(121)
    plt.pcolormesh(dtimes, freqs, spectra_ch1.T, vmin=level[0], vmax=level[1])
    PlotHelper.set_time_ticks(plt.gca(), with_date=False)
    # plt.gca().set_xlim(PlotHelper.lim_daily(dtimes))

    plt.subplot(122)
    plt.pcolormesh(dtimes, freqs, spectra_ch3.T, vmin=level[0], vmax=level[1])
    PlotHelper.set_time_ticks(plt.gca(), with_date=False)
    # plt.gca().set_xlim(PlotHelper.lim_daily(dtimes))

    save_path = (OUTPUT_DIR / save_name).with_suffix('.png')
    fig.savefig(str(save_path))


def plot_fitting(save_name, freqs, noise_temp, output_power, gain, bias,
                 bandwidth, save_dir='Calibration'):
    assert len(freqs) == output_power.shape[1] == len(gain) == len(bias) == 3
    assert len(noise_temp) == len(output_power)

    fig = plt.figure(figsize=(8, 3))
    for i, (f, out_temp, g, b) in enumerate(zip(freqs, output_power.T, gain, bias)):
        plt.subplot(1, 3, i + 1)
        plt.scatter(noise_temp, out_temp)
        # plt.xlabel('Input noise, K')
        plt.xlabel('Входной шум, К')
        # plt.ylabel('Output power')
        plt.ylabel('Выходная мощность')
        # plt.title(f'{f / 1e6:.2f} MHz')
        plt.title(f'{f / 1e6:.2f} МГц')

        x = np.linspace(0, noise_temp.max())
        g *= Boltzmann * bandwidth
        y = g * (x + b)
        plt.plot(x, y, color='C1')

        plt.ylim(y[0] * 0.9, y[-1] * 1.1)

    out_dir = OUTPUT_DIR / save_dir
    if not out_dir.exists():
        out_dir.mkdir()

    fig.tight_layout()
    save_path = (out_dir / save_name).with_suffix('.png')
    fig.savefig(str(save_path))


def plot_gain_bias(save_name, freqs, gain, bias, decibel_gain=True, save_dir='Calibration'):
    fig = plt.figure(figsize=(7, 4))
    freqs /= 1e6  # to MHz
    plt.subplot(211)
    gain = 10 * np.log10(gain) if decibel_gain else gain
    plt.plot(freqs, gain)
    # plt.title('Gain')
    plt.title('Усиление')
    plt.xlim(freqs[0], freqs[-1])
    if decibel_gain:
        plt.ylim(60, 80)
        plt.xlabel('Частота, МГц')
        plt.ylabel('Усиление, дБ')
    else:
        plt.ylim(0, PlotHelper.autolevel(gain)[1] * 1.4)
        plt.xticks([])
        plt.ylabel('Усиление')
        plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.subplot(212)
    plt.plot(freqs, bias)
    # plt.title('Bias')
    plt.title('Собственные шумы')
    # plt.xlabel('Frequency, MHz')
    plt.xlabel('Частота, МГц')
    plt.ylabel(r'$T_ш$, К')
    plt.xlim(freqs[0], freqs[-1])
    plt.ylim(0, 75)

    out_dir = OUTPUT_DIR / save_dir
    if not out_dir.exists():
        out_dir.mkdir()

    fig.tight_layout()
    save_path = (out_dir / save_name).with_suffix('.png')
    fig.savefig(str(save_path))


DDCParams = namedtuple('DDCParams', ['start_time', 'sampling_frequency', 'sampling_period',
                                     'central_freq', 'n_channels', 'n_samples'])


def parse_parameters(filepath: Path) -> DDCParams:
    params = parse_parameter_file(filepath.with_suffix('.par'))

    start_time = dt.datetime.strptime(filepath.name, f'{DTIME_FMT}_DDC4.bin')
    sampling_frequency = params['SAMPLING_RATE']
    sampling_period = 1 / sampling_frequency
    central_freq = params['SHIFT_FREQUENCY']
    n_channels = params['NUMBER_OF_CHANNELS']
    n_samples = params['NUMBER_OF_SAMPLES']
    return DDCParams(start_time, sampling_frequency, sampling_period, central_freq, n_channels,
                     n_samples)


def take_quadratures(filepath: Path, n_samples: int, n_channels: int, skip_n_samples: int):
    with open(str(filepath), 'rb') as file:
        while True:
            try:
                skip_n_bytes = skip_n_samples * n_channels * 2 * 2  # 2 bytes for 2 quads
                print(f'Skip {skip_n_bytes} bytes ({skip_n_samples} samples)')
                file.seek(file.tell() + skip_n_bytes)
                yield read_n_samples(file, n_samples, n_channels)
            except FileEndReached:
                break


class QuadsFilter:
    def __init__(self, hard_threshold, filter_threshold, outlier_max_rate):
        self.hard_threshold = hard_threshold
        self.filter = MedianAdAroundMedianFilter(filter_threshold)
        self.outlier_max_rate = outlier_max_rate

    def __call__(self, quads):
        shape = quads.shape
        flat_quads = quads.ravel()
        mask = self.filter(flat_quads.real).mask | self.filter(flat_quads.imag).mask
        mask |= np.abs(flat_quads) > self.hard_threshold

        # plt.plot(np.arange(quads.size), quads.real)
        # plt.plot(np.arange(quads.size)[~mask], quads.real[~mask])
        # plt.title(f'Channel {ch}')
        # plt.show()

        mask = mask.reshape(shape)

        drop_mask = np.zeros(shape[0], dtype=bool)
        outlier_rate = mask.sum(axis=1) / shape[1]
        drop_mask |= outlier_rate > self.outlier_max_rate

        filtered_quads = quads[~drop_mask]
        left_indices = np.arange(shape[0])[~drop_mask]
        return left_indices, filtered_quads


def make_spectra(filename,
                 filt=None,
                 n_fft=N_FFT,
                 n_acc=N_ACCUMULATION,
                 skip_n_samples=SKIP_N_SAMPLES,
                 n_sp=N_SPECTRA,
                 window=False):
    filepath = DATA_DIR / filename
    print(f'Process file {filepath}')

    params = parse_parameters(filepath)

    sampling_period = params.sampling_period
    central_freq = params.central_freq
    n_channels = params.n_channels

    n_samples = n_fft * n_acc

    freqs = central_freq + np.fft.fftfreq(n=n_fft, d=sampling_period)
    freqs = np.fft.fftshift(freqs)

    sp = {ch: [] for ch in range(n_channels)}

    generator = take_quadratures(filepath, n_samples, n_channels, skip_n_samples)
    for i, quads_dict in enumerate(generator):
        for ch, quads in quads_dict.items():
            quads *= ADC_VOLTS_PER_BIT  # quadratures to voltage
            new_quads = quads.reshape((n_acc, n_fft))
            if filt is not None:
                x, new_quads = filt(new_quads)

                n_dropped = quads.size - new_quads.size
                print(f'[{i+1}/{n_sp}] '
                      f'Number of dropped samples: {n_dropped} '
                      f'({n_dropped * 100 / n_samples:.2f} %)')
            else:
                x = np.arange(n_acc)

            if new_quads.size == 0:
                continue

            if window:
                new_quads *= hann(n_fft)
            fft = np.fft.fftshift(np.fft.fft(new_quads, axis=1))

            pws = calc_power_spectra(x, fft)  # Volts^2
            pws /= ADC_INPUT_RESISTANCE  # Watts
            sp[ch].append(pws)
            # plt.plot(freqs, pws)
            # plt.title(f'Channel {ch}')
            # plt.show()

        if i + 1 == n_sp:
            break

    print('Processing finished')

    arr_ch1 = np.array(sp[0])
    arr_ch2 = np.array(sp[1]) if len(sp) > 1 else None

    return params, freqs, arr_ch1, arr_ch2


def perform_calibration(description, filepaths: Dict[str, float], filt=None, n_fft=N_FFT,
                        n_acc=N_ACCUMULATION, skip_n_samples=SKIP_N_SAMPLES):
    print(f'Perform calibration for {description}')

    first = True
    freqs = None
    params = None
    spectra = []
    noise_temperatures = []
    for filepath, noise_temp in filepaths.items():
        params, fr, sp_ch1, sp_ch2 = make_spectra(
            DATA_DIR / filepath, filt=filt, n_fft=n_fft, n_acc=n_acc, n_sp=N_SPECTRA,
            skip_n_samples=skip_n_samples
        )
        assert sp_ch2 is None

        if first:
            first = False
            freqs = fr
        else:
            assert (fr == freqs).all()

        for s in sp_ch1:
            spectra.append(s)
            noise_temperatures.append(noise_temp)

    spectra = np.array(spectra)
    bandwidth = params.sampling_frequency / n_fft
    output_temperatures = spectra / (Boltzmann * bandwidth)
    noise_temperatures = np.array(noise_temperatures)

    coef_arr = np.stack([np.ones_like(noise_temperatures), noise_temperatures]).T

    bias, gain = np.linalg.pinv(coef_arr) @ output_temperatures
    bias /= gain

    plot_idxs = [int(n_fft * x) for x in [0.25, 0.5, 0.75]]

    save_name = params.start_time.strftime(DATE_FMT) + '_' + description + '_fitting.png'
    plot_fitting(save_name, freqs[plot_idxs], noise_temperatures, spectra[:, plot_idxs],
                 gain[plot_idxs], bias[plot_idxs], bandwidth)

    save_name = params.start_time.strftime(DATE_FMT) + '_' + description + '_gain_bias.png'
    plot_gain_bias(save_name, freqs, gain, bias)


def make_all_spectra(filt=None):
    # filename : description
    # sync for synchronization from external Tk0
    # 158 for measurements at 158 MHz carrier frequency
    # 20 for measurements at ~18.75 MHz intermediate frequency
    # f1.4,f2 for measurements of noise generator with noise figure F = 1.4, 2
    # f0 when noise generator does not include additional noise (F = 1)
    filenames = OrderedDict([
        # ('noise/20191021_124746_DDC4.bin', ('158_sync_upp_50Ohm_mini_circuits', None, (-140, -110))),
        # ('noise/20191021_130607_DDC4.bin', ('158_sync_upp_f0.0', None, (-140, -110))),
        # ('noise/20191021_130821_DDC4.bin', ('158_sync_upp_f1.4', None, (-140, -110))),
        # ('noise/20191021_131028_DDC4.bin', ('158_sync_upp_f2.0', None, (-140, -110))),
        # ('noise/20191021_132113_DDC4.bin', ('20_sync_upp_f0.0', None, (-150, -90))),
        # ('noise/20191021_132448_DDC4.bin', ('20_sync_upp_f1.4', None, (-150, -90))),
        # ('noise/20191021_132714_DDC4.bin', ('20_sync_upp_f2.0', None, (-150, -90))),
        # ('noise/20191021_133444_DDC4.bin', ('20_sync_low_f0.0', None, (-150, -90))),
        # ('noise/20191021_133718_DDC4.bin', ('20_sync_low_f1.4', None, (-150, -90))),
        # ('noise/20191021_133931_DDC4.bin', ('20_sync_low_f2.0', None, (-150, -90))),
        # ('passive2/20191022_103907_DDC4.bin', ('156_cyg', None, None)),
        # ('passive2/20191022_104918_DDC4.bin', ('156_cyg', None, None)),
        # ('passive2/20191022_110044_DDC4.bin', ('156_cyg', None, None)),
        # ('passive2/20191022_111217_DDC4.bin', ('156_cyg', None, None)),
        # ('passive2/20191022_112406_DDC4.bin', ('156_cyg', None, None)),
        # ('passive2/20191022_113548_DDC4.bin', ('156_cyg', None, None)),
        # ('passive2/20191022_114745_DDC4.bin', ('156_cyg', None, None)),
        # ('passive2/20191022_120401_DDC4.bin', ('156_cyg', None, None)),
        # # main room
        # ('noise2/20191022_183506_DDC4.bin', ('158_sync_nocable_filt_f0.0', None, (-130, -90))),
        # ('noise2/20191022_183705_DDC4.bin', ('158_sync_nocable_filt_f1.4', None, (-130, -90))),
        # ('noise2/20191022_183838_DDC4.bin', ('158_sync_nocable_filt_f2.0', None, (-130, -90))),
        # # shielded room
        # ('noise3/20191023_023110_DDC4.bin', ('158_shield_nocable_arr_filt_f0.0', None, (-130, -90))),
        # ('noise3/20191023_023206_DDC4.bin', ('158_shield_nocable_arr_filt_f1.4', None, (-130, -90))),
        # ('noise3/20191023_023249_DDC4.bin', ('158_shield_nocable_arr_filt_f2.0', None, (-130, -90))),
        # ('noise3/20191023_023339_DDC4.bin', ('158_shield_nocable_arr_filt_f4.0', None, (-130, -90))),
        # ('noise3/20191023_030026_DDC4.bin', ('158_shield_nocable_filt_arr_f0.0', None, (-130, -90))),
        # ('noise3/20191023_030127_DDC4.bin', ('158_shield_nocable_filt_arr_f1.4', None, (-130, -90))),
        # ('noise3/20191023_030207_DDC4.bin', ('158_shield_nocable_filt_arr_f2.0', None, (-130, -90))),
        # ('noise3/20191023_030250_DDC4.bin', ('158_shield_nocable_filt_arr_f4.0', None, (-130, -90))),
        # Home tests
        # # Noise generator
        # ('noise_home_tests/20191106_051552_DDC4.bin', ('158_home_ng1_off', None, None)),
        # ('noise_home_tests/20191106_051500_DDC4.bin', ('158_home_ng1_on', None, None)),
        # ('noise_home_tests/20191106_063906_DDC4.bin', ('158_home_ng2_off', None, None)),
        # ('noise_home_tests/20191106_063937_DDC4.bin', ('158_home_ng2_on', None, None)),
        # ('noise_home_tests/20191106_075704_DDC4.bin', ('158_home_ng3_off', None, None)),
        # ('noise_home_tests/20191106_075801_DDC4.bin', ('158_home_ng3_on1', None, None)),
        # ('noise_home_tests/20191106_080058_DDC4.bin', ('158_home_ng3_on2', None, None)),
        # # Noise figure measurement device
        # ('noise_home_tests/20191106_060710_DDC4.bin', ('158_home_fm1_off', None, None)),
        # ('noise_home_tests/20191106_060807_DDC4.bin', ('158_home_fm1_on', None, None)),
        # ('noise_home_tests/20191106_063807_DDC4.bin', ('158_home_fm2_off', None, None)),
        # ('noise_home_tests/20191106_063829_DDC4.bin', ('158_home_fm2_on', None, None)),
        # ('noise_home_tests/20191106_075553_DDC4.bin', ('158_home_fm3_off', None, None)),
        # ('noise_home_tests/20191106_075620_DDC4.bin', ('158_home_fm3_on', None, None)),
        # Noisy filter with F=3 (old technique measurement)
        # ('noise_home_tests/20191106_083432_DDC4.bin', ('158_home_fm_noisy_off', None, None)),
        # ('noise_home_tests/20191106_083500_DDC4.bin', ('158_home_fm_noisy_on', None, None)),
        # Home - second day of measurements
        ('noise_home_tests/20191107_021118_DDC4.bin', ('158_home_ng4_off', None, None)),
        ('noise_home_tests/20191107_021141_DDC4.bin', ('158_home_ng4_on', None, None)),
        ('noise_home_tests/20191107_021206_DDC4.bin', ('158_home_ng5_off', None, None)),
        ('noise_home_tests/20191107_021226_DDC4.bin', ('158_home_ng5_on', None, None)),
        ('noise_home_tests/20191107_021434_DDC4.bin', ('158_home_fm4_off', None, None)),
        ('noise_home_tests/20191107_021500_DDC4.bin', ('158_home_fm4_on', None, None)),
        ('noise_home_tests/20191107_021545_DDC4.bin', ('158_home_ng6_off', None, None)),
        ('noise_home_tests/20191107_021603_DDC4.bin', ('158_home_ng6_on1', None, None)),
        ('noise_home_tests/20191107_021649_DDC4.bin', ('158_home_fm5_off', None, None)),
        ('noise_home_tests/20191107_021720_DDC4.bin', ('158_home_fm5_on', None, None)),
    ])

    for filename, (descr, ylim_lin, ylim_log) in filenames.items():
        params, freqs, sp_ch1, sp_ch2 = make_spectra(filename, filt=filt)

        save_name = params.start_time.strftime(DTIME_FMT)
        if descr is not None:
            save_name += f'_{descr}'
        plot_spectra_1d(save_name + '_lin.png', freqs, sp_ch1, sp_ch2, log_scale=False,
                        ylim=ylim_lin)
        plot_spectra_1d(save_name + '_log.png', freqs, sp_ch1, sp_ch2, log_scale=True,
                        ylim=ylim_log)


def perform_all_calibration(filt=None):
    # filename : noise temperature
    temp_off = 290
    temp_f1_4 = 696
    temp_f2 = 870
    temp_f4 = 1450

    noise_gen_enr = 6  # dB
    temp_noise_gen = temp_off * (10 ** (noise_gen_enr / 10) + 1)  # ~1444 K

    calib_files = [
        # Tuple[description, Dict[filename, temperature]]
        # ('158_full', {
        #      'noise/20191021_130607_DDC4.bin': temp_off,
        #      'noise/20191021_130821_DDC4.bin': temp_f1_4,
        #      'noise/20191021_131028_DDC4.bin': temp_f2,
        #  }),
        # ('20_upp',
        #  {
        #      'noise/20191021_132113_DDC4.bin': temp_off,
        #      'noise/20191021_132448_DDC4.bin': temp_f1_4,
        #      'noise/20191021_132714_DDC4.bin': temp_f2,
        #  }),
        # ('20_low',
        #  {
        #      'noise/20191021_133444_DDC4.bin': temp_off,
        #      'noise/20191021_133718_DDC4.bin': temp_f1_4,
        #      'noise/20191021_133931_DDC4.bin': temp_f2,
        #  }),
        # ('158_nocable_arr_filt',
        #  {
        #      'noise3/20191023_023110_DDC4.bin': temp_off,
        #      'noise3/20191023_023206_DDC4.bin': temp_f1_4,
        #      'noise3/20191023_023249_DDC4.bin': temp_f2,
        #      'noise3/20191023_023339_DDC4.bin': temp_f4,
        #  }),
        # ('158_nocable_filt_arr',
        #  {
        #      'noise3/20191023_030026_DDC4.bin': temp_off,
        #      'noise3/20191023_030127_DDC4.bin': temp_f1_4,
        #      'noise3/20191023_030207_DDC4.bin': temp_f2,
        #      'noise3/20191023_030250_DDC4.bin': temp_f4,
        #  }),

        # Home noise measurements
        # ('158_home_noise_gen', {
        #     'noise_home_tests/20191106_051552_DDC4.bin': temp_off,
        #     'noise_home_tests/20191106_051500_DDC4.bin': temp_noise_gen,
        # }),
        # ('158_home_figure_measure', {
        #     'noise_home_tests/20191106_060710_DDC4.bin': temp_off,
        #     'noise_home_tests/20191106_060807_DDC4.bin': temp_f4,
        # }),
        # ('158_home_figure_measure2', {
        #     'noise_home_tests/20191106_063807_DDC4.bin': temp_off,
        #     'noise_home_tests/20191106_063829_DDC4.bin': temp_f4,
        # }),
        # ('158_home_noise_gen2', {
        #     'noise_home_tests/20191106_063906_DDC4.bin': temp_off,
        #     'noise_home_tests/20191106_063937_DDC4.bin': temp_noise_gen,
        # }),
        # ('158_home_figure_measure3', {
        #     'noise_home_tests/20191106_075553_DDC4.bin': temp_off,
        #     'noise_home_tests/20191106_075620_DDC4.bin': temp_f4,
        # }),
        # ('158_home_noise_gen3', {
        #     'noise_home_tests/20191106_075704_DDC4.bin': temp_off,
        #     'noise_home_tests/20191106_075801_DDC4.bin': temp_noise_gen,
        #     'noise_home_tests/20191106_080058_DDC4.bin': temp_noise_gen,
        # }),
        # ('158_home_figure_measure_noisy', {
        #     'noise_home_tests/20191106_083432_DDC4.bin': temp_off,
        #     'noise_home_tests/20191106_083500_DDC4.bin': temp_f4,
        # }),
        ('158_home_noise_gen4', {
            'noise_home_tests/20191107_021118_DDC4.bin': temp_off,
            'noise_home_tests/20191107_021141_DDC4.bin': temp_noise_gen,
        }),
        ('158_home_noise_gen5', {
            'noise_home_tests/20191107_021206_DDC4.bin': temp_off,
            'noise_home_tests/20191107_021226_DDC4.bin': temp_noise_gen,
        }),
        ('158_home_figure_measure4', {
            'noise_home_tests/20191107_021434_DDC4.bin': temp_off,
            'noise_home_tests/20191107_021500_DDC4.bin': temp_f4,
        }),
        ('158_home_noise_gen6', {
            'noise_home_tests/20191107_021545_DDC4.bin': temp_off,
            'noise_home_tests/20191107_021603_DDC4.bin': temp_noise_gen,
        }),
        ('158_home_figure_measure5', {
            'noise_home_tests/20191107_021649_DDC4.bin': temp_off,
            'noise_home_tests/20191107_021720_DDC4.bin': temp_f4,
        }),
    ]

    for descr, filepaths_dict in calib_files:
        perform_calibration(descr, filepaths_dict, filt=filt)


def passive_data_processing():
    quadratures_filter = QuadsFilter(
        hard_threshold=HARD_THRESHOLD,
        filter_threshold=FILTER_THRESHOLD,
        outlier_max_rate=OULIER_MAX_RATE,
    )
    # make_all_spectra(filt=quadratures_filter)
    perform_all_calibration(filt=quadratures_filter)


def active_sat_data_processing():
    filename = '20191114_051654_DDC4.bin'

    filepath = DATA_DIR / filename
    print(f'Process file {filepath}')

    params = parse_parameters(filepath)

    sampling_period = params.sampling_period
    central_freq = params.central_freq
    n_channels = params.n_channels
    n_samples = params.n_samples

    generator = take_quadratures(filepath, n_samples * 5, n_channels, n_samples * 10)

    for quads_dict in generator:
        quads = quads_dict[0].reshape(5, n_samples)
        plt.figure(figsize=(15, 5))
        for q in quads:

            plt.plot(q.real)
            #plt.plot(q.imag)
        plt.show()


def active_is_data_processing():
    filename = '20191114_030635_DDC4.bin'
    # filename = '20191114_035027_DDC4.bin'

    filepath = DATA_DIR / filename
    print(f'Process file {filepath}')

    params = parse_parameters(filepath)

    sampling_period = params.sampling_period
    central_freq = params.central_freq
    n_channels = params.n_channels
    n_samples = params.n_samples

    dlength = 200e-6 / sampling_period
    code = square_barker(int(dlength), 5)

    n_acc = 1000

    nyq = 0.5 / sampling_period
    crit_norm_freq = 100000 / nyq

    filt = signal.butter(7, crit_norm_freq)  # type: Tuple[np.ndarray, np.ndarray]

    generator = take_quadratures(filepath, n_samples * n_acc, n_channels, 0)

    all_power = []
    for quads_dict in generator:
        quads = quads_dict[0].reshape(n_acc, n_samples)

        # for q in quads:
        #     plt.plot(q.real)
        #     plt.plot(q.imag)
        #     plt.show()

        freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, d=sampling_period)) + central_freq

        spectra = np.fft.fftshift(np.mean(np.abs(np.fft.fft(quads, axis=1)) ** 2, axis=0))

        quads = signal.lfilter(filt[0], filt[1], quads, axis=1)

        filt_spectra = np.fft.fftshift(np.mean(np.abs(np.fft.fft(quads, axis=1)) ** 2, axis=0))
        plt.plot(freqs, spectra)
        plt.plot(freqs, filt_spectra)
        plt.show()

        quads = np.apply_along_axis(signal.correlate, axis=1, arr=quads, in2=code, mode='same') \
                / np.sqrt(len(code))

        power = (quads.real ** 2 + quads.imag ** 2).mean(axis=0)
        plt.figure(figsize=(15, 5))
        plt.plot(power)
        # plt.ylim(6000, 10000)
        plt.ylim(30000, 50000)
        plt.show()
        all_power.append(power)

    plt.pcolormesh(np.array(all_power), vmin=30000, vmax=50000)
    plt.show()


if __name__ == '__main__':
    passive_data_processing()
    # active_sat_data_processing()
    # active_is_data_processing()

