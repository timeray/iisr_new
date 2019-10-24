from iisr.plots.helpers import PlotHelper
from matplotlib import pyplot as plt
from iisr.filtering import MedianAdAroundMedianFilter
from pathlib import Path
from typing import BinaryIO, Dict
from iisr.utils import uneven_mean
from configparser import ConfigParser
from collections import OrderedDict, namedtuple
from scipy.constants import Boltzmann
from scipy.signal.windows import hann
import datetime as dt
import numpy as np


cfg_parser = ConfigParser()
CONFIG_NAME = 'ddc4_tests_config.ini'

if not cfg_parser.read('ddc4_tests_config.ini'):
    raise FileNotFoundError(f'Cannot find config {CONFIG_NAME}')

COMMON_SECTION = 'Common'
assert COMMON_SECTION in cfg_parser
cfg_common = cfg_parser[COMMON_SECTION]

DATE_FMT = '%Y%m%d'
DTIME_FMT = '%Y%m%d_%H%M%S'

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
        ylim = (-40, 20) if ylim is None else ylim
    else:
        # ylabel = 'Power, rel.u.'
        ylabel = 'Мощность, отн.ед.'
        ylim = (0, PlotHelper.autolevel(spectra_ch1)[1]) if ylim is None else ylim

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

    out_dir = OUTPUT_DIR / save_dir
    if not out_dir.exists():
        out_dir.mkdir()

    fig.tight_layout()
    save_path = (out_dir / save_name).with_suffix('.png')
    fig.savefig(str(save_path))


def plot_gain_bias(save_name, freqs, gain, bias, save_dir='Calibration'):
    fig = plt.figure(figsize=(7, 4))
    freqs /= 1e6  # to MHz
    plt.subplot(211)
    plt.plot(freqs, gain)
    # plt.title('Gain')
    plt.title('Усиление')
    plt.xlim(freqs[0], freqs[-1])
    plt.ylim(0, PlotHelper.autolevel(gain)[1] * 1.4)
    plt.xticks([])
    plt.ylabel('Усиление')

    plt.subplot(212)
    plt.plot(freqs, bias)
    # plt.title('Bias')
    plt.title('Собственные шумы')
    # plt.xlabel('Frequency, MHz')
    plt.xlabel('Частота, МГц')
    plt.ylabel(r'$T_ш$, К')
    plt.xlim(freqs[0], freqs[-1])
    plt.ylim(0, 600)

    out_dir = OUTPUT_DIR / save_dir
    if not out_dir.exists():
        out_dir.mkdir()

    fig.tight_layout()
    save_path = (out_dir / save_name).with_suffix('.png')
    fig.savefig(str(save_path))


DDCParams = namedtuple('DDCParams', ['start_time', 'sampling_frequency', 'sampling_period',
                                     'central_freq', 'n_channels'])


def parse_parameters(filepath: Path) -> DDCParams:
    params = parse_parameter_file(filepath.with_suffix('.par'))

    start_time = dt.datetime.strptime(filepath.name, f'{DTIME_FMT}_DDC4.bin')
    sampling_frequency = params['SAMPLING_RATE']
    sampling_period = 1 / sampling_frequency
    central_freq = params['SHIFT_FREQUENCY']
    n_channels = params['NUMBER_OF_CHANNELS']
    return DDCParams(start_time, sampling_frequency, sampling_period, central_freq, n_channels)


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

            pws = calc_power_spectra(x, fft)
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
            DATA_DIR / filepath, filt=filt, n_fft=n_fft, n_acc=n_acc, n_sp=3,
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
        ('20191021_124746_DDC4.bin', ('158_sync_upp_50Ohm_mini_circuits', (0, 0.4), (-40, 0))),
        ('20191021_130607_DDC4.bin', ('158_sync_upp_f0.0', (0, 0.4), (-40, 0))),
        ('20191021_130821_DDC4.bin', ('158_sync_upp_f1.4', (0, 0.4), (-40, 0))),
        ('20191021_131028_DDC4.bin', ('158_sync_upp_f2.0', (0, 0.4), (-40, 0))),
        ('20191021_132113_DDC4.bin', ('20_sync_upp_f0.0', None, (-40, 20))),
        ('20191021_132448_DDC4.bin', ('20_sync_upp_f1.4', None, (-40, 20))),
        ('20191021_132714_DDC4.bin', ('20_sync_upp_f2.0', None, (-40, 20))),
        ('20191021_133444_DDC4.bin', ('20_sync_low_f0.0', None, (-40, 20))),
        ('20191021_133718_DDC4.bin', ('20_sync_low_f1.4', None, (-40, 20))),
        ('20191021_133931_DDC4.bin', ('20_sync_low_f2.0', None, (-40, 20))),
        ('20191022_103907_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        ('20191022_104918_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        ('20191022_110044_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        ('20191022_111217_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        ('20191022_112406_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        ('20191022_113548_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        ('20191022_114745_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        ('20191022_120401_DDC4.bin', ('156_cyg', (0, 0.4), None)),
        # main room
        ('20191022_183506_DDC4.bin', ('158_sync_nocable_filt_f0.0', None, (-20, 20))),
        ('20191022_183705_DDC4.bin', ('158_sync_nocable_filt_f1.4', None, (-20, 20))),
        ('20191022_183838_DDC4.bin', ('158_sync_nocable_filt_f2.0', None, (-20, 20))),
        # shielded room
        ('20191023_023110_DDC4.bin', ('158_shield_nocable_arr_filt_f0.0', None, (-20, 20))),
        ('20191023_023206_DDC4.bin', ('158_shield_nocable_arr_filt_f1.4', None, (-20, 20))),
        ('20191023_023249_DDC4.bin', ('158_shield_nocable_arr_filt_f2.0', None, (-20, 20))),
        ('20191023_023339_DDC4.bin', ('158_shield_nocable_arr_filt_f4.0', None, (-20, 20))),
        ('20191023_030026_DDC4.bin', ('158_shield_nocable_filt_arr_f0.0', None, (-20, 20))),
        ('20191023_030127_DDC4.bin', ('158_shield_nocable_filt_arr_f1.4', None, (-20, 20))),
        ('20191023_030207_DDC4.bin', ('158_shield_nocable_filt_arr_f2.0', None, (-20, 20))),
        ('20191023_030250_DDC4.bin', ('158_shield_nocable_filt_arr_f4.0', None, (-20, 20))),
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

    calib_files = [
        # Tuple[description, Dict[filename, temperature]]
        ('158_full', {
             '20191021_130607_DDC4.bin': temp_off,
             '20191021_130821_DDC4.bin': temp_f1_4,
             '20191021_131028_DDC4.bin': temp_f2,
         }),
        ('20_upp',
         {
             '20191021_132113_DDC4.bin': temp_off,
             '20191021_132448_DDC4.bin': temp_f1_4,
             '20191021_132714_DDC4.bin': temp_f2,
         }),
        ('20_low',
         {
             '20191021_133444_DDC4.bin': temp_off,
             '20191021_133718_DDC4.bin': temp_f1_4,
             '20191021_133931_DDC4.bin': temp_f2,
         }),
        ('158_nocable_arr_filt',
         {
             '20191023_023110_DDC4.bin': temp_off,
             '20191023_023206_DDC4.bin': temp_f1_4,
             '20191023_023249_DDC4.bin': temp_f2,
             '20191023_023339_DDC4.bin': temp_f4,
         }),
        ('158_nocable_filt_arr',
         {
             '20191023_030026_DDC4.bin': temp_off,
             '20191023_030127_DDC4.bin': temp_f1_4,
             '20191023_030207_DDC4.bin': temp_f2,
             '20191023_030250_DDC4.bin': temp_f4,
         }),
    ]
    for descr, filepaths_dict in calib_files:
        perform_calibration(descr, filepaths_dict, filt=filt)


if __name__ == '__main__':
    quadratures_filter = QuadsFilter(
        hard_threshold=HARD_THRESHOLD,
        filter_threshold=FILTER_THRESHOLD,
        outlier_max_rate=OULIER_MAX_RATE,
    )
    make_all_spectra(filt=quadratures_filter)
    perform_all_calibration(filt=quadratures_filter)
