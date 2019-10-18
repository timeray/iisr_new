from matplotlib import pyplot as plt
from iisr.plots.helpers import PlotHelper
from iisr.filtering import MedianAdAroundMedianFilter
from pathlib import Path
from typing import BinaryIO, Dict
from iisr.utils import uneven_mean
from configparser import ConfigParser
import datetime as dt
import numpy as np


cfg_parser = ConfigParser()
CONFIG_NAME = 'ddc4_tests_config.ini'

if not cfg_parser.read('ddc4_tests_config.ini'):
    raise FileNotFoundError(f'Cannot find config {CONFIG_NAME}')

COMMON_SECTION = 'Common'
assert COMMON_SECTION in cfg_parser
assert 'data_dirpath' in cfg_parser[COMMON_SECTION]
assert 'output_dirname' in cfg_parser[COMMON_SECTION]


DATA_DIR = Path(cfg_parser[COMMON_SECTION]['data_dirpath'])


def read_n_samples(stream: BinaryIO, n_samples: int, n_channels: int) -> Dict[int, np.ndarray]:
    assert n_channels > 0
    types_list = []
    for ch in range(n_channels):
        types_list.append((f'quad_I_ch{ch}', '<i2', 1))
        types_list.append((f'quad_Q_ch{ch}', '<i2', 1))

    dtype = np.dtype(types_list)

    quadratures = np.fromfile(stream, dtype=dtype, count=n_samples)

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


def plot_spectra(dtimes, freqs, spectra_ch1, spectra_ch3):
    # plt.figure(figsize=(10, 7))
    # plt.subplot(121)
    # plt.plot(freqs, spectra_ch1.T)
    # plt.subplot(122)
    # plt.plot(freqs, spectra_ch3.T)
    # plt.show()

    plt.figure(figsize=(10, 7))
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
    plt.show()


def main():
    filename = '20191016_051053_DDC4.bin'
    start_time = dt.datetime(2019, 10, 16, 5, 10, 53)
    sampling_frequency = 5e6
    sampling_period = 1 / sampling_frequency
    central_freq = 158e6
    n_channels = 2

    filter_threshold = 5.0
    outlier_max_rate = 0.01

    n_fft = 1024
    n_acc = 1000
    n_sp = 100

    n_samples = n_fft * n_acc

    filt = MedianAdAroundMedianFilter(filter_threshold)

    freqs = central_freq + np.fft.fftfreq(n=n_fft, d=sampling_period)
    freqs = np.fft.fftshift(freqs)
    k_sp = 0

    filepath = DATA_DIR / filename
    with open(str(filepath), 'rb') as file:
        sp = {ch: [] for ch in range(n_channels)}
        dtimes = []
        while True:
            quads_dict = read_n_samples(file, n_samples, n_channels)
            elapsed_seconds = k_sp * n_samples * sampling_period
            dtimes.append(start_time + dt.timedelta(seconds=elapsed_seconds))

            for ch, quads in quads_dict.items():
                mask = filt(quads.real).mask | filt(quads.imag).mask
                mask = mask.reshape((n_acc, n_fft))

                drop_mask = np.zeros(n_acc, dtype=bool)
                outlier_rate = mask.sum(axis=1) / n_fft
                drop_mask |= outlier_rate > outlier_max_rate

                quads = quads.reshape((n_acc, n_fft))[~drop_mask]

                x = np.arange(n_acc)[~drop_mask]

                n_dropped = drop_mask.sum() * n_fft
                print(f'[{k_sp+1}/{n_sp}: {dtimes[-1].time()}] '
                      f'Number of dropped samples: {n_dropped} '
                      f'({n_dropped * 100 / n_samples:.2f} %)')

                if quads.size == 0:
                    continue

                fft = np.fft.fftshift(np.fft.fft(quads, axis=1))

                pws = calc_power_spectra(x, fft)
                sp[ch].append(pws)
                # plt.plot(freqs, pws)
                # plt.title(f'Channel {ch}')
                # plt.show()

            k_sp += 1
            if k_sp == n_sp:
                plot_spectra(dtimes, freqs, np.array(sp[0]), np.array(sp[1]))


if __name__ == '__main__':
    main()
