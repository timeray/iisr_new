import numpy as np
from matplotlib import pyplot as plt
from iisr_old.filter.digital import FirFilter, CicFilter, DigitalFilter
from iisr_old.plotting import PlotHelper


def plot_cic_response():
    fs = 100e3
    decimation = 25
    cic_fn = fs / 2

    freqs = np.linspace(0, cic_fn, 500)
    cic_filter = CicFilter(decimation)
    cic_h = cic_filter(freqs)
    cic_h_dB = 20 * np.log10(cic_h)
    plt.plot(freqs, cic_h_dB)
    plt.xlim(freqs[0], freqs[-1])
    plt.ylim(-120, 0)
    plt.xlabel('Частота, кГц')
    plt.title('АЧХ CIC-фильтра')

    plt.show()


def plot_fir_response():
    fs = 4e3
    fir_fn = fs / 2

    freqs = np.linspace(0, fir_fn, 500, endpoint=False)
    fir_filter = FirFilter(fs_kHz=fs)
    fir_h = fir_filter(freqs)
    fir_h_dB = 20 * np.log10(fir_h)
    plt.plot(freqs, fir_h_dB)
    plt.xlim(freqs[0], freqs[-1])
    plt.ylim(-120, 0)
    plt.xlabel('Частота, кГц')
    plt.title('АЧХ FIR-фильтра')

    plt.show()


def plot_all_response():
    fs = 100e3
    cic_decimation = 25
    fir_decimation = 4
    fir_fs = fs / cic_decimation
    fir_fn = fir_fs / 2

    freqs = np.linspace(0, fir_fn, 500, endpoint=False)

    fig = plt.figure(figsize=(15, 5))

    ax_cic = plt.subplot(221)
    cic_filter = CicFilter(cic_decimation, fs_kHz=fs)
    cic_h = cic_filter(freqs)
    cic_h_dB = 20 * np.log10(cic_h)
    ax_cic.plot(freqs, cic_h_dB)
    ax_cic.set_xlim(freqs[0], freqs[-1])
    ax_cic.set_ylim(-120, 0)
    ax_cic.set_title('АЧХ CIC-фильтра')

    ax_fir = plt.subplot(223, sharex=ax_cic)
    fir_filter = FirFilter(fs_kHz=fir_fs)
    fir_h = fir_filter(freqs)
    fir_h_dB = 20 * np.log10(fir_h)
    ax_fir.plot(freqs, fir_h_dB)
    ax_fir.set_xlim(freqs[0], freqs[-1])
    ax_fir.set_ylim(-120, 0)
    ax_fir.set_xlabel('Частота, кГц')
    ax_fir.set_title('АЧХ FIR-фильтра')

    ax_all = plt.subplot(122)
    digital_filter = DigitalFilter(cic_decimation, fs_kHz=fs)
    all_h = digital_filter(freqs)
    all_h_dB = 20 * np.log10(all_h)
    ax_all.plot(freqs, all_h_dB)
    ax_all.set_xlim(freqs[0], freqs[-1])
    ax_all.set_ylim(-120, 0)
    ax_all.set_xlabel('Частота, кГц')
    ax_all.set_title('Общая АЧХ цифрового фильтра')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_cic_response()
    # plot_fir_response()
    plot_all_response()