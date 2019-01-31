import numpy as np
from matplotlib import pyplot as plt
from iisr.models.clutter import SignalWithClutter, SignalWithoutClutter, compute_power, signal_fft


def plot_all_phase(signal):
    plt.plot(np.angle(signal.T))


def plot_all_module(signal):
    plt.plot(np.abs(signal.T))


def plot_power_clutter_vs_no_clutter(raw_power, clean_power, orig_power):
    plt.plot(raw_power, label='Raw power')
    plt.plot(clean_power, label='Power without clutter')
    plt.plot(orig_power, label='Original power')
    plt.legend()


def plot_signal_fft(ax, signal):
    power_spectra = compute_power(signal_fft(signal), reduce=False)
    vmin = np.percentile(power_spectra.ravel(), 5)
    vmax = np.percentile(power_spectra.ravel(), 95)
    ax.pcolormesh(power_spectra.T, vmin=vmin, vmax=vmax)


def plot_estimated_variations(phase_variation, amplitude_variation):
    plt.figure(figsize=(8, 5))
    ax_phase = plt.subplot(211)
    ax_phase.plot(phase_variation)

    ax_ampl = plt.subplot(212)
    ax_ampl.plot(amplitude_variation)


def plot_fft_compare(original_signal, phase_corrected_signal, no_clutter_signal):
    plt.figure(figsize=(15, 5))

    ax_raw = plt.subplot(131)
    plot_signal_fft(ax_raw, original_signal)

    ax_phase = plt.subplot(132)
    plot_signal_fft(ax_phase, phase_corrected_signal)

    ax_no = plt.subplot(133)
    plot_signal_fft(ax_no, no_clutter_signal)


if __name__ == '__main__':
    raw_signal = SignalWithClutter(amplitude_variation_fraction=0.1, phase_variation_amplitude=0.5)
    signal_no_clutter = SignalWithoutClutter(raw_signal)

    # plot_all_phase(raw_signal.quadratures)
    # plot_all_module(raw_signal.quadratures)
    # plot_fft_compare(raw_signal.quadratures,
    #                  signal_no_clutter.aligned_signal,
    #                  signal_no_clutter.aligned_signal_no_clutter)
    # plot_estimated_variations(signal_no_clutter.corr_phase, signal_no_clutter.amplitude_drift)
    plot_power_clutter_vs_no_clutter(raw_signal.power, signal_no_clutter.power,
                                     raw_signal.power_no_clutter)
    plt.show()
