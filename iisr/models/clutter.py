import numpy as np


def compute_power(signal, reduce=True, axis=0):
    if reduce:
        return (np.abs(signal) ** 2).mean(axis=axis)
    else:
        return np.abs(signal) ** 2


def signal_fft(signal):
    return np.fft.fftshift(np.fft.fft(signal, axis=0), axes=0)


def gaussian(x, mean, std, normalize=True):
    if normalize:
        norm = 1 / np.sqrt(2 * np.pi * std)
    else:
        norm = 1
    return norm * np.exp(-(x - mean) ** 2 / 2 / std ** 2)


class SignalWithClutter:
    def __init__(self, n=700, n_samples=1024, incoherent_power=30,
                 powerful_noise_peak_power=2e6,
                 powerful_noise_envelope_mean_sample=50,
                 powerful_noise_envelope_std=20,
                 clutter_power=1e5, clutter_start_sample=50, clutter_stop_sample=180,
                 phase_variation_amplitude=0.5, amplitude_variation_fraction=0.1,
                 receiver_bias_power=30, receiver_bias_variation=0.1):
        self.n = n
        self.n_samples = n_samples
        self.incoherent_power = incoherent_power
        self.clutter_power = clutter_power
        self.clutter_stop_sample = clutter_stop_sample
        self.phase_variation_amplitude = phase_variation_amplitude
        self.amplitude_variation_fraction = amplitude_variation_fraction
        self.receiver_bias_power = receiver_bias_power

        # Create basic quadratures
        quadratures = np.zeros((n, n_samples), dtype=complex)

        # Add powerful incoherent signal in the beginning
        powerful_noise = np.random.randn(n, n_samples) + 1j * np.random.randn(n, n_samples)
        powerful_noise *= np.sqrt(powerful_noise_peak_power / 2)
        powerful_noise_envelope = gaussian(np.arange(n_samples),
                                  powerful_noise_envelope_mean_sample,
                                  powerful_noise_envelope_std,
                                  normalize=False)
        powerful_noise *= powerful_noise_envelope
        quadratures += powerful_noise

        # Add receiver bias noise
        noise = np.random.randn(n, n_samples) + 1j * np.random.randn(n, n_samples)
        noise_power = (1 + (2 * np.random.rand(n) - 1) * receiver_bias_variation) \
                      * np.sqrt(receiver_bias_power)
        noise *= noise_power[:, None]
        quadratures += noise

        # Add incoherent scatter signal
        is_signal = np.random.randn(n, n_samples) + 1j * np.random.randn(n, n_samples)
        is_signal *= np.sqrt(incoherent_power / 2)
        quadratures += is_signal

        quadratures_with_clutter = quadratures.copy()

        # Add clutter with random phase
        phase = (np.random.rand() * 2 - 1) * np.pi
        clutter_slice = slice(clutter_start_sample, clutter_stop_sample)
        quadratures_with_clutter[:, clutter_slice] += np.sqrt(clutter_power) * np.exp(phase * 1j)

        # Add variations to final signal
        if phase_variation_amplitude != 0:
            phase_variation = np.sin(np.arange(n) * np.pi / n)
            phase_multiplier = np.exp(1j * phase_variation_amplitude * phase_variation[:, None])
            quadratures *= phase_multiplier
            quadratures_with_clutter *= phase_multiplier

        if amplitude_variation_fraction != 0:
            amplitude_variation = np.cos(np.arange(n) * np.pi / n / 2)
            amplitude_multiplier = 1 + amplitude_variation_fraction * amplitude_variation[:, None]
            quadratures *= amplitude_multiplier
            quadratures_with_clutter *= amplitude_multiplier

        self.quadratures = quadratures_with_clutter
        self.power = compute_power(quadratures_with_clutter)
        self.power_no_clutter = compute_power(quadratures)


class SignalWithoutClutter:
    def __init__(self, signal: SignalWithClutter, idx_slice: slice = slice(160, 200)):
        mask = np.zeros(signal.n_samples, dtype=bool)
        mask[idx_slice] = True

        corr = (signal.quadratures[0, mask] * signal.quadratures[:, mask].conj()).sum(axis=1)
        self.corr_phase = np.angle(corr)

        self.aligned_signal = signal.quadratures * np.exp(1j * self.corr_phase[:, None])
        self.clutter = self.aligned_signal.mean(axis=0)

        self.clutter_norm = (np.abs(self.clutter[mask]) ** 2).sum()
        self.amplitude_drift = (self.aligned_signal[:, mask]
                                * self.clutter[mask].conj()).sum(axis=1) \
                               / self.clutter_norm
        self.aligned_signal_no_clutter = self.aligned_signal \
                                         - self.amplitude_drift[:, None] * self.clutter
        self.power = compute_power(self.aligned_signal_no_clutter)
