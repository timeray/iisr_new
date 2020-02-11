"""
Approximation of IISR digital filter.
It is two-stage filter: CIC + FIR.
"""
from scipy import signal
import numpy as np
from iisr_old.config import CORE_PATH


N_MOVING_AVG_STAGES = 5
PATH_TO_FIR_COEFFICIENTS = str(CORE_PATH / 'iisr_old' / 'filter' / 'L255_20K.taps')


class CicFilter:
    def __init__(self, decimation, n_mov_avg=N_MOVING_AVG_STAGES, m=1,
                 fs_kHz=100e3):
        """
        Frequency response of CIC filter.

        Parameters
        ----------
        decimation: int
            Decimation of input.
        n_mov_avg: int
            Successive stages of moving averages.
        m: int
            Free integer.
        fs_kHz: float
            Sampling frequency, kHz.
        """
        self.decimation = decimation
        self.n_mov_avg = n_mov_avg
        self.m = m
        self.fs_kHz = fs_kHz
        self.fn = self.fs_kHz / 2

    def __call__(self, freqs_kHz):
        """
        Calculate response at given frequencies.

        Parameters
        ----------
        freqs_kHz: np.ndarray
            Frequencies, kHz.

        Returns
        -------
        response: np.ndarray
            Frequency response of CIC filter.
        """
        freqs_kHz = np.asarray(freqs_kHz)

        if (np.abs(freqs_kHz) > self.fn).any():
            raise ValueError('Input frequencies exceed '
                             'Nyquist frequency {:.2f} kHz'
                             ''.format(self.fn))

        freqs = (freqs_kHz / self.fs_kHz) * np.pi
        numerator = np.sin(self.decimation * self.m * freqs)
        denominator = self.decimation * self.m * np.sin(freqs)

        # Set response to 1 where frequency equals 0 and mask warning
        with np.errstate(divide='ignore', invalid='ignore'):
            response = np.abs(numerator / denominator) ** self.n_mov_avg

        response[np.isclose(freqs, 0.)] = 1.

        return response


class FirFilter:
    def __init__(self, path=PATH_TO_FIR_COEFFICIENTS, fs_kHz=1e3):
        """
        Frequency response of FIR filter.

        Parameters
        ----------
        path: str
            Path to file with coefficients.
        fs_kHz: float
            Sampling frequency, kHz.
        """
        self.coefficients = read_fir_coefficients(path)
        self.fs_kHz = fs_kHz
        self.fn = fs_kHz / 2

    def __call__(self, freqs_kHz):
        """
        Calculate response at given frequencies.

        Parameters
        ----------
        freqs_kHz: np.ndarray
            Frequencies, kHz.

        Returns
        -------
        response: np.ndarray
            Frequency response of CIC filter.
        """
        freqs_kHz = np.asarray(freqs_kHz)

        if (np.abs(freqs_kHz) > self.fn).any():
            raise ValueError('Input frequencies exceed '
                             'Nyquist frequency {:.2f} kHz'
                             ''.format(self.fn))

        normalized_freqs = (freqs_kHz / self.fs_kHz) * 2 * np.pi
        normalized_coefs = self.coefficients / self.coefficients.sum()
        _, response = signal.freqz(normalized_coefs, worN=normalized_freqs)
        return np.abs(response)


def read_fir_coefficients(path):
    """
    Read coefficients from *.taps files.

    Parameters
    ----------
    path: str
        Path to file of coefficients.

    Returns
    -------
    out: np.array
        Array of coefficients.
    """
    with open(path) as file:
        return np.array([int(coef) for coef in file.readlines()])


class DigitalFilter:
    def __init__(self,
                 cic_decimation=25,
                 cic_n=N_MOVING_AVG_STAGES,
                 fir_coefficients_filepath=PATH_TO_FIR_COEFFICIENTS,
                 fs_kHz=100e3):
        """
        IISR digital filter.

        Parameters
        ----------
        cic_decimation: int
            Decimation of CIC filter.
        cic_n: number
            Successive stages of moving average.
        fir_coefficients_filepath: str
            Path of file with FIR filter coefficients.
        fs_kHz: float
            Clock frequency, kHz.
        """
        self.cic_filter = CicFilter(decimation=cic_decimation, n_mov_avg=cic_n,
                                    fs_kHz=fs_kHz)

        self.fir_filter = FirFilter(path=fir_coefficients_filepath,
                                    fs_kHz=fs_kHz / cic_decimation)

    def __call__(self, freqs_kHz):
        """
        Calculate response.

        Parameters
        ----------
        freqs_kHz: np.array
            Frequencies, kHz.

        Returns
        -------
        response: np.ndarray
            Response of the IISR digital filter.
        """
        return self.cic_filter(freqs_kHz) * self.fir_filter(freqs_kHz)
