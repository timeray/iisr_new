from collections import defaultdict
from pathlib import Path

import numpy as np
import datetime as dt

from typing import List, Dict, Union, Optional

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from iisr.postprocessing.passive import SourceTrackInfo, SkyPowerInfo, CalibrationInfo, \
    SunPatternInfo, SunFluxInfo
from iisr.preprocessing.passive import PassiveScan, PassiveTrack
from iisr.units import Frequency
from iisr.plots.helpers import *

FIGSIZE_ONELONG = (15, 7.2)  # (10, 3.6)
FIGSIZE_TWOLONG = (15, 10)  # (10, 7.2)


def __get_split_args(dtimes_, timeout: dt.timedelta):
    return np.where(np.abs(np.diff(dtimes_)) > timeout)[0] + 1


def _split_1d_by_timeout(dtimes_, *arrs, timeout: dt.timedelta):
    split_args = __get_split_args(dtimes_, timeout)
    new_dtimes = np.split(dtimes_, split_args)
    new_arrs = [np.split(arr, split_args) for arr in arrs]
    return [new_dtimes] + new_arrs


def _split_scan_by_timeout(dtimes_, freqs_, array_: Union[np.ndarray, Dict], timeout: dt.timedelta):
    split_args = __get_split_args(dtimes_, timeout)
    new_dtimes = np.split(dtimes_, split_args)
    new_freqs = [freqs_ for _ in range(len(split_args)+1)]
    if isinstance(array_, dict):
        new_arrays = {ch: np.split(arr, split_args) for ch, arr in array_.items()}
    else:
        new_arrays = np.split(array_, split_args)
    return new_dtimes, new_freqs, new_arrays


def _split_track_by_timeout(dtimes_, freqs_, array_: Union[np.ndarray, Dict], timeout: dt.timedelta):
    split_args = __get_split_args(dtimes_, timeout)
    new_dtimes = np.split(dtimes_, split_args)
    new_freqs = [Frequency(arr, 'Hz') for arr in np.split(freqs_['Hz'], split_args)]
    if isinstance(array_, dict):
        new_arrays = {ch: np.split(arr, split_args) for ch, arr in array_.items()}
    else:
        new_arrays = np.split(array_, split_args)
    return new_dtimes, new_freqs, new_arrays


def _prepare_scan_tracks_spectra(scan: Optional[PassiveScan], tracks: List[PassiveTrack],
                                 decimation: int, timeout: dt.timedelta):
    dtimes = []
    freqs = []
    value = defaultdict(list)

    if scan is not None:
        dtimes_scan, freqs_scan, spectra_scan = scan.time_marks, scan.frequencies, scan.spectra
        dtimes_scan = dtimes_scan[::decimation]
        spectra_scan = {ch: sp[::decimation] for ch, sp in spectra_scan.items()}

        dtimes_scan, freqs_scan, spectra_scan = _split_scan_by_timeout(
            dtimes_scan, freqs_scan, spectra_scan, timeout=timeout
        )

        dtimes.extend(dtimes_scan)
        freqs.extend(freqs_scan)
        for ch, arr in spectra_scan.items():
            value[ch].extend(arr)

    for track in tracks:
        dtimes_src, freqs_src, spectra_src = track.time_marks, track.frequencies, track.spectra
        dtimes_src = dtimes_src[::decimation]
        freqs_src = Frequency(freqs_src['Hz'][::decimation], 'Hz')
        spectra_src = {ch: sp[::decimation] for ch, sp in spectra_src.items()}

        dtimes_src, freqs_src, spectra_src = _split_track_by_timeout(
            dtimes_src, freqs_src, spectra_src, timeout=timeout
        )

        dtimes.extend(dtimes_src)
        freqs.extend(freqs_src)
        for ch, arr in spectra_src.items():
            value[ch].extend(arr)
    return dtimes, freqs, value


def _prepare_scan_tracks_coherence(scan: Optional[PassiveScan], tracks: List[PassiveTrack],
                                   decimation: int, timeout: dt.timedelta):
    dtimes = []
    freqs = []
    value = []

    if scan is not None:
        dtimes_scan, freqs_scan, coherence_scan = scan.time_marks, scan.frequencies, scan.coherence
        dtimes_scan = dtimes_scan[::decimation]
        coherence_scan = coherence_scan[::decimation]

        dtimes_scan, freqs_scan, coherence_scan = _split_scan_by_timeout(
            dtimes_scan, freqs_scan, coherence_scan, timeout=timeout
        )

        dtimes.extend(dtimes_scan)
        freqs.extend(freqs_scan)
        value.extend(coherence_scan)

    for track in tracks:
        dtimes_src, freqs_src, coherence_src = track.time_marks, track.frequencies, track.coherence
        dtimes_src = dtimes_src[::decimation]
        freqs_src = Frequency(freqs_src['Hz'][::decimation], 'Hz')
        coherence_src = coherence_src[::decimation]

        dtimes_src, freqs_src, coherence_src = _split_track_by_timeout(
            dtimes_src, freqs_src, coherence_src, timeout=timeout
        )

        dtimes.extend(dtimes_src)
        freqs.extend(freqs_src)
        value.extend(coherence_src)
    return dtimes, freqs, value


@plot_languages()
def plot_daily_spectra(scan: Optional[PassiveScan], tracks: List[PassiveTrack], save_folder: Path,
                       decimation=10, figsize=FIGSIZE_ONELONG,
                       level=None, timeout=dt.timedelta(minutes=30), colored=True, language=None):
    """
    Plot overall spectra for passive data.
    """
    # Store frequency and spectra arrays - they have different shape
    # for scan mode and for tracking mode
    dtimes, freqs, spectra = _prepare_scan_tracks_spectra(scan, tracks, decimation, timeout)

    for sp in spectra.values():
        for i in range(len(sp)):
            sp[i] = np.ma.array(sp[i], mask=np.isnan(sp[i]))

    if colored:
        cmap = None
    else:
        cmap = 'gray'

    for ch, spectrum in spectra.items():
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)  # type: plt.Axes

        if level is None:
            low, upp = PlotHelper.autolevel(np.ma.concatenate([sp.ravel() for sp in spectrum]))
        else:
            low, upp = level

        for dt_, fr, sp in zip(dtimes, freqs, spectrum):
            dt_ = PlotHelper.pcolor_adjust_coordinates(dt_)
            fr = PlotHelper.pcolor_adjust_coordinates(fr['MHz'])
            pcm = ax.pcolormesh(dt_, fr.T, sp.T, vmin=low, vmax=upp, cmap=cmap)

        PlotHelper.set_time_ticks(ax, with_date=True)
        ax.set_xlim(PlotHelper.lim_daily(np.concatenate([dt_ for dt_ in dtimes])))

        title = lang_string({'ru': 'Принятая мощность, отн.ед',
                             'en': 'Received power, rel.units'}, language)
        ax.set_title(title)
        xlabel = lang_string({'ru': 'Время, UT', 'en': 'Time, UT'}, language)
        ax.set_xlabel(xlabel)
        ylabel = lang_string({'ru': 'Частота, МГц', 'en': 'Frequency, MHz'},
                             language)
        ax.set_ylabel(ylabel)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        plt.colorbar(pcm, cax=cax)
        fig.tight_layout()

        save_name = 'spectra_ch{}'.format(ch)
        fig.savefig(save_folder / save_name)

        plt.close(fig)


@plot_languages()
def plot_daily_coherence(scan: Optional[PassiveScan], tracks: List[PassiveTrack], save_folder: Path,
                         decimation=10, figsize=FIGSIZE_TWOLONG,
                         timeout=dt.timedelta(minutes=30), colored=True, language=None):
    """
    Plot coherence for passive data.
    """
    # Store frequency and spectra arrays - they have different shape
    # for scan mode and for tracking mode

    dtimes, freqs, coherence = _prepare_scan_tracks_coherence(scan, tracks, decimation, timeout)

    fig = plt.figure(figsize=figsize)

    ax_module = plt.subplot(211)  # type: plt.Axes
    ax_phase = plt.subplot(212)  # type: plt.Axes

    if colored:
        cmap = None
    else:
        cmap = 'gray'

    for dt_, fr, coh in zip(dtimes, freqs, coherence):
        dt_ = PlotHelper.pcolor_adjust_coordinates(dt_)
        fr = PlotHelper.pcolor_adjust_coordinates(fr['MHz'])
        module = np.abs(coh)
        phase = np.angle(coh)
        pcm_module = ax_module.pcolormesh(dt_, fr.T, module.T, vmin=0., vmax=1., cmap=cmap)
        pcm_phase = ax_phase.pcolormesh(dt_, fr.T, phase.T, vmin=-np.pi, vmax=np.pi, cmap=cmap)

    PlotHelper.set_time_ticks(ax_module, with_date=True)
    PlotHelper.set_time_ticks(ax_phase, with_date=True)
    dtimes = np.concatenate([dt_ for dt_ in dtimes])
    ax_module.set_xlim(PlotHelper.lim_daily(dtimes))
    ax_phase.set_xlim(PlotHelper.lim_daily(dtimes))
    title = {'ru': 'Модуль коэффциента когерентности', 'en': 'Coherence coefficient module'}
    ax_module.set_title(lang_string(title, language))
    title = {'ru': 'Аргумент коэффциента когерентности', 'en': 'Coherence coefficient angle'}
    ax_phase.set_title(lang_string(title, language))
    ax_phase.set_xlabel(lang_string({'ru': 'Время, UT', 'en': 'Time, UT'}, language))
    ylabel = {'ru': 'Частота, кГц', 'en': 'Frequency, kHz'}
    ax_module.set_ylabel(lang_string(ylabel, language))
    ax_phase.set_ylabel(lang_string(ylabel, language))

    divider = make_axes_locatable(ax_module)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(pcm_module, cax=cax)

    divider = make_axes_locatable(ax_phase)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(pcm_phase, cax=cax)

    fig.tight_layout()

    save_name = 'coherence.png'
    fig.savefig(save_folder / save_name)

    plt.close(fig)


@plot_languages()
def plot_processed_tracks(track: SourceTrackInfo, save_folder: Path, colored=True,
                          figsize=FIGSIZE_ONELONG, timeout=dt.timedelta(minutes=10), language=None):
    time_marks = track.time_marks
    spectra = track.spectra_central_track

    channels = sorted(spectra)
    spectra = [spectra[ch] for ch in channels]

    res = _split_1d_by_timeout(time_marks, *spectra, timeout=timeout)
    tm_split, spectra_split = res[0], res[1:]

    color = 'C0' if colored else 'gray'

    for ch, sp_split in zip(channels, spectra_split):
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)  # type: plt.Axes

        for tm, sp in zip(tm_split, sp_split):
            ax.plot(tm, sp, color=color)

        PlotHelper.set_time_ticks(ax, with_date=True, minticks=10)

        ax.set_xlim(time_marks[0], time_marks[-1])

        xlabel_time = lang_string({'ru': 'Время, UT', 'en': 'Time, UT'}, language)
        ax.set_xlabel(xlabel_time)

        ylabel_time = lang_string({'ru': 'Мощность, отн.ед.', 'en': 'Power, rel.units'}, language)
        ax.set_ylabel(ylabel_time)

        fig.tight_layout()

        save_name = f'track_spectra_{track.mode.name}_ch{ch}.png'
        fig.savefig(save_folder / save_name)

        plt.close(fig)


@plot_languages()
def plot_sky_power(sky_power_info: SkyPowerInfo, save_folder: Path, colored=True,
                   figsize=FIGSIZE_ONELONG, level=None, timeout=dt.timedelta(minutes=10),
                   language=None):

    time_marks = sky_power_info.time_marks
    frequencies = sky_power_info.frequencies
    power = sky_power_info.values

    time_marks, frequencies, power = _split_scan_by_timeout(
        time_marks, frequencies, power, timeout=timeout
    )

    if colored:
        cmap = None
    else:
        cmap = 'gray'

    for ch, pwr_ch in power.items():
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)  # type: plt.Axes

        if level is None:
            level = PlotHelper.autolevel(np.ma.concatenate([val.ravel() for val in pwr_ch]))

        for tm, fr, pwr in zip(time_marks, frequencies, pwr_ch):
            tm = PlotHelper.pcolor_adjust_coordinates(tm)
            fr = PlotHelper.pcolor_adjust_coordinates(fr['MHz'])
            pcm = ax.pcolormesh(tm, fr.T, pwr.T, vmin=level[0], vmax=level[1], cmap=cmap)

        PlotHelper.set_time_ticks(ax, with_date=True)
        ax.set_xlim(PlotHelper.lim_daily(np.concatenate([tm for tm in time_marks])))

        title = lang_string({'ru': 'Мощность шума неба, Вт', 'en': 'Sky power, W'}, language)
        xlabel = lang_string({'ru': 'Время, UT', 'en': 'Time, UT'}, language)
        ylabel = lang_string({'ru': 'Частота, МГц', 'en': 'Frequency, MHz'}, language)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        plt.colorbar(pcm, cax=cax)
        fig.tight_layout()

        save_name = 'sky_power_ch{}'.format(ch)
        fig.savefig(save_folder / save_name)

        plt.close(fig)


@plot_languages()
def plot_calibration(calib_info: CalibrationInfo, save_folder: Path, colored=True,
                   figsize=FIGSIZE_ONELONG, language=None):
    frequencies = calib_info.frequencies
    central_frequencies = calib_info.central_frequencies
    gains = calib_info.gains
    channels = list(gains.keys())
    biases = calib_info.biases

    ch_label = {'en': 'Ch = {}', 'ru': 'Кан = {}'}[language]
    gain_title = {'en': 'Gain', 'ru': 'Усиление'}[language]
    bias_title = {'en': 'Bias', 'ru': 'Смещение'}[language]

    fig = plt.figure(figsize=figsize)
    xlim = (frequencies['MHz'][0], frequencies['MHz'][-1])
    for ch in channels:
        ax_gain = plt.subplot(211)  # type: plt.Axes
        ax_gain.plot(frequencies['MHz'], gains[ch], label=ch_label.format(ch))
        ax_gain.set_title(gain_title)
        ax_gain.legend()
        ax_gain.grid(True)
        ax_gain.set_xlim(*xlim)

        ax_bias = plt.subplot(212)  # type: plt.Axes
        ax_bias.plot(central_frequencies['MHz'], biases[ch], label=ch_label.format(ch))
        ax_bias.set_title(bias_title)
        ax_bias.legend()
        ax_bias.grid(True)
        ax_bias.set_xlim(*xlim)

    plt.tight_layout()

    fig.savefig(save_folder / 'calibration.png')


@plot_languages()
def plot_absolute_daily_spectra(calib_info: CalibrationInfo, save_folder: Path, colored=True,
                                figsize=FIGSIZE_ONELONG, timeout=dt.timedelta(minutes=10),
                                level=None, language=None):
    """
    Plot overall calibrated spectra for passive data.
    """
    time_marks = calib_info.time_marks
    freqs = calib_info.frequencies
    spectra = calib_info.calibrated_spectra

    time_marks, freqs, spectra = _split_scan_by_timeout(time_marks, freqs, spectra, timeout=timeout)

    for sp in spectra.values():
        for i in range(len(sp)):
            sp[i] = np.ma.array(sp[i], mask=np.isnan(sp[i]))

    if colored:
        cmap = None
    else:
        cmap = 'gray'

    for ch, spectrum in spectra.items():
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)  # type: plt.Axes

        if level is None:
            low, upp = PlotHelper.autolevel(np.ma.concatenate([sp.ravel() for sp in spectrum]))
        else:
            low, upp = level

        for tm, fr, sp in zip(time_marks, freqs, spectrum):
            tm = PlotHelper.pcolor_adjust_coordinates(tm)
            fr = PlotHelper.pcolor_adjust_coordinates(fr['MHz'])
            pcm = ax.pcolormesh(tm, fr.T, sp.T, vmin=low, vmax=upp, cmap=cmap)

        PlotHelper.set_time_ticks(ax, with_date=True)
        ax.set_xlim(PlotHelper.lim_daily(np.concatenate([tm for tm in time_marks])))

        title = lang_string({'ru': 'Принятая мощность, Вт',
                             'en': 'Received power, W'}, language)
        ax.set_title(title)
        xlabel = lang_string({'ru': 'Время, UT', 'en': 'Time, UT'}, language)
        ax.set_xlabel(xlabel)
        ylabel = lang_string({'ru': 'Частота, МГц', 'en': 'Frequency, MHz'},
                             language)
        ax.set_ylabel(ylabel)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        plt.colorbar(pcm, cax=cax)
        fig.tight_layout()

        save_name = 'calibrated_spectra_ch{}'.format(ch)
        fig.savefig(save_folder / save_name)

        plt.close(fig)


@plot_languages()
def plot_sun_pattern_vs_power(sun_pattern_info: SunPatternInfo, save_folder: Path, colored=True,
                              figsize=FIGSIZE_ONELONG, language=None):
    time_marks = sun_pattern_info.time_marks
    conv_pattern = sun_pattern_info.convolved_pattern
    channels = list(conv_pattern.keys())
    power = sun_pattern_info.max_power
    pattern = sun_pattern_info.pattern

    power_label = {'en': 'Receiver power', 'ru': 'Принятая мощность'}[language]
    conv_pattern_label = {'en': 'Convolved pattern', 'ru': 'Свертка с ДН'}[language]

    for ch in channels:
        fig = plt.figure(figsize=figsize)
        title = {'en': f'Received power vs convolved pattern (ch = {ch})',
                 'ru': f'Принятая мощность и свертка с ДН (кан = {ch})'}[language]

        ax_conv_pattern = plt.subplot(211)  # type: plt.Axes
        ax_conv_pattern.plot(time_marks, conv_pattern[ch], label=power_label, color='C0')
        ax_power = ax_conv_pattern.twinx()
        ax_power.plot(time_marks, power[ch], label=conv_pattern_label, color='C1')

        PlotHelper.set_time_ticks(ax_conv_pattern, with_date=True)

        ax_conv_pattern.set_title(title)
        ax_conv_pattern.legend()

        ax_pattern = plt.subplot(212)  # type: plt.Axes
        ax_pattern.plot(time_marks, pattern[ch])
        PlotHelper.set_time_ticks(ax_pattern, with_date=True)

        fig.tight_layout()

        save_name = f'sun_power_vs_pattern_ch{ch}'
        fig.savefig(save_folder / save_name)

        plt.close(fig)


@plot_languages()
def plot_sun_flux(sun_flux_info: SunFluxInfo, save_folder: Path, colored=True,
                  figsize=FIGSIZE_ONELONG, language=None):
    channels = sun_flux_info.channels
    time_marks = sun_flux_info.time_marks
    sun_flux = sun_flux_info.sun_flux

    watts2sfu = 10e22

    label = {'en': 'Ch = {}', 'ru': 'Кан = {}'}[language]
    xlabel = {'en': 'Time, UT', 'ru': 'Время, UT'}[language]
    ylabel = {'en': 'Sun flux, sfu', 'ru': 'Поток Солнца, sfu'}[language]

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)  # type: plt.Axes
    for ch in channels:
        ax.plot(time_marks, sun_flux[ch] * watts2sfu, label=label.format(label))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    PlotHelper.set_time_ticks(ax, with_date=True)

    plt.tight_layout()

    fig.savefig(save_folder / 'sun_flux.png')
