from collections import defaultdict
from pathlib import Path

import numpy as np
import datetime as dt

from typing import List, Dict, Union, Optional

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from iisr.postprocessing.passive import SourceTrackInfo
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
    fig = plt.figure(figsize=figsize)

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
        ax = plt.subplot(111)  # type: plt.Axes

        if level is None:
            level = PlotHelper.autolevel(
                np.ma.concatenate([sp.ravel() for sp in spectrum]),
            )

        for dt_, fr, sp in zip(dtimes, freqs, spectrum):
            dt_ = PlotHelper.pcolor_adjust_coordinates(dt_)
            fr = PlotHelper.pcolor_adjust_coordinates(fr['MHz'])
            pcm = ax.pcolormesh(dt_, fr.T, sp.T, vmin=level[0], vmax=level[1], cmap=cmap)

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

        plt.clf()


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
