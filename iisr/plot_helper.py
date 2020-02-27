import matplotlib
matplotlib.use('Agg')
from matplotlib import cm as mplcm
from matplotlib import pyplot as plt, colors
from matplotlib import dates as mdates
from datetime import datetime, timedelta
from iisr.time import tz_dict

import numpy as np
from typing import Sequence


def lim_daily(date, min_duration=60):
    if isinstance(date, np.ndarray):
        date = date.astype(datetime)

    if isinstance(date, Sequence) or isinstance(date, np.ndarray):
        if abs(date[1] - date[0]).total_seconds() > min_duration * 60:
            date1 = date[0]
        else:
            date1 = date[1]

        if abs(date[-1] - date[-2]).total_seconds() > min_duration * 60:
            date2 = date[-1]
        else:
            date2 = date[-2]

    else:
        date1 = date2 = date

    low_boundary = datetime(date1.year, date1.month, date1.day)
    date2 = datetime(date2.year, date2.month, date2.day)
    upp_boundary = date2 + timedelta(days=1)

    return low_boundary, upp_boundary


def pcolor_adjust_coordinates(coord_arr):
    diff = coord_arr[1] - coord_arr[0]
    coord_arr = coord_arr - diff / 2
    return np.concatenate((coord_arr, np.array([coord_arr[-1] + diff])))


def cycle_colors(ax, n_colors):
    cm = plt.get_cmap('jet')
    c_norm = colors.Normalize(vmin=0, vmax=n_colors - 1)
    scalar_map = mplcm.ScalarMappable(norm=c_norm, cmap=cm)
    clrs = [scalar_map.to_rgba(i) for i in range(n_colors)]
    ax.set_prop_cycle('color', clrs)


def autolevel(data, low_percent=5, upp_percent=95):
    if np.ma.is_masked(data):
        data = data.compressed()
    try:
        return np.percentile(data, low_percent),\
               np.percentile(data, upp_percent)
    except IndexError:
        return np.nan, np.nan


def panel(shape=None, number=None):
    if shape is None:
        # autoshape
        fsize = number
        ncols = 3
        nrows = int(np.ceil(fsize / ncols))
        shape = (nrows, ncols)

    fig, axes = plt.subplots(shape[0], shape[1], squeeze=False,
                             sharex='all', sharey='all', figsize=(10, 10))
    return fig, axes


def set_time_ticks(ax, time_zone=None, with_date=False, minticks=5,
                   maxticks=None, axis='x'):
    try:
        tz = tz_dict[time_zone]
    except KeyError:
        tz = None

    locator = mdates.AutoDateLocator(tz=tz, minticks=minticks,
                                     maxticks=maxticks)
    formatter = mdates.AutoDateFormatter(locator, tz=tz)
    if with_date:
        fmt_min = '%H:%M\n%d/%m'
        fmt_hour = '%Hh\n%d/%m'
    else:
        fmt_min = '%H:%M'
        fmt_hour = '%Hh'

    fmt_day = '%d/%m'

    formatter.scaled[1 / (24. * 60.)] = fmt_min
    formatter.scaled[1. / 24.] = fmt_hour
    formatter.scaled[1.] = fmt_day

    if axis == 'x':
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    elif axis == 'y':
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
    else:
        raise ValueError('Unexpected agrument axis: {}'.format(axis))


def auto_axes_lim(ax, bounds_x=None, bounds_y=None,
                  list_x=None, list_y=None):
    if bounds_x is not None and bounds_y is not None:
        ax.set_xlim(bounds_x)
        ax.set_ylim(bounds_y)

    if bounds_x is None and bounds_y is None:
        return

    if len(list_x) != len(list_y):
        raise ValueError('Length of list_x and list_y should be equal')

    if bounds_x is not None:
        lim1 = ax.set_xlim
        lim2 = ax.set_ylim
        low, upp = bounds_x
        list1 = list_x
        list2 = list_y

    else:
        lim1 = ax.set_ylim
        lim2 = ax.set_xlim
        low, upp = bounds_y
        list1 = list_y
        list2 = list_x

    # Choose minimum among minima and maximum among maxima
    lim1([low, upp])
    minima = []
    maxima = []
    for arr1, arr2 in zip(list1, list2):
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        mask = (arr1 >= low) & (arr1 <= upp)
        minima.append(np.nanmin(arr2[mask]))
        maxima.append(np.nanmax(arr2[mask]))

    lim2(min(minima), max(maxima))
