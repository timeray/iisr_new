"""
Functions for filtering noise.
"""
from datetime import datetime, timedelta
import numpy as np
from abc import ABCMeta, abstractmethod


__all__ = ['MeanStdFilter', 'MeanAdAroundMeanFilter',
           'MeanAdAroundMedianFilter', 'MedianAdAroundMedianFilter',
           'filter_outliers', 'OutlierTimeSeriesFilter']


class Filter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Filter array"""


class SigmaFilter(Filter, metaclass=ABCMeta):
    def __init__(self, n_sigma):
        self.n_sigma = n_sigma

    @abstractmethod
    def __call__(self, array: np.ndarray, axis: int = -1):
        """Filter array using threshold"""


class TimeSeriesFilter(Filter, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, dtimes: np.ndarray, array: np.ndarray, axis: int = -1):
        """Filter time series, defined by dtimes array"""


class MeanStdFilter(SigmaFilter):
    """
    General standard deviation filter.
    ML estimator for Gaussian distribution.
    """
    def __call__(self, array: np.ndarray, axis: int = -1):
        mean = np.ma.mean(array, axis=axis, keepdims=True)
        std = np.ma.std(array, axis=axis, keepdims=True)

        # Use ma.greater, to avoid raising warning
        mask = np.ma.greater(np.abs(array - mean), (std * self.n_sigma))
        if np.ma.is_masked(array):
            mask |= array.mask
        return np.ma.array(array, mask=mask)


class MeanAdAroundMeanFilter(SigmaFilter):
    """
    Implement mean absolute deviation around mean.
    A bit more robust then standard deviation.

    See Also: https://en.wikipedia.org/wiki/Average_absolute_deviation
    """

    def __call__(self, array: np.ndarray, axis: int = -1):
        mean = np.ma.mean(array, axis=axis, keepdims=True)
        abs_dev = np.abs(array - mean)
        mean_abs_dev = np.ma.mean(abs_dev, axis=axis, keepdims=True)

        # Use ma.greater, to avoid raising warning
        mask = np.ma.greater(abs_dev, mean_abs_dev * self.n_sigma)
        if np.ma.is_masked(array):
            mask |= array.mask
        return np.ma.array(array, mask=mask)


class MeanAdAroundMedianFilter(SigmaFilter):
    """
    Implement mean absolute deviation around median.
    ML estimator for Laplace distribution.

    See Also: https://en.wikipedia.org/wiki/Average_absolute_deviation
    """

    def __call__(self, array: np.ndarray, axis: int = -1):
        median = np.ma.median(array, axis=axis, keepdims=True)
        abs_dev = np.abs(array - median)
        mean_abs_dev = np.ma.mean(abs_dev, axis=axis, keepdims=True)

        # Use ma.greater, to avoid raising warning
        mask = np.ma.greater(abs_dev, mean_abs_dev * self.n_sigma)
        if np.ma.is_masked(array):
            mask |= array.mask
        return np.ma.array(array, mask=mask)


class MedianAdAroundMedianFilter(SigmaFilter):
    """
    Implement median absolute deviation around median.
    Very robust estimator. Unlike mean deviation estimators, it can not
    be affected by single infinite outlier.

    See Also: https://en.wikipedia.org/wiki/Average_absolute_deviation

    """
    def __call__(self, array: np.ndarray, axis: int = -1):
        median = np.ma.median(array, axis=axis, keepdims=True)
        abs_dev = np.abs(array - median)
        median_abs_dev = np.ma.median(abs_dev, axis=axis, keepdims=True)

        # Use ma.greater, to avoid raising warning
        mask = np.ma.greater(abs_dev, median_abs_dev * self.n_sigma)
        if np.ma.is_masked(array):
            mask |= array.mask
        return np.ma.array(array, mask=mask)


def get_filter(filter_name: str, threshold: float) -> SigmaFilter:
    if filter_name == 'Std':
        return MeanStdFilter(threshold)
    elif filter_name == 'MeanAdMean':
        return MeanAdAroundMeanFilter(threshold)
    elif filter_name == 'MeanAdMedian':
        return MeanAdAroundMedianFilter(threshold)
    elif filter_name == 'MedianAdMedian':
        return MedianAdAroundMedianFilter(threshold)
    else:
        raise ValueError('Unknown filter {}'.format(filter_name))


def filter_outliers(array: np.ndarray, noise_filter: SigmaFilter,
                    window: int = None, padding='same', pad_value=None
                    ) -> np.ndarray:
    """
    Remove outliers from input data.

    Run moving window where deviation of the data is estimated
    and used to threshold outliers.

    Parameters
    ----------
    array: 1-d np.ndarray
        Input data.
    noise_filter: SigmaFilter
        Filter.
    window: int or None, default None
        Size of estimation window. If None, set windows equal to array size.
    padding: 'same' or 'valid', default 'same'
        Padding mode. If 'same', result array will have same shape as input
        array. Missing values at edges of array will be padded with pad_value.
        If 'valid', only available data will be used for estimation.
    pad_value: number, 2-element tuple or None
        Value to pad with. Valid only for 'same' padding.
        If None, padding values will be set to mean of data in closest window.

    Returns
    -------
    array: 1-d masked np.ndarray
        Array with masked outliers.
    """
    # Check arguments
    if padding not in ['same', 'valid']:
        raise ValueError('Incorrect padding {}'.format(padding))

    if array.ndim != 1:
        raise ValueError('Input array should be one-dimensional')

    if window is not None and (window < 0 or not isinstance(window, int)):
        raise ValueError('Incorrect window')

    if window is None:
        window = array.size

    if window > array.size:
        raise ValueError('Window {} is bigger than array size {}'
                         ''.format(window, array.size))

    # Pad array to get correct shape
    n_pad_values = window - 1

    if pad_value is not None:
        if isinstance(pad_value, tuple):
            left_pad_value, right_pad_value = pad_value
        else:
            left_pad_value = pad_value
            right_pad_value = pad_value
    else:
        left_pad_value = np.nanmean(array[:window])
        right_pad_value = np.nanmean(array[-window:])

    # If pad value is float, array must be cast to float for correct results
    pad_is_float = isinstance(left_pad_value, float) \
                   or isinstance(right_pad_value, float)
    if pad_is_float and array.dtype == int:
        array = array.astype(float)

    left_pad = int(np.floor(n_pad_values / 2))
    right_pad = int(np.ceil(n_pad_values / 2))

    padded_arr = np.pad(array,
                        mode='constant',
                        pad_width=(left_pad, right_pad),
                        constant_values=(left_pad_value, right_pad_value))
    padded_arr = np.ma.array(padded_arr, mask=np.isnan(padded_arr))

    # Divide array by windows
    index_arr = np.arange(padded_arr.size)
    dtype_bytes = index_arr.dtype.itemsize
    new_shape = (array.size, window)
    new_strides = (dtype_bytes, dtype_bytes)
    strided_index_array = np.lib.stride_tricks.as_strided(index_arr,
                                                          shape=new_shape,
                                                          strides=new_strides)

    # Apply filter. Output array will have shape N x Length of window
    masked_array = noise_filter(padded_arr[strided_index_array], axis=1)
    # If sample is masked in any window, mark that sample as outlier
    # Note that sample index in each window changes
    # Get indexes of all outliers in padded array
    mask_indexes = np.unique(strided_index_array[np.where(masked_array.mask)])
    # Get indexes for original array
    mask_indexes -= left_pad
    mask_indexes = mask_indexes[(mask_indexes >= 0)
                                & (mask_indexes < array.size)]
    # Turn indexes to boolean mask
    # zeros_like also copies array mask, if it is masked
    outlier_mask = np.zeros_like(array, dtype=bool)
    outlier_mask[mask_indexes] = True
    return np.ma.array(array, mask=outlier_mask)


class OutlierTimeSeriesFilter(TimeSeriesFilter):
    def __init__(self, filter_type, threshold, window=None, timeout=None,
                 below_window_mode='reduce', padding='same', pad_value=None):
        """
        Remove outliers from input time series data.

        Moving window is used to estimate the data.
        Since time series data may have discontinuity, extra care must be taken,
        when filtering is performed. If gap between subsequent samples exceeds
        specified timeout, array will be split to continue estimation separately.

        Parameters
        ----------
        filter_type: str
            Name of outlier sigma filter.
        threshold: number
            Threshold (sigma of filter).
        window: int or None, default None
            Size of estimation window. If None, set windows equal to array size.
        timeout: timedelta or None
            Maximum time difference between two subsequent samples.
            If None, data will not be examined for discontinuity.
        below_window_mode: 'reduce' or 'skip'
            Defines how arrays with size below windows length are handled.
            If 'reduce', than window will be reduced to the size of array and
            regular filtering will be applied.
            If 'skip', than filtering will not be applied.
        padding: 'same' or 'valid', default 'same'
            Padding mode. If 'same', result array will have same shape as input
            array. Missing values at edges of array will be padded with pad_value.
            If 'valid', only available data will be used for estimation.
        pad_value: number or None
            Value to pad with. Valid only for 'same' padding.
            If None, padding values will be set to mean of data in closest window.
        """
        if window is not None and (not isinstance(window, int) or window <= 0):
            raise ValueError('Window should be positive integer')
        self.window = window

        if timeout is not None and not isinstance(timeout, timedelta):
            raise ValueError('Timeout should be timedelta or None')
        self.timeout = timeout

        if below_window_mode not in ['reduce', 'skip']:
            raise ValueError('Incorrect below_window_mode')
        self.below_window_mode = below_window_mode

        self.padding = padding
        self.pad_value = pad_value

        self.filter = get_filter(filter_type, threshold=threshold)

    def __call__(self, dtimes: np.ndarray, array: np.ndarray, axis: int = -1):
        """
        Apply filter to the input time series data.

        If input data is masked array, masked values will not be considered.

        Parameters
        ----------
        dtimes: np.ndarray
            Datetimes of input sequence.
        array: np.ndarray or np.MaskedArray
            Input sequence to filter.
        axis: int
            Axis

        Returns
        -------
        out: np.ma.MaskedArray
            Filtered array with masked outliers.
        """
        dtimes = np.asarray(dtimes, dtype=datetime)
        array = np.ma.asarray(array)

        if dtimes.size != array.size:
            raise ValueError('Length of input arrays should be equal')
        size = dtimes.size

        if self.window is None:
            window = size
        else:
            window = self.window

        if window > size:
            raise ValueError('Window {} is bigger than whole data length {}'
                             ''.format(window, size))

        # Search for discontinuity
        if array.mask.any():
            dtimes = np.ma.array(dtimes, mask=array.mask)

        if self.timeout is not None:
            if isinstance(dtimes, np.ma.MaskedArray):
                no_mask_indices = np.where(~dtimes.mask)[0][1:]
                compressed_indices = \
                    np.where(np.diff(dtimes.compressed()) > self.timeout)[0]
                timeout_indices = no_mask_indices[compressed_indices]
            else:
                timeout_indices = np.where(np.diff(dtimes) >= self.timeout)[0]
                timeout_indices += 1  # np.diff shift indices

            array_split = np.split(array, timeout_indices)
        else:
            array_split = [array]

        # Apply window filtering
        masked_arrays = []
        for arr in array_split:
            if window <= arr.size:
                out_array = filter_outliers(arr, self.filter, window=window,
                                            pad_value=self.pad_value,
                                            padding=self.padding)
            elif window > arr.size and self.below_window_mode == 'reduce':
                out_array = filter_outliers(arr, self.filter, window=arr.size,
                                            pad_value=self.pad_value,
                                            padding=self.padding)
            elif window > arr.size and self.below_window_mode == 'skip':
                out_array = arr
            else:
                raise RuntimeError('Unexpected error when apply filtering')

            masked_arrays.append(out_array)

        return np.ma.concatenate(masked_arrays)
