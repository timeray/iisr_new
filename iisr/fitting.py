"""
Functions used to apply linear regression on received noise.
"""
import logging
from datetime import datetime, timedelta

import lmfit
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit

from iisr.utils import central_time
from iisr.digital_filter import DigitalFilter


ESTIMATED_N_HUMPS = 11


def fit_gauss(x, value, a_min, a_max, b_min, b_max, var_max):
    def gauss(values: np.ndarray, pars: lmfit.Parameters):
        a = pars['a']
        b = pars['b']
        mean = pars['mean']
        var = pars['var']
        return a * np.exp(-(values - mean) ** 2 / (2 * var)) + b

    def gauss_res(pars: lmfit.Parameters, values: np.ndarray,
                  target: np.ndarray):
        return target - gauss(values, pars)

    params = lmfit.Parameters()
    params.add('a', np.max(value), min=a_min, max=a_max)
    params.add('b', np.mean(value), min=b_min, max=b_max)
    params.add('mean', x.mean(), min=x.min(), max=x.max())
    params.add('var', value.var(), min=0., max=var_max)

    args = (x, value)
    fitter = lmfit.Minimizer(gauss_res, params, fcn_args=args)
    res = fitter.minimize(method='least_squares')

    model_gauss = gauss(x, res.params)
    return res, model_gauss


def fit_regression(train, target, rescale=True):
    """
    Estimate coefficients a, b of linear regression target = a * train + b.

    Uses robust regression with cauchy loss.

    If input array have length 0, return nans.

    Parameters
    ----------
    train: np.ndarray with shape (N, )
    target: np.ndarray with shape (N, )
    rescale: bool, default True
        If True, rescales data to [0, 1] range. It may help the algorithm to converge.

    Returns
    -------
    a: float
        Slope of linear regression.
    b: float
        Bias of linear regression.
    cov: np.ndarray
        Covariance matrix.
    """
    if len(train) == 1:
        return np.nan, np.nan, np.full((2, 2), np.nan)

    # Eliminate nans
    not_nan_mask = ~(np.isnan(target) | np.isnan(train))
    train = train[not_nan_mask]
    target = target[not_nan_mask]

    # Scaling variables for stable fitting
    if rescale:
        train_max = train.max()
        train_min = train.min()
        train_range = train_max - train_min

        target_max = target.max()
        target_min = target.min()
        target_range = target_max - target_min

        train = (train - train_min) / train_range
        target = (target - target_min) / target_range

    try:
        (a, b), cov = curve_fit(lambda x, a, b: a * x + b,
                                train, target,
                                p0=(1., 0.5), method='dogbox', loss='cauchy',
                                bounds=[(0, 0), (np.inf, 1)],
                                f_scale=0.25)
    # Optimal parameters not found
    except RuntimeError:
        a = np.nan
        b = np.nan
        cov = np.full((2, 2), np.nan)

    # Rescale coefficients back
    if rescale:
        a = a * target_range / train_range
        b = b * target_range + target_min - a * train_min

    return a, b, cov


def window_fit(x, y, window=None, stride=None, ret_slices=False):
    """
    Fit linear regression y = a * x + b in subsequent windows.

    If window is specified, data will be divided by windows, where separate
    coefficients will be estimated. In this case, results will be arrays with
    size equal to number of windows.

    Parameters
    ----------
    x: np.ndarray with shape (N, )
        Model array.
    y: np.ndarray with shape (N, )
        Target array.
    window: int or None, default None
        Length of estimation window. If None, uses all available data to fit
        parameters. In this case, single values are returned for gain and bias.
    stride: int or None
        Stride of window. If None or zero, no striding will be applied.
    ret_slices: bool, default False
        Whether to return list of slices of resultant windows.

    Returns
    -------
    a: np.ndarray or float
        Estimated line slope.
    b: np.ndarray or float
        Estimated bias.
    cov: np.ndarray with shape (N, 2, 2), where N is number of windows
        Covariance matrix.
    slices: list, optional
        List of slices for corresponding window. Only provided if `ret_slices`
        is True.
    """
    if stride is not None and stride != 0:
        raise NotImplementedError()

    if x.shape != y.shape:
        raise ValueError('Sky power and noise power should have equal shapes')

    if x.ndim > 1:
        raise ValueError('Input data should be one-dimensional')

    if x.size == 0:
        if ret_slices:
            return np.nan, np.nan, np.full((2, 2), np.nan), []
        else:
            return np.nan, np.nan, np.full((2, 2), np.nan)

    if window is None:
        window = x.size

    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError('Window should be positive integer')

    size = x.size

    if window > size:
        raise ValueError('Window ({}) is bigger than input data ({})'
                         ''.format(window, size))

    n_wins = size // window

    a = np.empty((n_wins,), dtype=float)
    b = np.empty((n_wins,), dtype=float)
    cov_matrix = np.empty((n_wins, 2, 2), dtype=float)
    slices = []

    for i in range(n_wins):
        sl = slice(i * window, (i+1) * window)
        slices.append(sl)
        fit_res = fit_regression(x[sl], y[sl])
        a[i] = fit_res[0]
        b[i] = fit_res[1]
        cov_matrix[i] = fit_res[2]

    # Return scalar if array has one element
    if n_wins == 1:
        result = [a.item(), b.item(), cov_matrix.squeeze()]
    else:
        result = [a, b, cov_matrix]

    if ret_slices:
        result.append(slices)

    return tuple(result)


def fit_sky_noise_1d(dtimes, sky_power, noise_power, window=None, timeout=None,
                     below_window_mode='skip'):
    """
    Fit sky noise power with actual data to estimate receiver bias and gain.
    Perform linear regression in specified window. It is assumed that
    provided data come from single frequency.

    Check if gap in the input time sequence exceeds timeout. If it does,
    input arrays will be split for each found gap in dtimes.

    After splitting the array according to timeout, regression will be
    estimated on the arrays.

    If splitting or windowing was not applied, results will be squeezed.

    Parameters
    ----------
    dtimes: sequence of length N
        Array of datetimes.
    sky_power: np.ndarray with shape (N, )
        Sky power.
    noise_power: np.ndarray with shape (N, )
        Noise power.
    window: int or None
        Length of estimation window. If None, uses all available points.
        Note that if timeout is specified, some arrays after splitting may
        size less than window. See below_window_mode to specify actions for
        this case.
    timeout: timedelta or None, default None
        If None then data would not be checked against timeout gaps.
    below_window_mode: 'reduce', 'skip' or 'raise', default 'skip'
        Defines how arrays with size below windows length are handled.
        If 'reduce', than window will be reduced to the size of array and
        regular fitting will be applied.
        If 'skip', than fitting will not be applied.
        If 'raise', raises values exception.

    Returns
    -------
    dtimes: np.ndarray or datetime
        Datetimes where gain and bias were estimated. It central datetime
        between beginning and end of the window.
    gain: np.ndarray or float
        Estimated receiver gain.
    bias: np.ndarray or float
        Estimated receiver bias.
    cov: np.ndarray with shape (M, 2, 2), where M is a number of windows
        Covariance matrix.
    """
    dtimes = np.array(dtimes, dtype=datetime)

    # Check inputs for validness to prevent any unexpected results
    if len(dtimes) != len(sky_power) or len(sky_power) != len(noise_power):
        raise ValueError('Input array should have equal length.')

    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError('Window should be positive integer')

    if timeout is not None and not isinstance(timeout, timedelta):
        raise ValueError('Timeout should be specified as datetime.timedelta')

    if below_window_mode not in ['reduce', 'skip', 'raise']:
        raise ValueError('Incorrect below_window_mode: {}'
                         ''.format(below_window_mode))

    # Exclude nans from the data
    nan_mask = np.isnan(sky_power) | np.isnan(noise_power)
    dtimes = dtimes[~nan_mask]
    sky_power = sky_power[~nan_mask]
    noise_power = noise_power[~nan_mask]

    size = len(dtimes)

    # Check window size
    if window is not None and ((window > size) and (below_window_mode == 'raise')):
        raise ValueError('Window is bigger than data')

    # Split input sequences if there are gaps in time series
    if timeout is not None:
        timeout_indices = np.where(np.abs(np.diff(dtimes)) >= timeout)[0] + 1
        dtimes_split = np.split(dtimes, timeout_indices)
        sky_power_split = np.split(sky_power, timeout_indices)
        noise_power_split = np.split(noise_power, timeout_indices)
    else:
        dtimes_split = [dtimes]
        sky_power_split = [sky_power]
        noise_power_split = [noise_power]

    # Perform estimation of parameters in windows
    zipped_splits = zip(dtimes_split, sky_power_split, noise_power_split)

    new_dtimes_list = []
    gain_splits = []
    bias_splits = []
    cov_splits = []
    for dt, sky_pwr, noise_pwr in zipped_splits:
        # If arrays are less than the window
        if window is not None and len(dt) < window:
            # Squeeze the window
            if below_window_mode == 'reduce':
                split_window = len(dt)
            # Do nothing. Return nans later if all splits were skipped
            elif below_window_mode == 'skip':
                continue
            elif below_window_mode == 'raise':
                raise ValueError('Window is bigger than data')
            else:
                raise RuntimeError('Unexpected below_window_mode')
        else:
            split_window = window

        gain, bias, cov, slices = window_fit(
            sky_pwr, noise_pwr, window=split_window, ret_slices=True
        )

        gain_splits.append(gain)
        bias_splits.append(bias)
        cov_splits.append(cov)

        # Find datetime that corresponds to estimated parameters
        new_dtimes_list.extend([central_time(dt[sl]) for sl in slices])

    new_dtimes = np.array(new_dtimes_list, dtype=datetime)

    # This may happen with 'skip' below_window_mode
    if not gain_splits:
        return central_time(dtimes), np.nan, np.nan, np.full((2, 2), np.nan)

    # Transform outputs to same shape and concatenate results
    # With np.r_ we generalize concatenation on zero dimension outputs
    gains = np.r_[tuple(gain_splits)]
    biases = np.r_[tuple(bias_splits)]
    covs = np.r_[tuple(['0,3'] + cov_splits)]  # M x 2 x 2

    # Replace all 0 gains with nans because it is not physical value
    zero_mask = np.isclose(gains, 0.)
    gains[zero_mask] = np.nan
    biases[zero_mask] = np.nan
    covs[zero_mask] = np.full((2, 2), np.nan)

    # If arrays have one element, return scalars and squeeze cov matrix
    if len(new_dtimes) == 1:
        new_dtimes = new_dtimes.item()
        gains = gains.item()
        biases = biases.item()
        covs = covs.squeeze()

    return new_dtimes, gains, biases, covs


def simple_gain(freqs, freq0, a, var, frequency_response=None):
    """Models gaussian gain:
    G = a * h(f) * exp(-(f-f0)^2 / (2 * var))

    Parameters
    ----------
    freqs: np.ndarray
        Frequencies where gain should be estimated, kHz.
    freq0: number
        Central frequency, kHz.
    a: number
        Amplitude of gaussian.
    var: number
        Variance of gaussian.
    frequency_response: np.ndarray or None, default None
        Additional frequency response of the filter. Response should be wrt
        amplitude. If None, no additional response will be added.

    Returns
    -------
    gain: np.ndarray or float
        Gain.
    """
    freqs_diff = freqs - freq0
    gain = a * np.exp(-(freqs_diff ** 2) / (2 * var))

    if frequency_response is None:
        return gain
    else:
        return frequency_response ** 2 * gain


def oscillating_gain(freqs, freq0, n_humps, a, b, phase, var, frequency_response=None):
    """
    Gaussian with oscillations. Sum of gaussian and cosine with amplitude that
    have same gaussian shape.

    g(f) = a * G(f, w, var) + b * sin(w*(f-f0) + phase) * G(f, w, var)
    where G(f, w, var) = exp(-(f-f0)^2 / (2 * var))

    Parameters
    ----------
    freqs: np.ndarray
        Frequencies where gain should be estimated, kHz.
    freq0: number
        Central frequency, kHz.
    n_humps: number
        Number of oscillation humps.
    a: number
        Amplitude of gaussian.
    b: number
        Amplitude of oscillations.
    phase: number
        Phase bias of cosine.
    var: number
        Variance of gaussian.
    frequency_response: np.ndarray or None, default None
        Additional frequency response of the filter. Response should be wrt
        amplitude. If None, no additional response will be added.

    Returns
    -------
    gain: np.ndarray or float
        Gain.
    """
    freqs_diff = freqs - freq0
    gauss = np.exp(-(freqs_diff ** 2) / (2 * var))

    if freqs_diff.size >= 2:
        omega = (2 * np.pi * n_humps) / freqs_diff.ptp()
    else:
        omega = 0.
    oscillating_gauss = gauss * (a + b * np.cos(omega * freqs_diff + phase))

    if frequency_response is None:
        return oscillating_gauss
    else:
        return frequency_response**2 * oscillating_gauss


def model_residual(pars: lmfit.Parameters, sky_noise: np.ndarray,
                   observed_noise: np.ndarray, freqs: np.ndarray,
                   weights: np.ndarray = None,
                   frequency_response=None,
                   gain_with_oscillations=False):
    """
    Residual for model with gain g(f):

    residual = observed_noise - g(f) * (sky_noise + bias)

    Parameters
    ----------
    pars: lmfit.Parameters
        Input vaiable parameters.
    sky_noise: np.ndarray or np.ma.MaskedArray of shape (M, N)
        Sky power for each frequency for each time bin.
    observed_noise: np.ndarray or np.ma.MaskedArray of shape (M, N)
        Observed power for each frequency and time bin.
    freqs: np.ndarray of shape (N, )
        Frequencies. Units must match with freq0.
    weights: np.ndarray or None, default None
        Weights for residuals. If None, no weighting will be applied.
        Array should broadcast to (M, N) input shape.
    frequency_response: np.ndarray or None, default None
        Additional frequency response of the filter. Response should be wrt
        amplitude. If None, no additional response will be added.
    gain_with_oscillations: bool
        Whether gain should be modelled with oscillations.

    Returns
    -------
    residual: one-dimensional np.ndarray
        Residual array. Masked values will be dropped.
    """
    pars = pars.valuesdict()
    if gain_with_oscillations:
        gain = oscillating_gain(freqs, pars['freq0'], pars['n_humps'],
                                pars['a'], pars['b'],
                                pars['phase'], pars['var'],
                                frequency_response=frequency_response)
    else:
        gain = simple_gain(freqs, pars['freq0'], pars['a'], pars['var'],
                           frequency_response=frequency_response)

    model = gain * (sky_noise + pars['bias'])

    if weights is None:
        residuals = model - observed_noise
    else:
        residuals = (model - observed_noise) / weights

    # Now all 2-d relations gone and we can flatten our array
    # All masked values of masked arrays also should be eliminated
    residuals = np.ma.compressed(residuals)

    return residuals


def fit_noise(sky_noise, observed_noise, freqs, freq0,
              vary_freq0=False,
              optimization_method='least_squares',
              optimization_kwargs=None,
              frequency_response=None,
              gain_with_oscillations=False,
              n_humps=None, n_humps_range=(0, 20),):
    """
    Fit two-dimensional model:

    y(f, t) = g(f) * (n(f, t) + bias),

    where g(f) - gaussian gain.

    Parameters
    ----------
    sky_noise: np.ndarray or np.ma.MaskedArray of shape (M, N)
        Sky power for each time bin and frequency.
    observed_noise: np.ndarray or np.ma.MaskedArray of shape (M, N)
        Observed power for each time bin and frequency.
    freqs: np.ndarray of shape (N, )
        Frequencies. Units should match with freq0.
    freq0: number
        Central frequency.
        It must be within frequencies range and have same units.
    vary_freq0: bool, default False
        If True, make central frequency a variable parameter with freq0 init.
    optimization_method: str, default 'least_squares'
        Name of optimization method.
    optimization_kwargs: dict or None, default None
        Parameters of optimization.
    frequency_response: np.ndarray or None, default None
        Additional frequency response of the filter. Response should be wrt
        amplitude. If None, no additional response will be added.
    gain_with_oscillations: bool
        Whether gain should be modelled with oscillations.
    n_humps: number or None, default None
        Number of oscillation humps in resulting gaussian.
        If None, tries to estimate best n_humps in the n_humps_range.
    n_humps_range: tuple of ints, default (0, 20)
        Search range for best n_humps parameter. Only valid when n_humps=None.

    Returns
    -------
    out: lmfit Minimizer results
        Contain parameters (params) and fit statistics.
    """
    # Check arguments
    if not isinstance(freqs, np.ndarray):
        raise ValueError('Frequencies must be a numpy array')

    if not isinstance(sky_noise, np.ndarray):
        raise ValueError('Sky noise must be a numpy array')

    if not isinstance(observed_noise, np.ndarray):
        raise ValueError('Observed noise must be a numpy array')

    if sky_noise.shape[1] != freqs.size or sky_noise.ndim != 2:
        raise ValueError('Sky noise array must have (M, N) shape '
                         'where N is a number of frequencies')

    if observed_noise.shape != sky_noise.shape:
        raise ValueError('Observed noise array must have (M, N) shape '
                         'where N is a number of frequencies')

    if (freqs <= 0.).any() or freq0 <= 0:
        raise ValueError('Frequencies must be positive')

    if len(freqs) < 2:
        raise ValueError('Number of input frequencies should be at least 2')

    if optimization_kwargs is None:
        optimization_kwargs = {}

    if gain_with_oscillations and n_humps is None:
        err_msg = 'n_humps_range should be tuple of two integers'
        if not isinstance(n_humps_range, tuple):
            raise ValueError(err_msg)

        if len(n_humps_range) != 2:
            raise ValueError(err_msg)

        if not isinstance(n_humps_range[0], int) or not isinstance(n_humps_range[1], int):

            raise ValueError(err_msg)

    # Scaling variables for stable fitting
    x_scale = np.nanmax(sky_noise) - np.nanmin(sky_noise)
    y_scale = np.nanmax(observed_noise) - np.nanmin(observed_noise)

    if x_scale == 0.:
        x_scale = 1.

    if y_scale == 0.:
        y_scale = 1.

    sky_noise = sky_noise / x_scale
    observed_noise = observed_noise / y_scale

    # Initialize parameters
    params = lmfit.Parameters()
    params.add('a', 1e-3, min=0., max=100.)
    params.add('var', freqs.ptp() * 0.1, min=0., max=freqs.ptp()**2)
    params.add('bias', 1e-3, min=0., max=100.)
    params.add('freq0', freq0, min=0., max=freqs.max()*1.5, vary=vary_freq0)

    if gain_with_oscillations:
        params.add('b', 1e-3, min=0., max=100.)
        params.add('phase', 0., min=-np.pi/2, max=np.pi/2)
        params.add('n_humps', n_humps, vary=False, min=0., max=20.)

    args = (sky_noise, observed_noise, freqs, None, frequency_response, gain_with_oscillations)

    def loss(fit_results):
        return fit_results.chisqr

    def optimize(params, opt_method=optimization_method, **method_kwargs):
        params = params.copy()
        fitter = lmfit.Minimizer(model_residual, params,
                                 fcn_args=args, nan_policy='raise',
                                 **method_kwargs)
        fit_results_zero_phase = fitter.minimize(method=opt_method)

        if gain_with_oscillations:
            fit_loss_zero_phase = loss(fit_results_zero_phase)
            params['phase'].set(value=np.pi, min=np.pi/2, max=3 * np.pi / 2)
            fitter = lmfit.Minimizer(model_residual, params, fcn_args=args, nan_policy='raise')
            fit_results_pi_phase = fitter.minimize(method=opt_method)
            fit_loss_pi_phase = loss(fit_results_pi_phase)

            if fit_loss_zero_phase < fit_loss_pi_phase:
                fit_results_zero_phase.params['phase'].set(min=-np.pi, max=np.pi)
                return fit_results_zero_phase
            else:
                phase = fit_results_pi_phase.params['phase']
                if phase > np.pi:
                    new_phase = phase - 2 * np.pi
                    fit_results_pi_phase.params['phase'].set(value=new_phase, min=-np.pi, max=np.pi)
                return fit_results_pi_phase
        else:
            return fit_results_zero_phase

    if gain_with_oscillations and n_humps is None:
        # Search for best fit n_humps, otherwise, given n_humps will be used
        logging.info('perform search for best-fit n_humps')
        best_loss = None
        best_fit = None
        params['n_humps'].set(vary=True)
        step = 1.
        for i in np.arange(n_humps_range[0], n_humps_range[1], step):
            params['n_humps'].set(min=i, max=i+step, value=i+step/2)
            fit_res = optimize(params, loss='huber', **optimization_kwargs)
            fit_loss = loss(fit_res)
            if best_loss is None or best_loss > fit_loss:
                best_loss = fit_loss
                best_fit = fit_res
        logging.info(f'n_humps search finished with {best_fit.params["n_humps"].value:.2f} '
                     f'best-fit value')
        fit_res = best_fit
    else:
        fit_res = optimize(params, **optimization_kwargs)

    # Rescale coefficients back
    a = fit_res.params['a']
    bias = fit_res.params['bias']
    a.set(value=a * y_scale / x_scale, min=0., max=np.inf)
    bias.set(value=bias * x_scale, min=0., max=np.inf)

    if gain_with_oscillations:
        b = fit_res.params['b']
        b.set(value=b * y_scale / x_scale, min=0., max=np.inf)

    return fit_res


def fit_sky_noise_2d(dtimes, freqs, central_freq, sky_power, noise_power, *,
                     window=None, final_window=True,
                     timeout=None, below_window_mode='skip',
                     vary_freq0=False,
                     gain_with_oscillations=False,
                     n_humps=ESTIMATED_N_HUMPS,
                     n_humps_search_range=None):
    """
    Perform fitting of noise power to sky_power over range of times and
    frequencies. Uses complex model that accounts for oscillations of gain.

    Unlike fit_sky_noise_1d, fitting is conducted also over frequencies and,
    therefore, data is divided by blocks, where each datetime correspond
    to whole range of frequencies.

    Check if gap in the input time sequence exceeds timeout. If it does,
    input arrays will be split for each found gap in dtimes.

    Parameters
    ----------
    dtimes: sequence of length N
        Array of datetimes.
    freqs: sequence of length M
        Array of frequencies.
    central_freq: number
        Central frequency for oscillating gaussian.
    sky_power: np.ndarray with shape (N, M)
        Sky power.
    noise_power: np.ndarray with shape (N, M)
        Noise power.
    window: int or None
        Length of estimation time window. If None, uses all available points.
        Note that if timeout is specified, some arrays after splitting may
        size less than window. See below_window_mode to specify actions for
        this case.
    final_window: bool, default True
        Whether to use final data points to estimate parameters as a last
        window.
    timeout: timedelta or None, default None
        If None then data would not be checked against timeout gaps.
    below_window_mode: 'reduce', 'skip' or 'raise', default 'skip'
        Defines how arrays with size below windows length are handled.
        If 'reduce', than window will be reduced to the size of array and
        regular fitting will be applied.
        If 'skip', than fitting will not be applied.
        If 'raise', raises values exception.
    vary_freq0: bool, default False
        If True, make central frequency a variable parameter with freq0 init.
    n_humps_search_range: tuple of ints or None, default None
        If None constant ESTIMATE_N_HUMPS will be used as n_humps.
        If not None, best-fit n_humps will be chosen from given range.

    Returns
    -------
    dtimes: np.ndarray of shape (C, )
        Datetimes where gain and bias were estimated. It central datetime
        between beginning and end of the window.
    gain: np.ndarray of shape (C, M)
        Estimated receiver gain.
    fit_params: Dict[np.array]
        Fitting parameters for each window.
    """
    dtimes = np.array(dtimes, dtype=datetime)
    freqs = np.array(freqs, dtype=float)

    # Check inputs for validness to prevent any unexpected results
    shape = (dtimes.size, freqs.size)

    if shape[0] == 0 or shape[1] == 0:
        raise ValueError(f'Input array is empty (n_dtimes: {shape[0]} n_freqs: {shape[1]})')

    if sky_power.shape != shape or noise_power.shape != shape:
        raise ValueError('Input arrays must have (N, M) shape, '
                         'where N - number of dtimes, M - number of frequencies')

    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError('Window should be positive integer')

    if timeout is not None and not isinstance(timeout, timedelta):
        raise ValueError('Timeout should be specified as datetime.timedelta')

    if below_window_mode not in ['reduce', 'skip', 'raise']:
        raise ValueError(f'Incorrect below_window_mode: {below_window_mode}')

    # Check window size
    if window is not None and ((window > shape[1]) and (below_window_mode == 'raise')):
        raise ValueError('Window is bigger than data')

    # Split input sequences if there are gaps in time series
    if timeout is not None:
        timeout_indices = np.where(np.abs(np.diff(dtimes)) >= timeout)[0] + 1
        dtimes_split = np.split(dtimes, timeout_indices)
        sky_power_split = np.split(sky_power, timeout_indices, axis=0)
        noise_power_split = np.split(noise_power, timeout_indices, axis=0)
    else:
        dtimes_split = [dtimes]
        sky_power_split = [sky_power]
        noise_power_split = [noise_power]

    # Calculate response of the IISR digital filter
    digital_filter = DigitalFilter()
    iisr_response = digital_filter(freqs - central_freq)

    new_dtimes_list = []
    gain_splits = []
    fit_params = defaultdict(list)

    # Perform estimation of parameters for each split
    zipped_splits = zip(dtimes_split, sky_power_split, noise_power_split)
    for dt, sky_pwr, noise_pwr in zipped_splits:
        split_shape = sky_pwr.shape

        # If arrays are less than the window
        if window is None:
            split_window = len(dt)
        elif window is not None and len(dt) < window:
            # Squeeze the window
            if below_window_mode == 'reduce':
                split_window = len(dt)
            # Do nothing. Return nans later if all splits were skipped
            elif below_window_mode == 'skip':
                continue
            elif below_window_mode == 'raise':
                raise ValueError('Window is bigger than data')
            else:
                raise RuntimeError('Unexpected below_window_mode')
        else:
            split_window = window

        # Perform window splitting. If include final_window, then
        # we need one more window for last points
        if final_window:
            n_wins = int(np.ceil(split_shape[0] / split_window))
        else:
            n_wins = split_shape[0] // split_window

        gain = np.empty((n_wins, shape[1]), dtype=float)

        for i in range(n_wins):
            # Account for possible final window
            win_end = min((i + 1) * split_window, split_shape[0])
            sl = slice(win_end - split_window, win_end)

            if gain_with_oscillations:
                # Estimate number of humps in the data by finding peaks
                if n_humps_search_range == 'auto':
                    mean_noise = noise_pwr[sl].mean(axis=0)
                    diff = np.diff(mean_noise)
                    max_peaks_args = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
                    min_peaks_args = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1

                    arg_diff_estimate = np.mean(
                        [np.median(np.diff(max_peaks_args)),
                         np.median(np.diff(min_peaks_args))]
                    )

                    n_humps = freqs.size / arg_diff_estimate

                    if np.isnan(n_humps):
                        n_humps = ESTIMATED_N_HUMPS
                        logging.info(f'n_humps estimation failed, use: {n_humps:.2f} constant')
                    else:
                        logging.info(f'n_humps estimate: {n_humps:.2f}')
                elif n_humps_search_range is not None:
                    n_humps = None

            # Fit the model
            fit_res = fit_noise(sky_pwr[sl], noise_pwr[sl], freqs,
                                freq0=central_freq,
                                n_humps=n_humps,
                                n_humps_range=n_humps_search_range,
                                vary_freq0=vary_freq0,
                                frequency_response=iisr_response,
                                gain_with_oscillations=gain_with_oscillations)
            pars = fit_res.params.valuesdict()

            if gain_with_oscillations:
                gain[i] = oscillating_gain(freqs, freq0=pars['freq0'],
                                           n_humps=pars['n_humps'],
                                           a=pars['a'], b=pars['b'],
                                           phase=pars['phase'], var=pars['var'],
                                           frequency_response=iisr_response)
            else:
                gain[i] = simple_gain(freqs, freq0=pars['freq0'], a=pars['a'], var=pars['var'],
                                      frequency_response=iisr_response)

            fit_params['bias'].append(pars['bias'])
            fit_params['freq0'].append(pars['freq0'])
            fit_params['a'].append(pars['a'])
            fit_params['var'].append(pars['var'])

            if gain_with_oscillations:
                fit_params['b'].append(pars['b'])
                fit_params['n_humps'].append(pars['n_humps'])
                fit_params['phase'].append(pars['phase'])

            # Find datetime that corresponds to estimated parameters
            new_dtimes_list.append(central_time(dt[sl]))

        gain_splits.append(gain)

    new_dtimes = np.array(new_dtimes_list, dtype=datetime)

    for key in fit_params:
        fit_params[key] = np.array(fit_params[key])

    # Transform outputs to same shape and concatenate results
    # With np.r_ we generalize concatenation on zero dimension outputs
    gains = np.r_[tuple(gain_splits)]

    # If for some dtime gains at all frequencies close to 0, set this column
    # to nans because such behavior is not physical
    zero_mask = np.isclose(gains, 0.).all(axis=1)
    gains[zero_mask] = np.nan

    return new_dtimes, gains, fit_params
