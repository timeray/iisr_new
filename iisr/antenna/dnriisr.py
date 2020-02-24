"""
Wrapper for C-functions calculating antenna pattern for Irkutsk Incoherent
Scatter Radar (IISR).
"""

from pathlib import Path
import ctypes as ct
from typing import Union

from scipy.integrate import simps
from functools import lru_cache
import numpy as np
import sys
from scipy.constants import speed_of_light
from tqdm import tqdm

# Constants
IISR_LON = np.deg2rad(103.255)
IISR_LAT = np.deg2rad(52.875)
IISR_HEIGHT = 500.
TEMPERATURE_SENSETIVITY = 0.01  # degree per Celcius degree

_dir_path = Path(__file__).parent
if sys.platform.startswith('win'):
    dll = ct.CDLL(str(_dir_path / 'DNRIISR.dll'))
elif sys.platform.startswith('linux'):
    dll = ct.CDLL(str(_dir_path / 'DNRIISR.so'))
else:
    raise OSError(f'Cannot load shared library for platform {sys.platform}')

dll.InitializationParamDNR()

# Type definitions
dll.CalcDNR.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_int]
dll.CalcDNR.restype = ct.c_double

dll.MonostaticDNR.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_int, ct.c_int]
dll.MonostaticDNR.restype = ct.c_double

dll.Topoc_to_Ant.argtypes = [ct.c_double, ct.c_double,
                             ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
dll.Topoc_to_Ant.restype = None

dll.Ant_to_Topoc.argtypes = [ct.c_double, ct.c_double,
                             ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
dll.Ant_to_Topoc.restype = None

dll.AzimMaxDNR.argtypes = [ct.c_double]
dll.AzimMaxDNR.restype = ct.c_double

dll.AzimMaxDNR_model.argtype = [ct.c_double]
dll.AzimMaxDNR_model.restype = ct.c_double


def _choose_type(t):
    if t == 'both':
        return 0
    elif t == 'upper':
        return 1
    elif t == 'lower':
        return 2
    elif t != 'both':
        raise ValueError('Available types are "upper", "lower", "both"')


# === Wrapped C-functions ===
def calc_pattern(freq, epsilon, gamma, two_way=False, dnr_type='both',
                 tr_type='both', rc_type='both', temperature=None):
    """
    Calculate antenna pattern (power).

    Args:
        freq: float
            Frequency, MHz.
        epsilon: float or NxM ndarray
            Epsilon antenna coordinate.
        gamma: float or NxM ndarray
            Gamma antenna coordinate.
        tr_type: {'both', 'upper', 'lower'}; default='both'
            Specify horn used for transmission. 'both' when both horns used.
        rc_type: {'both', 'upper', 'lower'}; default='both'
            Specify horn used for reception. 'both' when both horns used.
        dnr_type: {'both', 'upper', 'lower'}; default='both'
            Specify horn used for calculation of one-way pattern (see two_way).
            'both' when both horns used.
        two_way: bool; default=False
            Whether to calculate two way power antenna pattern (transmission
            pattern times reception pattern) or to calculate pattern specified
            at dnr_type. In former case dnr_type will be ignored, in latter
            case tr_type and rc_type will be ignored.
        temperature: float or ndarray, optional
            Ambient atmospheric temperature in Celcius degrees.

    Returns:
        out: double or NxM ndarray
            Antenna pattern at specified direction.
    """
    if temperature is not None:
        epsilon += np.deg2rad(TEMPERATURE_SENSETIVITY * temperature)

    if isinstance(epsilon, float) and isinstance(gamma, float):
        if two_way:
            tr_t = _choose_type(tr_type)
            rc_t = _choose_type(rc_type)
            return dll.MonostaticDNR(freq, epsilon, gamma, tr_t, rc_t)
        else:
            dnr_t = _choose_type(dnr_type)
            return dll.CalcDNR(freq, epsilon, gamma, dnr_t)

    else:
        epsilon, gamma = np.broadcast_arrays(epsilon, gamma)
        out_shape = epsilon.shape
        if len(epsilon.shape) == 1:
            epsilon = epsilon[:, np.newaxis]
            gamma = gamma[:, np.newaxis]
        shape = epsilon.shape

        rows, cols = shape

        eps_arr = ct.c_double * (cols * rows)
        gam_arr = ct.c_double * (cols * rows)

        eps = eps_arr()
        gam = gam_arr()

        for i in range(rows):
            for j in range(cols):
                eps[i*cols + j] = epsilon[i, j]
                gam[i*cols + j] = gamma[i, j]

        pattern_arr = ct.c_double * (cols * rows)
        pattern = pattern_arr()

        if two_way:
            dll.MonostaticDNR_arrays.argtypes = [
                ct.c_double, eps_arr, ct.c_int, gam_arr,
                ct.c_int, ct.c_int, ct.c_int, pattern_arr]
            dll.MonostaticDNR_arrays.restype = None

            tr_t = _choose_type(tr_type)
            rc_t = _choose_type(rc_type)
            dll.MonostaticDNR_arrays(freq, eps, rows, gam, cols,
                                     tr_t, rc_t, pattern)

        else:
            dll.CalcDNR_arrays.argtypes = [
                ct.c_double, eps_arr, ct.c_int, gam_arr,
                ct.c_int, ct.c_int, pattern_arr]
            dll.CalcDNR_arrays.restype = ct.c_int

            dnr_t = _choose_type(dnr_type)
            dll.CalcDNR_arrays(freq, eps, rows, gam, cols, dnr_t, pattern)

        arr = np.array(pattern)
        arr[np.isnan(arr)] = 0
        return arr.reshape(out_shape)


@np.vectorize
def topoc2ant_c_wrapped(el, az):
    """
    Convert topocentric coordinates to antenna coordinates.
    C-wrapped pseudo-vectorized function.
    It is recommended to use topoc2ant.

    Parameters
    ----------
    el: number or NumPy array
        Elevation angle, rad
    az: number or NumPy array
        Azimuth angle, rad

    Return
    ------
    gamma, epsilon [rad]
    """
    gam = ct.c_double()
    ep = ct.c_double()
    dll.Topoc_to_Ant(el, az, ct.byref(gam), ct.byref(ep))
    return gam.value, ep.value


@np.vectorize
def ant2topoc_c_wrapped(gam, ep):
    """
    Convert antenna coordinates to topocentric coordinates.
    C-wrapped pseudo-vectorized function.
    It is recommended to use ant2topoc.

    Parameters
    ----------
    gam: number or NumPy array
        Gamma antenna coordinate, rad
    ep: number or NumPy array
        Epsilon antenna coordinate, rad

    Return
    ------
    elevation, azimuth [rad]
    """
    el = ct.c_double()
    az = ct.c_double()
    dll.Ant_to_Topoc(ep, gam, ct.byref(el), ct.byref(az))
    return el.value, az.value


@np.vectorize
def ant_max_direction_c_wrapped(freq_khz):
    return dll.AzimMaxDNR(ct.c_double(freq_khz))


@np.vectorize
def ant_max_direction_model_c_wrapped(freq_khz):
    return dll.AzimMaxDNR_model(ct.c_double(freq_khz))


def topoc2ant(el, az):
    """
    Convert topocentric coordinates to antenna coordinates.

    Parameters
    ----------
    el: number or NumPy array
        Elevation angle, rad
    az: number or NumPy array
        Azimuth angle, rad

    Return
    ------
    gamma, epsilon [rad]
    """
    if isinstance(el, (float, int)) and isinstance(az, (float, int)):
        scalar = True
    else:
        scalar = False

    el = np.array(el, ndmin=1)
    az = np.array(az, ndmin=1)

    ua = np.deg2rad(7)
    ub = np.deg2rad(10)

    # !Avoid in-place operations on input arguments
    az_loc = ua + az
    ep = np.arcsin(-np.cos(az_loc) * np.cos(el))
    gam = ub - np.arctan2(-np.sin(az_loc) * np.cos(el), np.sin(el))

    if scalar:
        return gam.item(), ep.item()
    else:
        return gam, ep


def ant2topoc(gam, ep):
    """
    Convert antenna coordinates to topocentric coordinates.

    Parameters
    ----------
    gam: number or NumPy array
        Gamma antenna coordinate, rad
    ep: number or NumPy array
        Epsilon antenna coordinate, rad

    Return
    ------
    elevation, azimuth [rad]
    """
    if isinstance(gam, (float, int)) and isinstance(ep, (float, int)):
        scalar = True
    else:
        scalar = False

    gam = np.array(gam, ndmin=1)
    ep = np.array(ep, ndmin=1)

    ua = np.deg2rad(7)
    ub = np.deg2rad(10)

    # !Avoid in-place operations on input arguments
    gam_loc = ub - gam
    el = np.arcsin(np.cos(gam_loc) * np.cos(ep))
    az = 1.5*np.pi - ua - np.arctan2(np.sin(ep), np.cos(ep)*np.sin(gam_loc))

    if scalar:
        return el.item(), az.item()
    else:
        return el, az


def ant_max_direction(freq, fit='Lebedev_model'):
    """
    Return epsilon coordinate (in antenna system) for the maximum
    of antenna pattern.

    Parameters
    ----------
    freq: array-like or number
        Frequency, MHz.
    fit: str
        Fit origin: 'Lebedev', 'Vasiliev_old', 'Vasiliev'

    Return
    ------
    ep: array-like or number
        Epsilon of antenna coordinates, rad.
    """
    freq_range = (152, 164)
    freq = np.asarray(freq)
    freq[freq < freq_range[0]] = np.nan
    freq[freq > freq_range[1]] = np.nan

    if fit == 'Lebedev':
        freq_kilohertz = freq * 1e3  # to kHz, to float
        dep = 0.2816
        ep = 185634.9983 \
             - 4.877867876 * freq_kilohertz \
             + 4.803667396E-005 * freq_kilohertz**2 \
             - 2.102600271E-010 * freq_kilohertz**3 \
             + 3.453540782E-016 * freq_kilohertz**4
        ep += dep
        res = np.deg2rad(ep)
    elif fit == 'Lebedev_model':
        res = ant_max_direction_model_c_wrapped(freq * 1000.0)
    elif fit in ['Vasiliev_old', 'Vasiliev']:
        freq_hz = freq * 1e6  # to kHz, to float
        d = 0.87
        a = 0.136
        b = 0.014
        h = 0.384
        wavelength = speed_of_light / freq_hz
        k = 2.0 * np.pi / wavelength

        if fit == 'Vasiliev_old':
            hi_f = -32.3802499 \
                   + 6.329699841E-007 * freq_hz \
                   - 4.001034124E-015 * freq_hz ** 2 \
                   + 8.374351252E-024 * freq_hz ** 3
        else:
            hi_f = 13.5066804030779 \
                   - 23.894031675887E-008 * freq_hz \
                   + 15.2068384591317E-016 * freq_hz ** 2 \
                   - 3.28019314316317E-024 * freq_hz ** 3

        g = hi_f * np.sqrt(1 + ((a / (a + b)) * np.tan(k * h)) ** 2)
        theta = g - wavelength / d
        res = theta
    else:
        raise ValueError('Incorrect fit: {}'.format(fit))

    if res.size == 1:
        return res.item()
    else:
        return res


def freq_max_direction(ep: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return frequency at which given epsilon is a coordinate of pattern maximum.

    Args:
        ep: array-like or number
            Epsilon coordinate of antenna maximum, rad.

    Returns:
        freq: array-like or number
            Frequency, MHz.

    """
    eps_range = [-0.1, 0.728]
    ep = np.asarray(ep)
    ep[ep < eps_range[0]] = np.nan
    ep[ep > eps_range[1]] = np.nan

    coefs = [2.6391387779, -4.4269604510, -4.3101746665, 18.2875809605, 153.9381333482]
    res = np.polyval(coefs, ep)
    if res.size == 1:
        return res.item()
    else:
        return res


def obs2tar_position(ep, gam, distance, lat=IISR_LAT, lon=IISR_LON,
                     height=IISR_HEIGHT, return_above_sea=False):
    """
    Transform antenna coordinates of target to horizontal coordinates
    at the position of observer.

    Parameters
    ----------
    ep: number or NxM numpy array
        Epsilon coordinate of target in radar coordinate system, rad.
    gam: number or NxM numpy array
        Gamma coordinate of target in radar coordinate system, rad.
    distance: number or NxM numpy array
        Distance from the radar to target, meters.
    lat: number, default IISR latitude
        Latitude of the observer (radar), rad.
    lon: number, default IISR longitude
        Longitude of the observer (radar), rad.
    height: number, default IISR height
        Height of the observer (radar), meters.
    return_above_sea: bool, default False
        Return height above the sea. Originally return above the ground.
        See Notes.

    Returns
    -------
    lat: number or NxM numpy array
        Latitude of the target in horizontal coordinate system, rad.
    lon: number or NxM numpy array
        Longitude of the target in horizontal coordinate system, rad.
    height: number or NxM numpy array
        Height of the target  in horizontal coordinate system, meters.

    Notes
    -----
    With False flag return_above_sea, returned heights are heights
    above the sea minus observer height.
    """
    f_f = 1 / 298.257  # Earth compression
    e_e = 2.0 * f_f - f_f * f_f  # Earth eccentricity squared
    r0 = 6378140.0  # Biggest Earth semi-axis, m
    u_a = np.deg2rad(7)  # Angle between meridian and main axis of antenna
    u_b = np.deg2rad(10)  # Angle between normal to horn and zenith

    def find_h_fi(z, dist):
        phi = np.arctan(z / dist)
        for i in range(1, 16):
            c = 1 / np.sqrt(1 - e_e * np.sin(phi) ** 2)
            x = z + r0 * c * e_e * np.sin(phi)
            phi = np.arctan(x / dist)

        c = 1 / np.sqrt(1 - e_e * np.sin(phi) ** 2)
        x = dist / np.cos(phi) - r0 * c
        return phi, x

    zn = r0 / np.sqrt(1 - e_e * np.sin(lat) * np.sin(lat))
    rx = (zn + height) * np.cos(lat) * np.cos(lon)
    ry = (zn + height) * np.cos(lat) * np.sin(lon)
    rz = (zn * (1 - e_e) + height) * np.sin(lat)

    el = np.arcsin(np.cos(u_b - gam) * np.cos(ep))
    az = 1.5*np.pi - u_a - np.arctan2(np.sin(ep), np.cos(ep)*np.sin(u_b - gam))
    delt = np.arcsin(np.sin(el) * np.sin(lat)
                     + np.cos(az) * np.cos(el) * np.cos(lat))
    th = -np.arctan2(
        -np.sin(az) * np.cos(el),
        np.sin(el) * np.cos(lat) - np.cos(az) * np.cos(el) * np.sin(lat)
    )

    rx += distance * (np.cos(delt) * np.cos(lon + th))
    ry += distance * (np.cos(delt) * np.sin(lon + th))
    rz += distance * np.sin(delt)
    rr = np.sqrt(rx * rx + ry * ry)

    res_lon = np.arctan2(ry, rx)
    res_lat, h = find_h_fi(rz, rr)

    if return_above_sea:
        return res_lat, res_lon, h
    else:
        # TODO: should subtract altitude above sea and surface at
        # the position of target (phi, lg), not radar (lon, lat)
        return res_lat, res_lon, h - height


# === Shortcuts ===
def dist2height(distance, frequency):
    """
    Convert distance from IISR to height above the surface for certain
    frequency.

    Parameters
    ----------
    distance: number or NxM array
        Distance from the radar, meters.
    frequency: number of NxM array
        Frequency, MHz.

    Returns
    -------
    height: number of NxM array
    """
    frequency = np.asarray(frequency)
    distance = np.asarray(distance)
    assert ((frequency >= 147) & (frequency <= 167)).all()

    ep = ant_max_direction(freq=frequency)
    gam = 0
    _, _, height = obs2tar_position(ep, gam, distance, lat=IISR_LON,
                                    lon=IISR_LAT, height=IISR_HEIGHT)
    return height


@lru_cache()
def directivity(freq, angle_step_deg=0.1, method='simple',
                eps_width_deg=20, gam_width_deg=50, **kwargs):
    """
    Estimate directivity of IISR at given frequency.
    
    Parameters
    ----------
    freq: float
        Frequency, MHz.
    angle_step_deg: float
        Step in integration grid, degrees.
    method: 'simple', 'simps', default 'simps'
        Algorithm of integration. 'simple' or 'simps' should be preferred.
    eps_width_deg: float
        Width of epsilon integration grid (centered at maximum value of pattern), degrees.
    gam_width_deg: float
        Width of gamma integration grid (centered at maximum value of pattern), degrees.
    **kwargs:
        Arguments passed to calc_pattern.

    Notes
    -----
    Default values were chosen such that computation is as fast as possible with error within ~1%.

    Returns
    -------
    directivity: float
        Estimated directivity.
    """
    # Half widths
    eps_hw = np.deg2rad(eps_width_deg) / 2
    gam_hw = np.deg2rad(gam_width_deg) / 2
    step = np.deg2rad(angle_step_deg)

    ep = np.arange(-eps_hw, eps_hw + step, step)
    gam = np.arange(-gam_hw, gam_hw + step, step)
    gam, ep = np.meshgrid(gam, ep)

    pattern = calc_pattern(freq, ep + ant_max_direction(freq), gam, **kwargs)

    if method == 'simple':
        delta = step ** 2
        integral = np.dot(np.cos(ep[:, 0]), pattern.sum(axis=1)) * delta

    elif method == 'simps':
        integral_by_theta = simps(pattern * np.cos(ep), dx=step, axis=0)
        integral = simps(integral_by_theta, dx=step)

    else:
        raise ValueError('Wrong method ({})'.format(method))

    return 4 * np.pi * pattern.max() / integral


def monostatic_integral(freq, tr_type, rc_type, coord_sys='spherical'):
    """
    Calculate integral of (reception directivity * transmission directivity)
    over sphere. It used for volume scatter equations for atmospherical radars.

    Parameters
    ----------
    freq: number
        Frequency of IISR, MHz.
    tr_type: 'upper', 'lower' or 'both'
        Type of transmitting IISR horn.
    rc_type: 'upper', 'lower' or 'both'
        Type of receiving IISR horn.
    coord_sys: 'spherical', 'antenna' or 'phased_array'; default 'spherical'
        Coordinate system where to perform integration. If 'phased_array',
        system will be chosen such that antenna is handled as two crossed 
        linear phased arrays.
    Returns
    -------
    integral: float
        Integral of interest.
    """
    directivity_rc = directivity(freq, dnr_type=rc_type, coord_sys=coord_sys)
    directivity_tr = directivity(freq, dnr_type=tr_type, coord_sys=coord_sys)
    # Next one is not physical directivity; used just for convenience
    # Also note, that it is calculated over main beam and should not be normed
    mono_directivity = directivity(freq, integration_limits='beam',
                                   two_way=True, tr_type=tr_type,
                                   rc_type=rc_type, precision=500,
                                   coord_sys=coord_sys)
    return 4 * np.pi * directivity_rc * directivity_tr / mono_directivity


def approximate_directivity(frequencies, dnr_type):
    """Polynomical approximation of directivity.

    Args:
        frequencies: float or np.ndarray
            Frequencies, MHz.

    Returns:

    """
    coefs = {
        'lower': [-0.0600095838, 37.1204062200, -8616.7620104487,
                  889650.5824498440, -34468386.8650332987],
        'upper': [-0.0600095838, 37.1204062200, -8616.7620104487,
                  889650.5824498440, -34468386.8650332987],
        'both': [-0.1038420502, 64.2353988944, -14911.2823747195,
                 1539572.2749214517, -59650250.9854609445],
    }
    return np.polyval(coefs[dnr_type], frequencies)


def calc_directivity_approximation():
    n = 20
    frequency_range = np.linspace(152, 164, n)

    print('Directivity approximation')
    for dnr_type in ['upper', 'lower', 'both']:
        directivity_array = np.array(
            [directivity(f, dnr_type=dnr_type, angle_step_deg=0.05) for f
             in tqdm(frequency_range, total=n, desc='Directivity array calculation')]
        )
        coefs, residuals, _, _, _ = np.polyfit(frequency_range, directivity_array, deg=4, full=True)
        coefs_str = [f'{c:.10f}' for c in coefs]
        print(f'Polyfit coefficients for {dnr_type} horn = {", ".join(coefs_str)}'
              f' with residuals = {residuals.item():.20f}', file=sys.stderr)


def calc_freq_max_direction():
    n = 100
    freq_megahertz = np.linspace(152, 164, n)

    print('Approximate frequency at maximal direction')
    eps = ant_max_direction(freq_megahertz)
    coefs, residuals, _, _, _ = np.polyfit(eps, freq_megahertz, deg=4, full=True)
    coefs_str = [f'{c:.10f}' for c in coefs]
    print(f'Polyfit coefficients {", ".join(coefs_str)} '
          f'with residuals = {residuals.item():.20f}', file=sys.stderr)
