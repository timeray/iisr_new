"""
Wrapper for C-functions calculating antenna pattern for Irkutsk Incoherent
Scatter Radar (IISR).
"""


import ctypes as ct
from scipy.integrate import simps
from functools import lru_cache
import numpy as np
import sys
import os
from scipy.constants import speed_of_light

# Constants
IISR_LON = np.deg2rad(103.255)
IISR_LAT = np.deg2rad(52.875)
IISR_HEIGHT = 500.
TEMPERATURE_SENSETIVITY = 0.01  # degree per Celcius degree

_dir_path = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith('win'):
    dll = ct.CDLL(os.path.join(_dir_path, 'DNRIISR.dll'))
elif sys.platform.startswith('linux'):
    dll = ct.CDLL(os.path.join(_dir_path, 'DNRIISR.so'))
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
    if fit == 'Lebedev':
        freqkHz = freq * 1e3  # to kHz, to float
        dep = 0.2816
        ep = 185634.9983 \
             - 4.877867876 * freqkHz \
             + 4.803667396E-005 * freqkHz**2 \
             - 2.102600271E-010 * freqkHz**3 \
             + 3.453540782E-016 * freqkHz**4
        ep += dep
        return np.deg2rad(ep)
    elif fit == 'Lebedev_model':
        return ant_max_direction_model_c_wrapped(freq * 1000.0)
    elif fit in ['Vasiliev_old', 'Vasiliev']:
        freqHz = freq * 1e6  # to kHz, to float
        d = 0.87
        a = 0.136
        b = 0.014
        h = 0.384
        wavelength = speed_of_light / freqHz
        k = 2.0 * np.pi / wavelength

        if fit == 'Vasiliev_old':
            hi_f = -32.3802499 \
                   + 6.329699841E-007 * freqHz \
                   - 4.001034124E-015 * freqHz ** 2 \
                   + 8.374351252E-024 * freqHz ** 3
        else:
            hi_f = 13.5066804030779 \
                   - 23.894031675887E-008 * freqHz \
                   + 15.2068384591317E-016 * freqHz ** 2 \
                   - 3.28019314316317E-024 * freqHz ** 3

        g = hi_f * np.sqrt(1 + ((a / (a + b)) * np.tan(k * h)) ** 2)
        theta = g - wavelength / d
        return theta
    else:
        raise ValueError('Incorrect fit: {}'.format(fit))


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
    Ff = 1 / 298.257  # Earth compression
    Ee = 2.0 * Ff - Ff * Ff  # Earth eccentricity squared
    R0 = 6378140.0  # Biggest Earth semi-axis, m
    Ua = np.deg2rad(7)  # Angle between meridian and main axis of antenna
    Ub = np.deg2rad(10)  # Angle between normal to horn and zenith

    def find_h_fi(z, dist):
        phi = np.arctan(z / dist)
        for i in range(1, 16):
            C = 1 / np.sqrt(1 - Ee * np.sin(phi) ** 2)
            x = z + R0 * C * Ee * np.sin(phi)
            phi = np.arctan(x / dist)

        C = 1 / np.sqrt(1 - Ee * np.sin(phi) ** 2)
        x = dist / np.cos(phi) - R0 * C
        return phi, x

    ZN = R0 / np.sqrt(1 - Ee * np.sin(lat) * np.sin(lat))
    RX = (ZN + height) * np.cos(lat) * np.cos(lon)
    RY = (ZN + height) * np.cos(lat) * np.sin(lon)
    RZ = (ZN * (1 - Ee) + height) * np.sin(lat)

    el = np.arcsin(np.cos(Ub - gam) * np.cos(ep))
    az = 1.5*np.pi - Ua - np.arctan2(np.sin(ep), np.cos(ep)*np.sin(Ub - gam))
    delt = np.arcsin(np.sin(el) * np.sin(lat)
                     + np.cos(az) * np.cos(el) * np.cos(lat))
    th = -np.arctan2(
        -np.sin(az) * np.cos(el),
        np.sin(el) * np.cos(lat) - np.cos(az) * np.cos(el) * np.sin(lat)
    )

    RX += distance * (np.cos(delt) * np.cos(lon + th))
    RY += distance * (np.cos(delt) * np.sin(lon + th))
    RZ += distance * np.sin(delt)
    RR = np.sqrt(RX * RX + RY * RY)

    res_lon = np.arctan2(RY, RX)
    res_lat, H = find_h_fi(RZ, RR)

    if return_above_sea:
        return res_lat, res_lon, H
    else:
        # TODO: should subtract altitude above sea and surface at
        # the position of target (phi, lg), not radar (lon, lat)
        return res_lat, res_lon, H - height


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
