from typing import Tuple, Union

import numpy as np
from abc import ABCMeta

from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy import units as u
from astropy.time import Time

from scipy.integrate import simps
from tqdm import tqdm

from iisr.antenna.dnriisr import IISR_LAT, IISR_LON, IISR_HEIGHT, calc_pattern, topoc2ant, \
    freq_max_direction
from iisr.antenna.sun_utils import get_smoothed_elliptic_sun, SUN_INTEGRATION_REGION, \
    DEFAULT_SUN_FWHM
from iisr.math import axis_angle, rotate_vector, spherical2cartesian, cartesian2spherical, \
    gauss_fwhm2var, gaussian_on_sphere
from iisr.units import Frequency


class Kernel(metaclass=ABCMeta):
    def convolve_at(self, vector, freq, dnr_type, sun_north=None, temperature=None):
        """
        Convolve kernel with pattern at the position of given vector.

        Parameters
        ----------
        vector: 2-element tuple
            Destination vector in spherical coordinates.
        freq: number
            Frequency, MHz.
        dnr_type: 'upper', 'lower', both'
            Horn of IISR.
        sun_north: 2-element tuple or None, default None
            Approximate location of Sun north in topocentric coordinates.
        temperature: float or ndarray, optional
            Ambient atmospheric temperature in Celsius degrees.

        Returns
        -------
        pattern: np.ndarray
            Value of pattern.
        convolved_pattern: np.ndarray
            Estimate of convolution of pattern with kernel.
        """


class DeltaKernel(Kernel):
    def convolve_at(self, vector, freq, dnr_type, sun_north=None, temperature=None):
        """
        Convolve kernel with pattern at the position of given vector.

        Parameters
        ----------
        vector: 2-element tuple
            Destination vector in spherical coordinates.
        freq: number
            Frequency, MHz.
        dnr_type: 'upper', 'lower', both'
            Horn of IISR.
        sun_north: 2-element tuple or None, default None
            Approximate location of Sun north in topocentric coordinates.
            Useless for this kernel.
        temperature: float or ndarray, optional
            Ambient atmospheric temperature in Celcius degrees.

        Returns
        -------
        pattern: np.ndarray
            Value of pattern.
        convolved_pattern: np.ndarray
            Estimate of convolution of pattern with kernel.
        """
        theta, phi = vector
        gam, ep = topoc2ant(np.pi/2 - theta, phi)
        pattern = calc_pattern(freq, ep, gam, dnr_type=dnr_type, temperature=temperature)
        return pattern, pattern


class ConeKernel(Kernel, metaclass=ABCMeta):
    center = (0., 0., 1.)
    _integral = None

    """Mesh in spherical and cartesian coordinates."""
    mesh = NotImplemented
    """Step of the mesh in spherical coordinates."""
    mesh_step = NotImplemented
    """Values of the kernel at mesh points."""
    values = NotImplemented
    """Boolean to indicate asymmetry of the Sun"""
    asymmetric_sun = NotImplemented

    @property
    def integral(self):
        """Integral over gaussian"""
        if self._integral is None:
            dtheta, dphi = self.mesh_step
            cap_theta = self.mesh['spherical'][0]
            integral_by_theta = simps(self.values * np.sin(cap_theta),
                                      dx=dtheta, axis=0)
            self._integral = simps(integral_by_theta, dx=dphi)

        return self._integral

    def convolve_at(self, vector: Tuple, freq: int, dnr_type: str,
                    sun_north: Tuple = None,
                    temperature: Union[float, np.ndarray] = None):
        """
        Convolve kernel with pattern at the position of given vector.

        Parameters
        ----------
        vector: 2-element tuple
            Destination vector in topocentric coordinates.
        freq: number
            Frequency, MHz.
        dnr_type: 'upper', 'lower', both'
            Horn of IISR.
        sun_north: 2-element tuple or None, default None
            Approximate location of Sun north in topocentric coordinates.
        temperature: float or ndarray, optional
            Ambient atmospheric temperature in Celsius degrees.

        Returns
        -------
        pattern: np.ndarray
            Value of pattern at the vector.
        convolved_pattern: np.ndarray
            Estimate of convolution of pattern with kernel.
        """
        gam, eps = topoc2ant(np.pi / 2 - vector[0], vector[1])
        point_pattern = calc_pattern(freq, eps, gam, dnr_type=dnr_type, temperature=temperature)

        # Get axis of rotation
        vector = spherical2cartesian(*vector)
        axis_vector, angle = axis_angle(self.center, vector)

        # Get position of sun north and rotate underlying system mesh
        if sun_north is not None and self.asymmetric_sun:
            sun_north = spherical2cartesian(*sun_north)
            sun_north_vector = rotate_vector(sun_north, axis_vector, -angle)
            sun_north_angle = cartesian2spherical(*sun_north_vector)[1]
            vectors = self.mesh['spherical'].copy()
            vectors[1] += sun_north_angle
            vectors = spherical2cartesian(*vectors)
            vectors = [vec.ravel() for vec in vectors]
        else:
            vectors = self.mesh['cartesian'].reshape(3, -1)

        # Transform coordinates and perform rotation
        shape = self.values.shape
        new_cartesian_mesh = rotate_vector(vectors, axis_vector, angle)
        new_cartesian_mesh = new_cartesian_mesh.reshape(3, *shape)
        theta_mesh, phi_mesh = cartesian2spherical(*new_cartesian_mesh)
        gam_mesh, ep_mesh = topoc2ant(np.pi / 2 - theta_mesh, phi_mesh)

        # Calculate pattern at new coordinates
        pattern = calc_pattern(freq, ep_mesh, gam_mesh, dnr_type=dnr_type, temperature=temperature)

        # Integrate to get an estimate of the convolution
        dtheta, dphi = self.mesh_step
        cap_theta = self.mesh['spherical'][0]
        integral_by_theta = simps(pattern * self.values * np.sin(cap_theta),
                                  dx=dtheta, axis=1)
        integral = simps(integral_by_theta, dx=dphi)
        return point_pattern, integral


class GaussianKernel(ConeKernel):
    def __init__(self, size=SUN_INTEGRATION_REGION, fwhm=DEFAULT_SUN_FWHM, n_points=22500):
        """
        Create mesh grid centered at (theta, phi) = (0, 0) and evaluates
        gaussian.

        Parameters
        ----------
        size: number
            Angular size of mesh grid.
        fwhm: number
            Half-maximum full width of the gaussian.
        n_points: int
            Number of points in the mesh grid.
        """
        self.asymmetric_sun = False
        # Initialize mesh grid in spherical coordinates
        n_theta = np.sqrt(n_points)
        n_phi = n_theta
        phi, dphi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False,
                                retstep=True)
        theta, dtheta = np.linspace(0, size, n_theta, endpoint=False,
                                    retstep=True)
        phi, theta = np.meshgrid(phi, theta)

        self.mesh_step = (dtheta, dphi)
        self.mesh = {
            'spherical': np.stack([theta, phi]),
            'cartesian': np.stack(spherical2cartesian(theta, phi))
        }

        # Calculate gaussian at mesh grid
        var = gauss_fwhm2var(fwhm)
        center_spherical = cartesian2spherical(*self.center)
        self.values = gaussian_on_sphere(theta, phi, center_spherical[0],
                                         center_spherical[1], var=var)


class QuietSunKernel(ConeKernel):
    def __init__(self):
        """
        Create mesh grid centered at (theta, phi) = (0, 0) and evaluates
        normalized quiet sun brightness. N-S and E-W diameters of the Sun
        were taken from [Y. Leblanc, A. M. Le Squeren, 1968].

        As convolution on the sphere with acceptable accuracy demands
        considerable amount of computations, mesh grid and brightness are
        calculated separately and cached.
        """
        self.asymmetric_sun = True
        # Initialize mesh grid in spherical coordinates
        theta, phi, self.values = get_smoothed_elliptic_sun()
        self.mesh_step = np.diff(np.unique(theta))[0], np.diff(np.unique(phi))[0]

        self.mesh = {
            'spherical': np.stack([theta, phi]),
            'cartesian': np.stack(spherical2cartesian(theta, phi))
        }


def sun_pattern_1d(time_marks: np.ndarray, freqs: Frequency, dnr_type: str, kernel: Kernel):
    sun_coord = get_sun(Time(time_marks))
    # Sunpy.coordinates.sun.sky_position gives different coordinates
    # pos = [sky_position(tm) for tm in tqdm(time_marks, total=len(time_marks))]
    # ra = [x[0] for x in pos]
    # dec = [x[1] for x in pos]
    # sun_coord = GCRS(ra=ra, dec=dec)

    radar_loc = EarthLocation(lat=IISR_LAT * u.rad, lon=IISR_LON * u.rad, height=IISR_HEIGHT)
    observer_altaz = AltAz(location=radar_loc, obstime=time_marks)

    sun_coord_altaz = sun_coord.transform_to(observer_altaz)
    theta = np.pi / 2 - sun_coord_altaz.alt.radian
    phi = sun_coord_altaz.az.radian

    patterns = []
    convolved_patterns = []
    for tm, freq, vector in tqdm(zip(time_marks, freqs['MHz'], zip(theta, phi)),
                                 total=len(time_marks),
                                 desc='Estimate convolution of pattern and brightness'):
        pat, conv_pat = kernel.convolve_at(vector, freq, dnr_type=dnr_type)
        patterns.append(pat)
        convolved_patterns.append(conv_pat)

    return np.array(patterns), np.array(convolved_patterns)


def find_sun_max_frequencies(time_marks: np.ndarray) -> Frequency:
    """Return frequency at which the Sun's position epsilon is equal to epsilon at
    maximal pattern direction"""
    sun_coord = get_sun(Time(time_marks))

    radar_loc = EarthLocation(lat=IISR_LAT * u.rad, lon=IISR_LON * u.rad, height=IISR_HEIGHT)
    observer_altaz = AltAz(location=radar_loc, obstime=time_marks)

    sun_coord_altaz = sun_coord.transform_to(observer_altaz)

    _, eps = topoc2ant(sun_coord_altaz.alt.radian, sun_coord_altaz.az.radian)
    return Frequency(freq_max_direction(eps), 'MHz')
