"""Utilities to represent Sun for IISR"""
from iisr_old.tools import math
from scipy.integrate import simps
from typing import Tuple
from iisr import IISR_PATH
import numpy as np


CACHE_CONVOLUTION = True
CACHE_PATH = IISR_PATH / 'antenna' / 'sun_convolution' / 'smoothed_elliptic.npz'

# Sun diameters in north-south and east-west directions
# FWHM - full width half magnitude
# Taken from [Leblanc, Le Squeren, 1969], radiotelescope Nancay, 169 MHz
# where reported diameters were 32 min x 38 min by 95% power level
# To compensate for 95% level, I select additional coefficients, so that
# output brightness looks more like figures from the paper
NORTH_SOUTH_FWHM = np.deg2rad(0.533333)
EAST_WEST_FWHM = np.deg2rad(0.633333)
OPTICAL_DIAMETER = np.deg2rad(0.527)
SUN_INTEGRATION_REGION = np.deg2rad(1.)
DEFAULT_SUN_FWHM = np.deg2rad(0.5)


def _ellipse(theta_mesh, phi_mesh):
    first_term = np.sin(theta_mesh) ** 2 * np.cos(phi_mesh) ** 2 / np.sin(NORTH_SOUTH_FWHM / 2) ** 2
    second_term = np.sin(theta_mesh) ** 2 * np.sin(phi_mesh) ** 2 / np.sin(EAST_WEST_FWHM / 2) ** 2
    ellipse_mask = (first_term + second_term) <= 1

    return ellipse_mask.astype(float)


def calc_smoothed_elliptic_sun(mesh_n_points: int = 22500,
                               mesh_theta_max: float = SUN_INTEGRATION_REGION,
                               conv_n_points: int = 62500,
                               conv_region: float = SUN_INTEGRATION_REGION,
                               gaussian_fwhm: float = np.deg2rad(0.25),
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return spherical mesh grid and values of elliptic Sun brightness
    with center at theta = 0 and north aligned with phi = 0.
    Mesh grid is represented by equally distributed theta and phi.
    Brightness is smoothed by gaussian.

    Parameters
    ----------
    mesh_n_points: int
        Number of points of the created mesh grid.
    mesh_theta_max: float
        Maximum theta value, which limits size of the mesh grid.
    conv_n_points: int
        Number of points of the mesh grid to estimate convolution integral.
    conv_region: float
        Maximum theta value, which limits convolution region.
    gaussian_fwhm: float, default 0.1 deg
        Width of smoothing gaussian.

    Returns
    -------
    theta_mesh: 2-dim np.ndarray
        Theta mesh-grid coordinates of spherical system.
    phi_mesh: 2-dim np.ndarray
        Phi mesh-grid coordinates of spherical system.
    values: 2-dim np.ndarray
        Normalized values of the Sun brightness.
    """

    # Initialize mesh grid
    # Number of points in theta and phi are equal to increase resolution
    # in theta
    n_theta_mesh = int(np.sqrt(mesh_n_points))
    n_phi_mesh = n_theta_mesh
    phi_mesh = np.linspace(0., 2 * np.pi, n_phi_mesh, endpoint=False)
    theta_mesh = np.linspace(0., mesh_theta_max, n_theta_mesh, endpoint=True)
    phi_mesh, theta_mesh = np.meshgrid(phi_mesh, theta_mesh)

    # Initialize convolution grid
    # Number of points in theta and phi are equal to increase resolution
    # in theta
    n_theta_conv = int(np.sqrt(conv_n_points))
    n_phi_conv = n_theta_conv
    phi_conv, dphi_conv = np.linspace(0., 2 * np.pi, n_phi_conv,
                                      endpoint=False, retstep=True)
    theta_conv, dtheta_conv = np.linspace(0., conv_region, n_theta_conv,
                                          endpoint=True, retstep=True)
    phi_conv, theta_conv = np.meshgrid(phi_conv, theta_conv)
    conv_mesh = np.stack(math.spherical2cartesian(theta_conv, phi_conv))

    var = math.gauss_fwhm2var(gaussian_fwhm)
    sun_center_cartesian = (0., 0., 1.)
    sun_center_spherical = math.cartesian2spherical(*sun_center_cartesian)
    gaussian = math.gaussian_on_sphere(theta_conv,
                                       phi_conv,
                                       sun_center_spherical[0],
                                       sun_center_spherical[1],
                                       var=var)

    values = np.empty_like(theta_mesh, dtype=float)
    shape = theta_conv.shape
    vectors = conv_mesh.reshape(3, -1)
    for i in range(n_theta_mesh):
        print('Iter over theta {}/{}'.format(i + 1, n_theta_mesh))
        for j in range(n_phi_mesh):
            # Transform coordinates and perform rotation
            vector = math.spherical2cartesian(theta_mesh[i, j], phi_mesh[i, j])
            axis_vector, angle = math.axis_angle(sun_center_cartesian, vector)

            new_cartesian_mesh = math.rotate_vector(vectors, axis_vector,
                                                    angle)
            new_cartesian_mesh = new_cartesian_mesh.reshape(3, *shape)
            new_theta_conv, new_phi_conv = math.cartesian2spherical(
                *new_cartesian_mesh)
            ellipse = _ellipse(new_theta_conv, new_phi_conv)
            integral_by_theta = simps(
                ellipse * gaussian * np.sin(theta_conv),
                dx=dtheta_conv, axis=1
            )
            values[i, j] = simps(integral_by_theta, dx=dphi_conv)
    values /= values.max()
    return theta_mesh, phi_mesh, values


def cache_smoothed_elliptic_sun(*args, **kwargs):
    """Caches convolution of ellipsis and gaussian."""
    theta_mesh, phi_mesh, values = calc_smoothed_elliptic_sun(*args, **kwargs)
    if not CACHE_PATH.parent.exists():
        CACHE_PATH.parent.mkdir()
    np.savez(str(CACHE_PATH),
             theta_mesh=theta_mesh, phi_mesh=phi_mesh, values=values)


def get_smoothed_elliptic_sun(*args, **kwargs) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Try to load arrays
    if CACHE_CONVOLUTION and CACHE_PATH.exists():
        arrs = np.load(str(CACHE_PATH))
        return arrs['theta_mesh'], arrs['phi_mesh'], arrs['values']
    elif CACHE_CONVOLUTION and not CACHE_PATH.exists():
        raise FileNotFoundError(
            'Gaussian smoothing was not precalculated.\n'
            'Run cache_smoothed_elliptic_sun() or disable caching.')
    else:
        theta_mesh, phi_mesh, values = \
            calc_smoothed_elliptic_sun(*args, **kwargs)
        return theta_mesh, phi_mesh, values


get_smoothed_elliptic_sun.__doc__ = calc_smoothed_elliptic_sun.__doc__


if __name__ == '__main__':
    # Calculate cached convolution
    cache_smoothed_elliptic_sun()
