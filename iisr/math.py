import numpy as np
from scipy.interpolate import griddata


def haversine(lat1, long1, lat2, long2):
    """
    Compute great-circle distance using Haversine formula.
    https://en.wikipedia.org/wiki/Haversine_formula
    https://en.wikipedia.org/wiki/Great-circle_distance

    Input coordinate arrays must be broadcastable.

    Parameters
    ----------
    lat1: number or array_like
        Latitude of the first points on the sphere.
    long1: number or array_like
        Longitude of the first points on the sphere.
    lat2: number or array_like
        Latitude of the second points.
    long2: number or array_like
        Longitude of the second points.

    Returns
    -------
    distance: float or np.ndarray
    """
    lat1 = np.atleast_1d(lat1).astype(float)
    long1 = np.atleast_1d(long1).astype(float)
    lat2 = np.atleast_1d(lat2).astype(float)
    long2 = np.atleast_1d(long2).astype(float)

    term1 = long1 - long2
    del long1, long2
    np.abs(term1, out=term1)
    np.divide(term1, 2, out=term1)
    np.sin(term1, out=term1)
    np.square(term1, out=term1)
    np.multiply(term1, np.cos(lat1), out=term1)
    np.multiply(term1, np.cos(lat2), out=term1)

    term2 = lat1 - lat2
    del lat1, lat2
    np.abs(term2, out=term2)
    np.divide(term2, 2, out=term2)
    np.sin(term2, out=term2)
    np.square(term2, out=term2)

    distance = term1
    np.add(term1, term2, out=distance)
    del term2
    np.sqrt(distance, out=distance)
    np.arcsin(distance, out=distance)
    np.multiply(distance, 2, out=distance)

    if distance.size == 1:
        return distance.item()
    else:
        return distance


def spherical2cartesian(theta, phi):
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)


def cartesian2spherical(x, y, z):
    return np.arctan2(np.sqrt(x**2 + y**2), z), np.arctan2(y, x)


def normalize(vector, axis=-1):
    """
    Normalize a vector.

    Parameters
    ----------
    vector: array_like
        Vector.
    axis: int
        Axis along which normalization should be applied.
        Defaults to last axis.

    Returns
    -------
    normalized vector: np.ndarray
    """
    vector = np.asarray(vector)
    return vector / np.linalg.norm(vector, axis=axis, keepdims=True)


def axis_angle(vector1, vector2, axis=0):
    """
    Calculate axis angle representation of rotation from vec1 to vector2.
    Return nan vector if vectors are collinear.

    Parameters
    ----------
    vector1: array_like
        First vector in cartesian coordinates.
    vector2: array_like
        Second vector in cartesian coordinates.
    axis: int
        Axis that represent vector dimensions.

    Returns
    -------
    axis_vec: np.ndarray
        Unit vector representing direction of rotation axis.
    angle: np.ndarray
        Angle between the vectors. Have dim(axis_vec) - 1 dimension.
    """
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)
    norm1 = np.linalg.norm(vector1, axis=axis)
    norm2 = np.linalg.norm(vector2, axis=axis)
    angle = np.arccos((vector1 * vector2).sum(axis=axis) / (norm1 * norm2))
    return normalize(np.cross(vector1, vector2, axis=axis), axis=axis), angle


class Quaternion:
    def __init__(self, a, b, c, d):
        """
        Normalized quaternion for rotation of vectors in space.

        q = a + b*i + c*j + d*k,
        where a, b, c, d are real numbers, i, j, k are quaternion units.
        ||q||_2 = 1

        a, b, c, d should be numbers or arrays of same size
        """
        # Create 4xN array. It should be contiguous from vector to vector
        # to increase speed of multiplication.
        self.value = np.vstack([a, b, c, d])

    @property
    def vector(self):
        """Return (3xN) array representing a vector in Cartesian coordinates"""
        return self.value[1:]

    def conj(self):
        a, b, c, d = self.value
        return Quaternion(a, -b, -c, -d)

    @classmethod
    def from_axis_angle(cls, axis_vec, angle):
        """
        Create normalized quaternion from axis-angle representation of rotation

        Parameters
        ----------
        axis_vec: (3, N) array_like or (3, ) array_like
            Rotation axis vector.
        angle: (N, ) array_like or number
            Angle of rotation.

        Returns
        -------
        quaternion: Quaternion
        """
        axis_vec = np.asarray(axis_vec)
        if axis_vec.ndim == 1:
            axis_vec = axis_vec[:, None]

        if axis_vec.shape[0] != 3:
            raise ValueError('Input vector must have shape (3, N) but '
                             '{} was given'.format(axis_vec.shape))

        angle = np.atleast_1d(angle)
        if angle.size > 1 or axis_vec.shape[0] > 1:
            axis_vec, angle = np.broadcast_arrays(axis_vec, angle)
        axis_vec = normalize(axis_vec, axis=0)
        angle = angle / 2
        angle_sin = np.sin(angle)
        vector_part = angle_sin * axis_vec
        quaternion = (
            np.cos(angle)[0],
            vector_part[0],
            vector_part[1],
            vector_part[2],
        )
        return cls(*quaternion)

    @classmethod
    def from_vector(cls, vector):
        """
        Create quaternion, setting vector part.

        Parameters
        ----------
        vector: (3, N) array_like or (3, ) array_like

        Returns
        -------
        quaternion: Quaternion
        """
        vector = np.asarray(vector)
        if vector.ndim == 1:
            vector = vector[:, None]

        if vector.shape[0] != 3:
            raise ValueError('Input vector must have shape (3, N) but '
                             '{} was given'.format(vector.shape))
        n = vector.shape[1]
        return cls(np.zeros(n), vector[0], vector[1], vector[2])

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            a1, b1, c1, d1 = self.value
            a2, b2, c2, d2 = other.value
            a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
            b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
            c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
            d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
            return Quaternion(a, b, c, d)

        else:
            raise ValueError('Multiplication by scalar is not supported, '
                             'as quaternion should be normalized.')


def rotate_vector(vector, axis_vec, angle):
    """
    Rotate vector or array of vectors in Cartesian coordinates over
    axis vector.

    Axis-angle representation can be unique for all array of vectors or
    individual for each vector.

    All-nan axis vector is treated as special case when vectors are collinear
    so no rotation will be applied.

    Parameters
    ----------
    vector: (3, N) array_like
        Vector in Cartesian coordinates.
    axis_vec: (3, ) array_like or (3, N) array_like
        Axis vector over which rotation should be performed.
    angle: number or (N, ) array_like
        Angle of rotation.

    Returns
    -------
    rotated_vector: (3, N) np.ndarray
        Rotated vectors.
    """
    vector = np.asarray(vector)
    if vector.ndim == 1:
        vector = vector[:, None]

    # No rotation if axis vector contain all nans
    n = vector.shape[1]
    axis_vec = np.asarray(axis_vec)
    if axis_vec.ndim == 1:
        no_rotation_mask = (np.ones(n) * np.isnan(axis_vec).all()).astype(bool)
    else:
        no_rotation_mask = np.isnan(axis_vec).all(axis=0)

    vec_quaternion = Quaternion.from_vector(vector)
    axis_quaternion = Quaternion.from_axis_angle(axis_vec, angle)
    res_quaternion = axis_quaternion * vec_quaternion * axis_quaternion.conj()
    res_vector = res_quaternion.vector
    res_vector[:, no_rotation_mask] = vector[:, no_rotation_mask]
    return res_vector


def gauss_fwhm2var(angular_full_width):
    return angular_full_width**2 / (8 * np.log(2))


def gauss_var2fwhm(variance):
    return 2 * np.sqrt(2 * np.log(2) * variance)


def spherical2geographical(theta, phi):
    """Transform spherical coordinates to geographical coordinates"""
    return np.pi / 2 - theta, phi


def geographical2spherical(lat, long):
    """Transform geographical coordinates to spherical coordinates"""
    return np.pi / 2 - lat, long


def gaussian_on_sphere(theta, phi, theta0, phi0, var, normalize=False):
    """
    Compute symmetric gaussian at given spherical coordinates.

    Parameters
    ----------
    theta: number or array_like
        Theta coordinates, where gaussian should be evaluated.
    phi: number or array_like
        Phi coordinates, where gaussian should be evaluated.
    theta0: number
        Mean theta coordinate.
    phi0: number
        Mean phi coordinate.
    var: number
        Variance. It corresponds to angular extend of the gaussian.
        Use gauss_fwhw2var and gauss_var2fwhw helper functions to transit
        between angular extent and variance of the gaussian.
    normalize: bool, default False
        Normalize output gaussian, such that integral over the sphere will be
        equal to 1.

    Returns
    -------
    gaussian: number or array_like
        Gaussian at the given coordinates.
    """
    lat0, long0 = spherical2geographical(theta0, phi0)
    lat, long = spherical2geographical(theta, phi)
    distances = haversine(lat0, long0, lat, long)

    gaussian = np.exp(-distances ** 2 / (2 * var))

    if normalize:
        std = np.sqrt(var)
        norm = 1 / (std * np.sqrt(2 * np.pi))
        return norm * gaussian
    else:
        return gaussian


def masked_correlate(a, v, mode='valid', propagate_mask=False):
    """
    Correlation for masked arrays.
    It is implementation of np.ma.correlate from numpy 1.12.
    
    Parameters
    ----------
    a, v: array-like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `np.convolve` docstring.
    propagate_mask : bool
        If True, then a result element is masked if any masked element
        contributes towards it.
        If False, then a result element is only masked if no non-masked element
        contribute towards it.
    
    Returns
    -------
    out : MaskedArray
        Discrete cross-correlation of `a` and `v`.
    """

    if propagate_mask:
        # results which are contributed to
        # by either item in any pair being invalid
        mask = (
            np.correlate(np.ma.getmaskarray(a),
                         np.ones(np.shape(v), dtype=np.bool),
                         mode=mode)
          | np.correlate(np.ones(np.shape(a), dtype=np.bool),
                         np.ma.getmaskarray(v), mode=mode)
        )
        data = np.correlate(np.ma.getdata(a), np.ma.getdata(v), mode=mode)
    else:
        # results which are not contributed to by any pair of valid elements
        mask = ~np.correlate(~np.ma.getmaskarray(a), ~np.ma.getmaskarray(v),
                             mode=mode)
        data = np.correlate(np.ma.filled(a, 0), np.ma.filled(v, 0), mode=mode)

    return np.ma.masked_array(data, mask=mask)


def _moving_average(a, n=1):
    # Moving average from
    # http://stackoverflow.com/questions/14313510
    # /how-to-calculate-moving-average-using-numpy
    # Provides array with N-n+1 size
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def mov_average(data, n_avg, cycled=False):
    """
    Compute moving average. Output array will have same length as input.
    If input array has nan values then masked array would be returned. 
    
    Parameters
    ----------
    data: Nx1 array-like
        Data to average.
    n_avg: int
        Number of data point for averaging window.
    cycled: bool, default=False
        If False, missing boundary point would be considered as zeros.
        If True, boundary point would be taken from opposite corner of
        the input array.
    
    Returns
    -------
    out: Nx1 ndarray
        Moving average.
    """

    if np.isnan(data).any():
        data = np.ma.array(data, mask=np.isnan(data))

    if np.ma.is_masked(data):
        correlate = masked_correlate
        concatenate = np.ma.concatenate
        masked = True
        mask = data.mask
    else:
        data = np.asarray(data)
        correlate = np.correlate
        concatenate = np.concatenate
        masked = False
        mask = False

    if data.size < n_avg:
        raise ValueError('Averaging window is bigger than data length')
    elif n_avg <= 0:
        raise ValueError('Number of point to averaging should be positive')
    win = np.ones(n_avg)
    if cycled:
        if n_avg % 2 == 0:
            data = concatenate(
                (data[- (n_avg // 2) + data.size:], data,
                 data[: n_avg//2 - 1])
            )
        else:
            data = concatenate(
                (data[data.size - (n_avg // 2):], data, data[: n_avg // 2])
            )
        if masked:
            ones = np.ma.ones(data.size)
            ones.mask = data.mask
            window_len = correlate(ones, win, 'valid')
            mov_avg = correlate(data, win, 'valid') / window_len
        else:
            win /= n_avg
            mov_avg = correlate(data, win, 'valid')

    else:
        # we need to divide result by window length, but with 'same'
        # mode first and last points of data would be affected
        # by smaller window;
        # easiest (but slow) way to adjust coefficient of division,
        # i.e. window size is to correlate window with arrays of 1s.
        if masked:
            ones = np.ma.ones(data.size)
            ones.mask = data.mask
        else:
            ones = np.ones(data.size)

        window_len = correlate(ones, win, 'same')
        mov_avg = correlate(data, win, 'same') / window_len

    if masked:
        mov_avg.mask = mask

    return mov_avg


def window_reduce(arr, window, func, axis=None, keepdims=True):
    """
    Calculate func of array values in consecutive windows
    along specified axis.

    Values outside window split (at the end of the axis) will be dropped.

    arr: np.ndarray
    window: int
        Window length.
    func: function
        Function to apply. Must accept axis argument.
    axis: None or int
        Axis along which the mean is computed. None stands for last axis.
    keepdims: bool, default True
        If True, dimension of output array will be same as input array.
        If False, axis with size 1 will be squeezed.
    """
    if axis is None:
        axis = len(arr.shape) - 1
    elif axis < 0:
        raise ValueError('Axis should be non-negative integer or None')

    if window <= 0:
        raise ValueError('Window should be positive integer')

    axis_size = arr.shape[axis]

    if axis_size < window:
        raise ValueError('Array size along given axis is smaller than window')

    n_windows = axis_size // window

    new_shape = list(arr.shape)
    new_shape[axis] = n_windows
    new_shape.insert(axis+1, window)

    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(0, n_windows*window)

    result = func(arr[sl].reshape(new_shape), axis=axis+1)

    if keepdims:
        return result
    else:
        return result.squeeze()


def coordinate_griddata(coord1, coord2, array, new_coord1, new_coord2,
                        method='cubic', verbose=True):
    """Interpolate 2d array provided as coordinates array using griddata"""
    coord1, coord2 = np.meshgrid(coord1, coord2, indexing='ij')
    points = np.stack((coord1.ravel(), coord2.ravel()), axis=-1)
    del coord1, coord2

    return_shape = new_coord1.size, new_coord2.size
    new_coord1, new_coord2 = np.meshgrid(new_coord1, new_coord2, indexing='ij')
    new_points = np.stack((new_coord1.ravel(), new_coord2.ravel()), axis=-1)
    del new_coord1, new_coord2

    array = array.ravel()
    if np.ma.is_masked(array):
        if verbose:
            n_dropped = array.mask.sum()
            print('drop {} ({:.2f}%) masked points during 2D interpolation'
                  ''.format(n_dropped, 100 * n_dropped / array.size))
        points = points[~array.mask]
        array = array.compressed()

    array = griddata(points, array, new_points, method=method, rescale=True)
    array = np.ma.array(array, mask=np.isnan(array))
    return array.reshape(return_shape)


if __name__ == '__main__':
    from numpy.ma.testutils import assert_equal, assert_mask_equal
    from numpy.ma.testutils import assert_allclose
    from numpy.testing import assert_raises

    def test_masked_correlation():
        # Test for correlation from numpy 1.12
        a = np.ma.masked_equal(np.arange(5), 2)
        b = np.array([1, 1])

        test = masked_correlate(a, b, mode='full')
        assert_equal(test, np.ma.masked_equal([0, 1, -1, -1, 7, 4], -1))

        test = masked_correlate(a, b, propagate_mask=False, mode='full')
        assert_equal(test, np.ma.masked_equal([0, 1, 1, 3, 7, 4], -1))

        test = masked_correlate([1, 1], [1, 1, 1], mode='full')
        assert_equal(test, np.ma.masked_equal([1, 2, 2, 1], -1))

        a = [1, 1]
        b = np.ma.masked_equal([1, -1, -1, 1], -1)
        test = masked_correlate(a, b, propagate_mask=False, mode='full')
        assert_equal(test, np.ma.masked_equal([1, 1, -1, 1, 1], -1))
        test = masked_correlate(a, b, propagate_mask=True, mode='full')
        assert_equal(test, np.ma.masked_equal([1, -1, -1, -1, 1], -1))

    def test_mov_avg():
        odd = np.array([1, 2, 3, 4, 5])
        even = np.array([0, 1, 2, 3, 4, 5])
        ma_odd = np.ma.masked_equal(odd, 2)
        ma_even = np.ma.masked_equal(even, 2)

        with assert_raises(ValueError):
            mov_average(odd, 6)

        with assert_raises(ValueError):
            mov_average(odd, -1)

        # Should include tests for even and odd arrays, even and odd windows,
        # cycled mode and masked arrays

        # basic
        assert_equal(mov_average(odd, 3), np.array([1.5, 2, 3, 4, 4.5]))
        assert_equal(mov_average(odd, 4), np.array([1.5, 2, 2.5, 3.5, 4]))
        assert_equal(mov_average(even, 3), np.array([0.5, 1, 2, 3, 4, 4.5]))
        assert_equal(mov_average(even, 4), np.array([0.5, 1, 1.5, 2.5, 3.5, 4]))

        # cycled
        assert_allclose(mov_average(odd, 3, cycled=True),
                        np.array([8/3, 2, 3, 4, 10/3]))
        assert_allclose(mov_average(odd, 4, cycled=True),
                        np.array([3, 11/4, 2.5, 3.5, 13/4]))

        assert_allclose(mov_average(even, 3, cycled=True),
                        np.array([2, 1, 2, 3, 4, 3]))
        assert_allclose(mov_average(even, 4, cycled=True),
                        np.array([2.5, 2, 1.5, 2.5, 3.5, 3]))

        # masked basic
        mov_avg = mov_average(ma_odd, 3)
        correct = np.ma.masked_equal([1, -1, 3.5, 4, 4.5], -1)
        assert_equal(mov_avg, correct)
        assert_mask_equal(mov_avg.mask, correct.mask)

        # masked cycled
        mov_avg = mov_average(ma_even, 4, cycled=True)
        correct = np.ma.masked_equal([2.5, 2, -1, 8/3, 4, 3], -1)
        assert_allclose(mov_avg, correct)
        assert_mask_equal(mov_avg.mask, correct.mask)

    def test_window_mean_reduce():
        test_arr = np.array([
            [1, 2, 3, 4, 5, 6],
            [3, 1, 2, 0, 1, -1]
        ])
        func = np.mean

        with assert_raises(ValueError):
            # Axis should be non-negative
            print(window_reduce(test_arr, window=2, func=func, axis=-1))

        assert_equal(window_reduce(test_arr, window=2, func=func, axis=0),
                     np.array([[2., 1.5, 2.5, 2., 3., 2.5]]))

        assert_equal(window_reduce(test_arr, window=2, func=func, axis=0,
                                   keepdims=False),
                     np.array([2., 1.5, 2.5, 2., 3., 2.5]))

        with assert_raises(ValueError):
            # Window should be smaller or equal to size of reduced axis
            print(window_reduce(test_arr, window=3, func=func, axis=0))

        assert_equal(window_reduce(test_arr, window=2, func=func, axis=1),
                     np.array([[1.5, 3.5, 5.5], [2., 1., 0.]]))

        assert_equal(window_reduce(test_arr, window=2, func=func, axis=1),
                     np.array([[1.5, 3.5, 5.5], [2., 1., 0.]]))

        assert_equal(window_reduce(test_arr, window=3, func=func, axis=1),
                     np.array([[2., 5.], [2., 0.]]))

        assert_equal(window_reduce(test_arr, window=4, func=func, axis=1),
                     np.array([[2.5], [1.5]]))

        assert_equal(window_reduce(test_arr, window=6, func=func, axis=1),
                     np.array([[3.5], [1.]]))

        assert_equal(window_reduce(test_arr, window=1, func=func, axis=0),
                     test_arr)
        assert_equal(window_reduce(test_arr, window=1, func=func, axis=1),
                     test_arr)

        with assert_raises(ValueError):
            # Window should be positive
            window_reduce(test_arr, window=0, func=func)

        func = np.sum

        test_arr_1d = np.array([1., 2., 4., 5.])
        assert_equal(window_reduce(test_arr_1d, window=2, func=func),
                     np.array([3., 9.]))

    def test_haversine():
        def simple_formula(lat1, long1, lat2, long2):
            return np.arccos(
                np.sin(lat1) * np.sin(lat2)
                + np.cos(lat1) * np.cos(lat2) * np.cos(np.abs(long1 - long2))
            )

        lat0, long0 = -0.35, 3.5
        lats = np.array([0.2, 1.5, -0.4])
        longs = np.array([1., 0., 2.5])
        test_dists = simple_formula(lat0, long0, lats, longs)

        # Scalars
        dist = haversine(lat0, long0, lats[0], longs[0])
        assert dist == test_dists[0]
        assert type(dist) == type(float(test_dists[0]))

        # Arrays
        dists = haversine(lat0, long0, lats, longs)
        np.testing.assert_almost_equal(dists, test_dists)
        dists = haversine(lats, longs, lats, longs)
        np.testing.assert_almost_equal(dists, np.zeros_like(dists, dtype=float))

        # Same point
        assert haversine(0, 0, 0, 0) == 0.

    def test_spherical_geographical_conversion():
        test_theta, test_phi = 3 * np.pi / 4, 2.6
        test_lat, test_long = -np.pi / 4, 2.6

        theta, phi = geographical2spherical(test_lat, test_long)
        assert test_theta == theta
        assert test_phi == phi

        # Reciprocity
        theta, phi = geographical2spherical(
            *spherical2geographical(test_theta, test_phi)
        )
        assert (test_theta, test_phi) == (theta, phi)
        lat, long = spherical2geographical(
            *geographical2spherical(test_lat, test_long)
        )
        assert (test_lat, test_long) == (lat, long)

    def test_gauss_fwhm_var_conversion():
        test_var = 3
        test_width = 4.078

        width = gauss_var2fwhm(3)
        assert np.isclose(test_width, width, atol=1e-3)

        # Reciprocity
        width = gauss_var2fwhm(gauss_fwhm2var(test_width))
        assert width == test_width

        var = gauss_fwhm2var(gauss_var2fwhm(test_var))
        assert test_var == var

    def test_gaussian():
        assert gaussian_on_sphere(0, 0, 0, 0, 1) == 1.
        theta, phi = np.pi / 4, 0
        var = gauss_fwhm2var(theta * 2)
        assert gaussian_on_sphere(theta, phi, 0, 0, var) == 0.5

    def test_axis_angle():
        vector1 = (0., 0., 3.)
        vector2 = (2., 0., 0.)
        test_axis = (0., 1., 0.)
        test_angle = np.pi / 2
        axis, angle = axis_angle(vector1, vector2)
        np.testing.assert_almost_equal(axis, test_axis)
        assert angle == test_angle

        vector1 = np.array([[1., 0., 0.], [0., 1., 0.]])
        vector2 = np.array([[1., 0., 0.], [1., 0., 0.]])
        test_axis = np.array([[np.nan, np.nan, np.nan], [0., 0., -1.]])
        test_angle = np.array([0., np.pi / 2])
        axis, angle = axis_angle(vector1, vector2, axis=-1)
        np.testing.assert_almost_equal(axis, test_axis)
        np.testing.assert_almost_equal(angle, test_angle)

        axis, angle = axis_angle(vector1.T, vector2.T, axis=0)
        np.testing.assert_almost_equal(axis, test_axis.T)
        np.testing.assert_almost_equal(angle, test_angle)


    def test_rotate_vector():
        # Simple case
        vector = np.array([1., 1., 0])
        axis_vector = np.array([0., 0., 1.])
        angle = np.pi
        test_vector = np.array([-1., -1., 0])[:, None]
        res_vector = rotate_vector(vector, axis_vector, angle)
        np.testing.assert_almost_equal(test_vector, res_vector)

        # Two vectors, two axis vectors, one angle
        vector = np.array([[1., 1., 1.], [1., 1., 1.]]).T
        axis_vector = np.array([[1., -1., 0], [-1., -1., 0]]).T
        angle = np.pi
        test_vector = np.array([[-1., -1., -1.], [1., 1., -1.]]).T

        res_vector = rotate_vector(vector, axis_vector, angle)
        np.testing.assert_almost_equal(test_vector, res_vector)

        # Two vectors, one axis vector, two angles
        axis_vector = np.array([1., 0., 0.])
        angle = np.array([np.pi, -np.pi/4])
        test_vector = np.array([[1., -1., -1.], [1., np.sqrt(2), 0]]).T

        res_vector = rotate_vector(vector, axis_vector, angle)
        np.testing.assert_almost_equal(test_vector, res_vector)

        # Wiki example
        vector = np.array([1., 2., 3.])
        axis_vector = np.array([1., 1., 1.])
        angle = 2 * np.pi / 3
        test_vector = np.array([3., 1., 2])[:, None]
        res_vector = rotate_vector(vector, axis_vector, angle)
        np.testing.assert_almost_equal(test_vector, res_vector)

    test_masked_correlation()
    test_mov_avg()
    test_window_mean_reduce()
    test_haversine()
    test_spherical_geographical_conversion()
    test_gauss_fwhm_var_conversion()
    test_gaussian()
    test_axis_angle()
    test_rotate_vector()
    print('Tests passed')
