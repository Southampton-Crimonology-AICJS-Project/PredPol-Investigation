import logging
import numpy as np
from numba import types, njit
from numba.typed import Dict
from scipy.interpolate import interp2d
import ai_cjs


@njit(
    types.int64[:](
        types.int64, types.int64
    ),
    cache=True
)
def _iter(start=0, stop=10):
    """
    jit friendly range function to create a int64 iterator
    :param start: min value
    :param stop: max value + 1
    :return: [start,...,stop)
    """
    return np.array([i for i in range(start, stop)]).astype(np.int64)


# making the ability for jit dictionaries to be created=================================================================
# necessary for successfully initialising the array in the dictionary
float_array = types.float32[:]


@njit(cache=True)
def make_dict(_n=10):
    """
    Returns a numba typed dict to store PredPol parameters
    :return: types.DictType(keyty=types.unicode_type, valty='float32') keys 'u', 'theta' 'w'
    """
    pred_pol_inputs = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    pred_pol_inputs['u'] = np.ones(shape=(_n,)).astype(np.float32)
    pred_pol_inputs['theta'] = np.array([1.0]).astype(np.float32)
    pred_pol_inputs['w'] = np.array([1.0]).astype(np.float32)
    return pred_pol_inputs


# we must make the dictionary to initialise the function and tell jit what the typing is
_d = make_dict(_n=10)
# ======================================================================================================================


@njit(
    types.float32(
        types.DictType(keyty=types.unicode_type, valty=float_array),
        types.float32[:, :],
        types.int64,
        types.boolean[:],
        types.double
    ),
    cache=True
)
def cond_intensity(d, t_hist, _nx, n_mask, _t):
    """
    Probabilistic rate λn(t) of events in box n at time t
    :param d: PredPol parameters Dictonary
    :param t_hist: Data containing the timestamps of crimes
    :param _nx: index to grid location
    :param n_mask: Mask for t_hist that filters to a particular 150m^2 region
    :param _t: The time of the current event
    :return: λn(t)
    """
    _t_hist = t_hist[n_mask][:, 0][t_hist[n_mask][:, 0] < _t]  # Mask to events occurring before the current time
    # If the size of the remaining events is zero then return the background rate
    if _t_hist.size == 0:
        return d['u'][_nx]
    else:
        return d['u'][_nx] + d['theta'][0]*d['w'][0]*np.sum(np.exp(-d['w'][0]*(_t - _t_hist)))


# E-Step Functions======================================================================================================
@njit(
    types.float32(
        types.DictType(keyty=types.unicode_type, valty=float_array),
        types.float32[:, :],
        types.int64,
        types.boolean[:],
        types.int64,
        types.int64
    ),
    cache=True
)
def p_ij(d, t_hist, _nx, n_mask, i, j):
    """
    Probability that event j is a child of event i
    :param d: PredPol parameters Dictonary
    :param _nx: index to grid location
    :param t_hist: Data containing the timestamps of crimes in column 3
    :param n_mask: Mask for t_hist that filters to a particular 150m^2 region
    :param i: past event index
    :param j: current event index
    :return: P^{ij}_{n}
    """
    a = d['theta'][0]*d['w'][0] * np.exp(-d['w'][0]*(t_hist[n_mask][:, 0][j] - t_hist[n_mask][:, 0][i]))
    b = cond_intensity(d, t_hist, _nx, n_mask, t_hist[n_mask][:, 0][j])
    return np.divide(a, b)


@njit(
    types.float32(
        types.DictType(keyty=types.unicode_type, valty=float_array),
        types.float32[:, :],
        types.int64,
        types.boolean[:],
        types.int64
    ),
    cache=True
)
def p_j(d, t_hist, _nx, n_mask, j):
    """
    Probability that event j is due to poisson noise
    :param d: PredPol parameters Dictonary
    :param t_hist: Data containing the timestamps of crimes in column 3
    :param _nx: index to grid location
    :param n_mask: Mask for t_hist that filters to a particular 150m^2 region
    :param j: current event index
    :return: P^{j}_{n}
    """
    return np.divide(d['u'][_nx], cond_intensity(d, t_hist, _nx, n_mask, t_hist[n_mask][:, 2][j]))
# ======================================================================================================================


# M-Step Functions======================================================================================================
@njit(
    types.Tuple((
            types.float32,
            types.float32,
            types.float32
    ))(
        types.DictType(keyty=types.unicode_type, valty=float_array),
        types.float32[:, :],
        types.float32,
        types.float32,
        types.int64
    ),
    cache=True
)
def get_w_theta_split(d, t_hist, _j, dt, nx):
    """
    return the numerator and denominator for w and theta
    :param d: PredPol parameters Dictonary
    :param t_hist: Data containing the timestamps of crimes in column 3
    :param _j: current time
    :param dt: timestep
    :param nx: the maximum grid index
    :return: w
    """
    # Define numerator and denominators
    num = types.double(0.0)
    dom_w = types.double(0.0)
    dom_t = types.double(0.0)
    for _nx in _iter(start=types.int64(0), stop=nx):
        # Filter by grid location
        n_mask = (t_hist[:, 1] == _nx)
        # Filter by events that happen within the current time-step
        j_mask = (t_hist[n_mask][:, 0] > _j) * (t_hist[n_mask][:, 0] < _j+dt)

        ix = types.int64(np.sum(n_mask))  # Get the total of the mask this is the number of events in the grid location

        #  Loop over events occurring in the given grid cell
        for _jx in _iter(start=types.int64(0), stop=ix):
            #  Assert event j is in time-step
            if j_mask[_jx]:
                #  Loop over events occurring in the given grid cell
                for _ix in _iter(start=types.int64(0), stop=ix):
                    #  Assert event i is before the time-step
                    if (_jx != _ix) and (_j > t_hist[n_mask][:, 0][_ix]):
                        _p_ij = p_ij(d, t_hist, _nx, n_mask, _ix, _jx)  # Get the p_ij value
                        num += _p_ij  # Increment numerator
                        dt = (t_hist[n_mask][:, 0][_jx] - t_hist[n_mask][:, 0][_ix])  # Time delta between t and event
                        dom_w += _p_ij * dt  # Increment denominator
                        dom_t += 1  # Increment denominator for each event
    return num, dom_t, dom_w


@njit(
    types.float32[:](
        types.DictType(keyty=types.unicode_type, valty=float_array),
        types.float32[:, :],
        types.int64,
        types.int64,
        types.int64
    ),
    cache=True
)
def get_u(d, t_hist, _j, nx, t=1.0):
    """
    Update the value of u
    :param d: PredPol parameters Dictonary
    :param t_hist: Data
    :param _j: current event index
    :param nx: the maximum grid index
    :param t: Time interval
    :return: u
    """

    dom = t  # The denominator is the length of the observation interval
    u_out = np.full(nx, 1.0).astype(np.float32)  # Make the output array
    for _nx in _iter(start=types.int64(0), stop=nx):
        num = 0.0  # Sum for the numerator
        n_mask = (t_hist[:, 1] == _nx)  # Filter by grid location
        j_mask = (t_hist[n_mask][:, 0] > _j - t) * (t_hist[n_mask][:, 0] < _j)  # Filter by time step
        jx = np.sum(n_mask)  # Get the total of the mask this is the number of events in the grid location

        # Loop over the number of events in the grid location
        for _jx in _iter(start=types.int64(0), stop=jx):
            if not j_mask[_jx]:
                num += p_j(d, t_hist, _nx, n_mask, _jx)  # Increment numerator for each event
        if num > 0:
            u_out[_nx] = num  # If the numerator is more than 0 update the value
    u_out = np.divide(u_out, dom).astype(np.float32)
    return u_out
# ======================================================================================================================


# Griding Functions ====================================================================================================
@njit(
    types.int64(
        types.int64,
        types.int64,
        types.int64
    ),
    cache=True
)
def get_k(a, b, c):
    """
    Takes the i,j indexes of a square matrix side c and returns the corresponding k parameter used to search the
    n^2 matrix made with the outer product of flattend i,j
    :param a: i
    :param b: j
    :param c: side len
    :return: k
    """
    return (b % c) + (a * c)


@njit(
    types.Tuple((
        types.float32[:, :, :],
        types.float32[:],
        types.float32[:]
    ))(
        types.float32[:],
        types.float32[:],
        types.int64
    ),
    cache=True
)
def make_grid(lats, longs, devisions):
    """
    Take the lats & longs from data and use the min maxes to create a grid of x-y separations
    :param lats: latitude values
    :param longs: longitude values
    :param devisions: number of divisions to use N
    :return out: Matrix (N,N,2) this is the the x and y distances to the next point in the grid the bottom or furthest
    row are zeros where there is no further points.
    :return lat_space: latitude array N
    :return long_space: longitude array N
    """
    min_lat = np.nanmin(lats)
    min_long = np.nanmin(longs)
    max_lat = np.nanmax(lats)
    max_long = np.nanmax(longs)

    # Make repeating lat_long arrays ===================================================================================
    # Lats repeat min ... max, ..., min ... max
    lat_space = np.repeat(
        np.linspace(
            min_lat,
            max_lat,
            devisions),
        devisions
    ).astype(np.float32)
    # Longs repeat min ... min, ..., max ... max
    long_space = np.reshape(
        np.repeat(
            np.linspace(
                min_long,
                max_long,
                devisions),
            devisions
        ),
        (devisions, devisions)
    ).T.flatten().astype(np.float32)
    # ==================================================================================================================

    # Use the haversine formula to create distances between every lat long point =======================================
    distances_mat = ai_cjs.gf.haversine_dist(lat_space[:-1], lat_space[1:], long_space[:-1], long_space[1:])
    # ==================================================================================================================

    # Create an output array to fill with x separations (N,N) and y separations (N,N)===================================
    out = np.full((devisions, devisions, 2), 0.).astype(np.float32)
    # ==================================================================================================================

    # Loop over the extent of the arrays ===============================================================================
    _range = _iter(start=types.int64(0), stop=devisions)
    for i in _range:
        for j in _range:
            k = get_k(i, j, devisions)

            # latitude separations
            # drops the last N values
            if (k + 1) > devisions ** 2 - devisions:
                out[i, j, 0] = types.float32(-1.0)
            else:
                out[i, j, 0] = distances_mat[k, k + devisions - 1]

            # longitude separations
            # drops the devisions'th term
            if (k + 1) % devisions == 0:
                out[i, j, 1] = types.float32(-1.0)
            else:
                out[i, j, 1] = distances_mat[k, k]
    # ==================================================================================================================
    # Return the output longitude and latitude spacings
    return out, lat_space, long_space
# ======================================================================================================================


# Cartesian conversion==================================================================================================
def lat_long_to_x_y(_long_space, _lat_space, _x_sep, _y_sep, long_in, lat_in):
    """
    Uses the long-lat arrays and the x-y sep to generate an interpolator.
    Then puts the data into the interpolator to get the x-y coordinates.
    :param _long_space: longitude spacing for grid
    :param _lat_space: latitude spacing for grid
    :param _x_sep: longitude separations
    :param _y_sep: latitude separations
    :param long_in: data longitudes
    :param lat_in: data latitudes
    :return x_out: data x coordinates
    :return y_out: data y coordinates
    """

    # Take the cumulative sum of the separations to give distances from origin
    _x = np.cumsum(_x_sep, axis=1)
    _y = np.cumsum(_y_sep, axis=0)

    # Create 2d interpolation grids in lat long to make lookups
    x_interp = interp2d(_long_space, _lat_space, _x)
    y_interp = interp2d(_long_space, _lat_space, _y)

    # Make arrays to reorder outputs after the interp functions sort them
    x_sort = np.argsort(long_in)
    y_sort = np.argsort(lat_in)

    # Use the interp functions to get the x y coordinates for the input latitude longitude pairs
    x_out = x_interp(long_in, lat_in)
    y_out = y_interp(long_in, lat_in)

    # Sort and return arrays
    return (x_out[y_sort, x_sort]).astype(np.float32), (y_out[y_sort, x_sort]).astype(np.float32)
