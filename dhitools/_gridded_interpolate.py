"""
Gridded interpolation applied to dfsu unstructured mesh. Allows you to
downsample large mesh/dfsu files to something more manageable.

This improves scipy.interpolate.griddata as it computes the interpolation
weights (which don't change) rather than calculating the weights for each
point interpolation.
"""

import scipy.spatial.qhull as qhull
import numpy as np


def dfsu_to_grid(x, y, z, res=100):
    """
    Interpolate (x, y, z) node points to a regular gride at input resolution

    Parameters
    ----------
    x : ndarray, shape (num_nodes,)
        x node coordinates
    y : ndarray, shape (num_nodes,)
        y node coordinates
    z : ndarray, shape (num_nodes,)
        z node coordinates
    res : int
        Grid resolution

    Returns
    -------
    interp_z : ndarray, shape (len_xgrid, len_ygrid)
        Interpolated gridded z. Use dfsu_XY_meshgrid() to get gridded X and Y

    """
    # Sort out raster grid
    grid_x, grid_y = dfsu_XY_meshgrid(x, y, res)

    # Flattern grid to parse through custom interpolation methods
    all_p = np.column_stack((grid_x.flatten(), grid_y.flatten()))

    # Compute triangulation weights for node points
    xy = np.column_stack((x, y))
    vtx, wts = interp_weights(xy, all_p)

    # Interpolate points (to flatterned array) and then reshape
    interp_z_flat = interpolate(z, vtx, wts)
    interp_z = np.reshape(interp_z_flat, (grid_x.shape[0], grid_x.shape[1]))

    return interp_z


def interp_weights(xy, uv, d=2):
    """
    Calculate interpolation weights for all xy node points at uv gridded points

    Parameters
    ----------
    xy : ndarray, shape (num_nodes,2)
        Unstructured node (x, y) coordinates
    uv : ndarray, shape (len_xgrid * len_ygrid, 2)
        Flatted (x,y) coordinates for X and Y gridded points to calculate
        interpolation weights at
    d : int
        Number of dimensions

    Returns
    -------
    vertices : ndarray, shape (len_xgrid * len_ygrid, 3)
        Vertices for triangulation applied to (x, y)
    weights : ndarray, shape (len_xgrid * len_ygrid, 3)
        Weights for gridded X and Y
    """
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    return vertices, weights


def interpolate(input_z, vertices, weights, fill_value=None):
    """
    Interpolate z values to grid using vertices from node triangulation and
    weights at gridded points (these are from interp_weights())

    Parameters
    ----------
    input_z : ndarray, shape (num_nodes,)
        Z values to interpolate from at node coordinates
    vertices : ndarray, shape (len_xgrid * len_ygrid, 3)
        Vertices for triangulation applied to (x, y)
    weights : ndarray, shape (len_xgrid * len_ygrid, 3)
        Weights for gridded X and Y

    Returns
    -------
    interp_z : ndarray, shape (len_xgrid, len_ygrid)
        Interpolated grid points at X and Y meshgrid

    """
    if fill_value is None:
        fill_value = np.nan
    interp_z = np.einsum('nj,nj->n', np.take(input_z, vertices), weights)
    interp_z[np.any(weights < 0, axis=1)] = fill_value

    return interp_z


def dfsu_details(x, y):
    """
    Get min and max for input x and y ndarrays; shape (num_nodes,)
    """

    min_x = x.min()
    max_x = x.max()
    min_y = y.min()
    max_y = y.max()

    return min_x, max_x, min_y, max_y


def dfsu_XY_meshgrid(x, y, res=100):
    """
    Calculate X and Y meshgrid for input x and y node points.

    Returns grid_x ang grid_y of shape (len_xgrid, len_ygrid)
    """
    min_x, max_x, min_y, max_y = dfsu_details(x, y)
    grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x, res),
                                 np.arange(min_y, max_y, res))

    return grid_x, grid_y
