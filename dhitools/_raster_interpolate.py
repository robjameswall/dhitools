"""Raster interpolation functions

Depends on GDAL/OGR
"""

# Author: Robert Wall

import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from scipy.interpolate import RegularGridInterpolator


def raster_details(input_raster):
    '''
    Read GDAL supported raster format with elevation data in first band and
    return details:

    Parameters
    ----------
    input_raster : str
        filepath to raster

    Returns
    -------
    xres : int
        x grid spacing
    yres : int
        y grid spacing
    xmin : int
        smallest x value
    ymin : int
        smallest y value
    x_size : int
        number of x columns
    y_size : int
        number of y columns
    nodata : int
        value assigned to nodata
    '''
    raster = gdal.Open(input_raster)
    ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
    xres, yres = abs(xres), abs(yres)
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize
    nodata = raster.GetRasterBand(1).GetNoDataValue()
    xmin = ulx
    xmax = ulx + x_size * xres
    ymax = uly
    ymin = uly - y_size * yres

    return xres, yres, xmin, xmax, ymin, ymax, x_size, y_size, nodata


def raster_XYZ(input_raster):
    '''
    Read GDAL supported raster format with elevation data in first band and
    return data as np array:
    
    Parameters
    ----------
    input_raster : str
        filepath to raster

    Returns
    -------
    x : ndarray
        Raster x values
    y : ndarray
        Raster y values
    XYZ : ndarray, shape (3, meshgrid(x, y))
        Meshgrids of X, Y, Z
    '''
    xres, yres, xmin, xmax, ymin, ymax, x_size, y_size, nodata = raster_details(input_raster)
    x = np.linspace(xmin, xmax, x_size)
    y = np.linspace(ymin, ymax, y_size)
    Z = gdal_array.LoadFile(input_raster)
    Z[Z == nodata] = np.nan
    Z = np.flipud(Z)
    X, Y = np.meshgrid(x, y)
    XYZ = np.array([X, Y, Z])

    return x, y, XYZ


def interpolate_from_raster(input_raster, xy_array_to_interpolate,
                            method='nearest'):
    '''
    Interpolate input_raster to xy array.

    Parameters
    ----------
    input_raster : str
        filepath to raster
    xy_array : ndarray, shape (num_x, num_y)
        Node values to interpolate z at
    
    Returns
    -------
    interp_z : ndarray, shape (len(xy_array))
        Interpolated z values as xy_array (x, y)

    See Also
    --------

    See scipy.interpolate.RegularGridInterpolator documentation for further
    details.
    '''
    x_raster, y_raster, raster_grid = raster_XYZ(input_raster)
    interp_f = RegularGridInterpolator((y_raster, x_raster),
                                       raster_grid[2],
                                       bounds_error=False,
                                       fill_value=np.nan)
    # Need to flip
    xy_flipped = np.fliplr(xy_array_to_interpolate)
    interp_z = interp_f(xy_flipped, method=method)

    return interp_z
