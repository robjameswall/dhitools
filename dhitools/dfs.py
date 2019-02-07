"""DHI MIKE21 dfs0/1/2 functions
"""

import numpy as np
import datetime as dt
import pandas as pd
from . import _utils
from . import config
import os
import clr

# Set path to MIKE SDK
sdk_path = config.MIKE_SDK
dfs_dll = config.MIKE_DFS
eum_dll = config.MIKE_EUM
clr.AddReference(os.path.join(sdk_path, dfs_dll))
clr.AddReference(os.path.join(sdk_path, eum_dll))

# Import .NET libraries
import DHI.Generic.MikeZero.DFS as dfs


class _Dfs(object):
    """
    Function description...

    Parameters
    ----------
    input_1 : dtype, shape (n_components,)
        input_1 description...
    input_2 : int
        input_2 description...

    Returns
    -------
    weights : array, shape (n_components,)

    """

    def __init__(self, dfs_object):
        self.items = self.dfs_info(dfs_object)

        self.time = self.dfs_time()

    def dfs_info(self, dfs_object):
        items = {}
        itemnames = [[n.Name, n.Quantity.UnitAbbreviation] for n in dfs_object.ItemInfo]
        time_obj = dfs_object.FileInfo.TimeAxis
        dt_start_obj = time_obj.StartDateTime
        items['num_timesteps'] = time_obj.NumberOfTimeSteps
        self.number_tstep = items['num_timesteps']
        self.num_items = dfs_object.ItemInfo.Count
        self.timestep = time_obj.TimeStep
        self.start_datetime = dt.datetime(year=dt_start_obj.Year,
                                          month=dt_start_obj.Month,
                                          day=dt_start_obj.Day,
                                          hour=dt_start_obj.Hour,
                                          minute=dt_start_obj.Minute,
                                          second=dt_start_obj.Second)
        self.end_datetime = self.start_datetime + \
            dt.timedelta(seconds=self.timestep * self.number_tstep)
        items['names'] = []

        for ind, it in enumerate(itemnames):
            # Create key from itemname and add to dictionary
            itemName = str(it[0])
            itemUnit = str(it[1])
            items[itemName] = {}
            items[itemName]['unit'] = itemUnit
            items[itemName]['index'] = ind
            items['names'].append(itemName)

        return items

    def dfs_time(self):
        time = np.arange(self.start_datetime,
                         self.end_datetime,
                         dt.timedelta(seconds=self.timestep)).astype(dt.datetime)
        return time

    def summary(self):
        """
        Prints a summary of the dfs
        """
        print("Input file: {}".format(self.filename))
        print("Time start = {}".format(dt.datetime.strftime(self.start_datetime,
                                                            "%d/%m/%Y %H:%M:%S")))
        print("Number of timesteps = {}".format(self.number_tstep))
        print("Timestep = {}".format(self.timestep))
        print("\n")
        print("Number of items = {}".format(self.num_items))

        if(self.filename.endswith(".dfs1")):
            print("number of profile points = {}".format(self.num_points))

        print("Items:")
        for n in self.items['names']:
            print("{}, unit = {}, index = {}".format(n, self.items[n]['unit'],
                                                     self.items[n]['index']))


class Dfs0(_Dfs):
    """
    Function description...

    Parameters
    ----------
    input_1 : dtype, shape (n_components,)
        input_1 description...
    input_2 : int
        input_2 description...

    Returns
    -------
    weights : array, shape (n_components,)

    """

    def __init__(self, filename):
        self.filename = filename
        dfs0_object = dfs.DfsFileFactory.DfsGenericOpen(self.filename)
        super(Dfs0, self).__init__(dfs0_object)
        self.data = self.read_dfs0(dfs0_object)

    def read_dfs0(self, dfs0_object, close=True):
        """
        Function description...

        Parameters
        ----------
        input_1 : dtype, shape (n_components,)
            input_1 description...
        input_2 : int
            input_2 description...

        Returns
        -------
        weights : array, shape (n_components,)

        """
        out_arr = np.zeros((self.number_tstep, self.num_items))

        for i in range(self.number_tstep):
            for j in range(self.num_items):
                item_data = dfs0_object.ReadItemTimeStep(j + 1, i)
                out_arr[i,j] = item_data.Data[0]

        out_df = pd.DataFrame(data=out_arr, columns=self.items['names'],
                              index=self.time)

        if close:
            dfs0_object.Close()

        return out_df


class Dfs1(_Dfs):
    """
    Function description...

    Parameters
    ----------
    input_1 : dtype, shape (n_components,)
        input_1 description...
    input_2 : int
        input_2 description...

    Returns
    -------
    weights : array, shape (n_components,)

    """

    def __init__(self, filename):
        self.filename = filename
        dfs1_object = dfs.DfsFileFactory.Dfs1FileOpen(self.filename)
        self.num_points = len(dfs1_object.ReadItemTimeStep(1,0).Data)
        super(Dfs1, self).__init__(dfs1_object)
        self.read_dfs1(dfs1_object)

    def read_dfs1(self, dfs1_object, close=True):
        """
        Function description...

        Parameters
        ----------
        input_1 : dtype, shape (n_components,)
            input_1 description...
        input_2 : int
            input_2 description...

        Returns
        -------
        weights : array, shape (n_components,)

        """
        for itemname in self.items['names']:
            item_idx = self.items[itemname]['index'] + 1
            out_arr = np.zeros((self.number_tstep, self.num_points))

            for i in range(self.number_tstep):
                out_arr[i,:] = _utils.dotnet_arr_to_ndarr(dfs1_object.ReadItemTimeStep(item_idx, i).Data)
            out_df = pd.DataFrame(data=out_arr, index=self.time)

        self.items[itemname]["data"] = out_df

        if close:
            dfs1_object.Close()

    def item_data(self, item_name):
        """
        Function description...

        Parameters
        ----------
        input_1 : dtype, shape (n_components,)
            input_1 description...
        input_2 : int
            input_2 description...

        Returns
        -------
        weights : array, shape (n_components,)

        """

        return self.items[item_name]["data"]


class Dfs2(_Dfs):
    """
    Function description...

    Parameters
    ----------
    input_1 : dtype, shape (n_components,)
        input_1 description...
    input_2 : int
        input_2 description...

    Returns
    -------
    weights : array, shape (n_components,)

    """

    def __init__(self, filename):
        self.filename = filename
        dfs2_object = dfs.DfsFileFactory.Dfs2FileOpen(self.filename)
        self.num_points = len(dfs2_object.ReadItemTimeStep(1,0).Data)
        super(Dfs2, self).__init__(dfs2_object)
        self.read_dfs2(dfs2_object)

    def read_dfs2(self, dfs2_object, close=True):
        sa = dfs2_object.SpatialAxis
        fi = dfs2_object.FileInfo
        self.projection = str(fi.Projection.WKTString)
        self.x_min = sa.X0
        self.del_x = sa.Dx
        self.x_count = sa.XCount
        self.x_max = self.x_min + (self.del_x * self.x_count)
        self.y_min = sa.Y0
        self.del_y = sa.Dy
        self.y_count = sa.YCount
        self.y_max = self.y_min + (self.del_y * self.y_count)
        self.gridshape = (self.y_count, self.x_count)

        self.X, self.Y = np.meshgrid(np.arange(self.x_min, self.x_max, self.del_x),
                                     np.arange(self.y_min, self.y_max, self.del_y))

        # No data values
        self.nodata_float = fi.DeleteValueFloat
        self.nodata_double = fi.DeleteValueDouble
        self.nodata_int = fi.DeleteValueInt

        if close:
            dfs2_object.Close()

    def item_data(self, item_name, tstep_start=None, tstep_end=None):
        """
        Function description...

        Parameters
        ----------
        input_1 : dtype, shape (n_components,)
            input_1 description...
        input_2 : int
            input_2 description...

        Returns
        -------
        weights : array, shape (n_components,)

        """
        dfs2_object = dfs.DfsFileFactory.Dfs2FileOpen(self.filename)
        data = _item_data(dfs2_object=dfs2_object, item_name=item_name,
                          item_info=self.items, tstep_start=tstep_start,
                          tstep_end=tstep_end, gridshape=self.gridshape)
        dfs2_object.Close()

        return data


def _item_data(dfs2_object, item_name, item_info, gridshape,
               tstep_start=None, tstep_end=None):
    """ Read specified item_name dfs2 data """

    item_idx = item_info[item_name]['index'] + 1
    if tstep_start is None:
        tstep_start = 0

    if tstep_end is None:
        # Only get one tstep specified by tstep_start
        tstep_end = tstep_start + 1
    elif tstep_end == -1:
        # Get from tstep_start to the end
        tstep_end = item_info['num_timesteps']
    else:
        # Add one to include tstep_end in output
        tstep_end += 1

    t_range = range(tstep_start, tstep_end)
    ndshape = gridshape + (len(t_range),)
    data = np.zeros(shape=(ndshape))
    for i, t in enumerate(t_range):
        data[:,:,i] = _utils.dotnet_arr_to_ndarr(dfs2_object.ReadItemTimeStep(item_idx, t).Data).reshape(gridshape)

    return data
