"""DHI MIKE21 dfsu functions

Author: Robert Wall

"""

import numpy as np
import mesh
import _utils
from dotenv import load_dotenv, find_dotenv
import os
import clr

# Set path to MIKE SDK
load_dotenv(find_dotenv())
sdk_path = os.getenv('MIKE_SDK')
dfs_dll = os.getenv('MIKE_DFS')
clr.AddReference(os.path.join(sdk_path, dfs_dll))
clr.AddReference('System')

# Import .NET libraries
import DHI.Generic.MikeZero.DFS as dfs

class Dfsu(mesh.Mesh):
    """
    MIKE21 .dfsu

    Parameters
    ----------
    filename : str
        Path to .dfsu

    Attributes
    ----------
    filename : str
        Path to .dfsu
    items : dict
        List .dfsu items (ie. surface elevation, current speed), item index
        to lookup in .dfsu, item units and counts of elements, nodes and
        time steps.
    projection : str
        .dfsu spatial projection string in WKT format
    ele_table : ndarray, shape (num_ele, 3)
        Defines for each element the nodes that define the element.
    node_table : ndarray, shape (num_nodes, n)
        Defines for each node the element adjacent to this node. May contain
        padded zeros
    nodes : ndarray, shape (num_nodes, 3)
        (x,y,z) coordinate for each node
    elements : ndarray, shape (num_ele, 3)
        (x,y,z) coordinate for each element

    See Also
    ----------
    Many of these methods have been adapated from the DHI MATLAB Toolbox:
        https://www.mikepoweredbydhi.com/download/mike-by-dhi-tools/
        coastandseatools/dhi-matlab-toolbox
    """

    def __init__(self, filename=None):
        super(Dfsu, self).__init__(filename)
        if self.filename is not None:
            dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
            self.items = _dfsu_info(dfsu_object)
            dfsu_object.Close()

    def read_dfsu(self, filename):
        """ Read in .dfsu file and read attributes """
        self.read_mesh(filename)

    def summary(self):
        print("Input .dfsu file: {}".format(self.filename))
        print("Num. Elmts = {}".format(self.num_elements))
        print("Num. Nodes = {}".format(self.num_nodes))
        print("Mean elevation = {}".format(np.mean(self.nodes[:, 2])))
        print("Projection = \n {}".format(self.projection))
        print("\n")
        print("Number of items = {}".format(len(self.items) - 3))
        print("Items:")
        for k in self.items.keys():
            if k not in ['num_elements', 'num_nodes', 'num_timesteps']:
                print("{}, unit = {}, index = {}".format(k,
                                                         self.items[k]['unit'],
                                                         self.items[k]['index']))

    def item_element_data(self, item_name, tstep_start=None, tstep_end=None,
                          element_list=None):
        """
        Get element data for specified item with option to specify range of
        timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for element data. Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for element data. Allows for range of time
            steps to be returned, where `tstep_end` is included.Must be
            positive int <= number of timesteps
            If `None`, returns single time step specified by `tstep_start`
            If `-1`, returns all time steps from `tstep_start`:end
        element_list : list, optional
            Provide list of elements. Element numbers are as seen by MIKE
            programs and adjusted for Python indexing.

        Returns
        -------
        ele_data : ndarray, shape (num_elements,[tstep_end-tstep_start])
            Element data for specified item and time steps
            `element_list` will change num_elements returned in `ele_data`

        """
        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        ele_data = _element_data(dfsu_object=dfsu_object, item_name=item_name,
                                 item_info=self.items, tstep_start=tstep_start,
                                 tstep_end=tstep_end, element_list=element_list)
        dfsu_object.Close()

        return ele_data

    def item_node_data(self, item_name, tstep_start=None, tstep_end=None):
        """
        Get node data for specified item with option to specify range of
        timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return node data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for node data. Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for node data. Allows for range of time
            steps to be returned, where `tstep_end` is included.Must be
            positive int <= number of timesteps
            If `None`, returns single time step specified by `tstep_start`
            If `-1`, returns all time steps from `tstep_start`:end

        Returns
        -------
        node_data : ndarray, shape (num_elements,[tstep_end-tstep_start])
            Node data for specified item and time steps
        """
        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        node_data = _node_data(dfsu_object=dfsu_object, item_name=item_name,
                               item_info=self.items, ele_cords=self.elements,
                               node_cords=self.nodes, node_table=self.node_table,
                               tstep_start=tstep_start, tstep_end=tstep_end)
        dfsu_object.Close()

        return node_data

    def max_item(self, item_name, tstep_start=None, tstep_end=None,
                 current_dir=False):
        """
        Calculate maximum element value for specified item over entire model or
        within specific range of timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for data considered in determining maximum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining maximum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        current_dir : boolean
            If True, returns corresponding current direction value occuring at
            the maxmimum of specified `item_name`.

        Returns
        -------
        If `current_dir` is False:
        max_ele : ndarray, shape (num_elements,)
            Maximum elements values for specified item

        If `current_dir` is True
        max_ele : ndarray, shape (num_elements,)
            Maximum elements values for specified item
        max_current_dir : ndarray, shape (num_elements,)
            Current direction corresponding to `max_ele`

        """
        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        max_ele = _item_aggregate_stats(dfsu_object, item_name,
                                        self.items, tstep_start=tstep_start,
                                        tstep_end=tstep_end,
                                        current_dir=current_dir)
        dfsu_object.Close()

        if current_dir:
            return max_ele[0], max_ele[1]
        else:
            return max_ele

    def min_item(self, item_name, tstep_start=None, tstep_end=None,
                 current_dir=False):
        """
        Calculate minimum element value for specified item over entire model or
        within specific range of timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for data considered in determining minimum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining minimum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        current_dir : boolean
            If True, returns corresponding current direction value occuring at
            the maxmimum of specified `item_name`.

        Returns
        -------
        If `current_dir` is False:
        min_ele : ndarray, shape (num_elements,)
            Minimum elements values for specified item

        If `current_dir` is True
        min_ele : ndarray, shape (num_elements,)
            Minimum elements values for specified item
        max_current_dir : ndarray, shape (num_elements,)
            Current direction corresponding to `min_ele`

        """

        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        min_ele = _item_aggregate_stats(dfsu_object, item_name,
                                        self.items, tstep_start=tstep_start,
                                        tstep_end=tstep_end, return_max=False,
                                        current_dir=current_dir)
        dfsu_object.Close()

        if current_dir:
            return min_ele[0], min_ele[1]
        else:
            return min_ele


def _dfsu_info(dfsu_object):
    """
    Make a dictionary with .dfsu items and other attributes.

    See class attributes
    """
    itemnames = [[n.Name, n.Quantity.UnitAbbreviation] for n in dfsu_object.ItemInfo]
    items = {}

    for ind, it in enumerate(itemnames):

        # Create key from itemname and add to dictionary
        itemName = str(it[0])
        itemUnit = str(it[1])
        items[itemName] = {}
        items[itemName]['unit'] = itemUnit
        items[itemName]['index'] = ind

    items['num_timesteps'] = dfsu_object.NumberOfTimeSteps
    items['num_nodes'] = dfsu_object.NumberOfNodes
    items['num_elements'] = dfsu_object.NumberOfElements

    dfsu_object.Close()

    return items


"""
Read item node and element data
"""


def _element_data(dfsu_object, item_name, item_info,
                  tstep_start=None, tstep_end=None,
                  element_list=None):
    """ Read specified item_name element data """
    if element_list:
        # Subtract zero to match Python idx'ing
        element_list = [e-1 for e in element_list]

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
    if element_list:
        ele_data = np.zeros(shape=(len(element_list), len(t_range)))
    else:
        ele_data = np.zeros(shape=(item_info['num_elements'], len(t_range)))
    for i, t in enumerate(t_range):
        if element_list:
            ele_data[:,i] = _utils.dotnet_arr_to_ndarr(dfsu_object.ReadItemTimeStep(item_idx, t).Data)[element_list]
        else:
            ele_data[:,i] = _utils.dotnet_arr_to_ndarr(dfsu_object.ReadItemTimeStep(item_idx, t).Data)

    return ele_data


def _node_data(dfsu_object, item_name, item_info,
               ele_cords, node_cords, node_table,
               tstep_start=None, tstep_end=None):
    """ Read specified item_name node data """

    # Get item_name element data
    ele_data = _element_data(dfsu_object, item_name, item_info,
                             tstep_start, tstep_end)

    # Get item_name node data
    node_data = np.zeros(shape=(len(node_cords), ele_data.shape[1]))
    for i in range(ele_data.shape[1]):
        node_data[:,i] = _map_ele_to_node(node_table, ele_cords, node_cords, ele_data[:,i])

    return node_data


def _interp_node_z(nn,node_table,xe,ye,ze,xn,yn):
    """
    Calculate value at node (xn,yn) from element center values (xe, ye, ze).

    Attempts to use Psuedo Lapalce procedure by [Holmes, Connel 1989]. If this
    fails, uses an inverse distance average.

    Parameters
    ----------
    nn : int
        Node number to solve node value.
    node_table : ndarray, shape (num_nodes, n)
        Defines for each node the element adjacent to this node. May contain
        padded zeros
    xe : ndarray, shape (num_elements, 1)
        Element x vector
    ye : ndarray, shape (num_elements, 1)
        Element y vector
    ze : ndarray, shape (num_elements, 1)
        Element x vector
    xn : ndarray, shape (num_nodes, 1)
        Node x vector
    yn : ndarray, shape (num_nodes, 1)
        Node x vector

    Returns
    -------
    weights : array, shape (n_components,)

    See Also
    -------
    DHI MIKE MATLAB Toolbox; specifically `mzCalcNodeValuePL.m`

    Holmes, D. G. and Connell, S. D. (1989), Solution of the
        2D Navier-Stokes on unstructured adaptive grids, AIAA Pap.
        89-1932 in Proc. AIAA 9th CFD Conference.
    """
    nelmts = len(np.where(node_table[nn,:] != 0)[0])

    if nelmts < 1:
        zn = np.nan
        return zn

    Rx = 0
    Ry = 0
    Ixx = 0
    Iyy = 0
    Ixy = 0

    for i in range(nelmts):
        el_id = int(node_table[nn,i]-1)
        dx = xe[el_id] - xn[nn]
        dy = ye[el_id] - yn[nn]
        Rx = Rx + dx
        Ry = Ry + dy
        Ixx = Ixx + dx*dx
        Iyy = Iyy + dy*dy
        Ixy = Ixy + dx*dy

    lamda = Ixx*Iyy - Ixy*Ixy

    # Pseudo laplace procedure
    if abs(lamda) > 1e-10*(Ixx*Iyy):
        lamda_x = (Ixy*Ry - Iyy*Rx)/lamda
        lamda_y = (Ixy*Rx - Ixx*Ry)/lamda

        omega_sum = float(0)
        zn = float(0)

        for i in range(nelmts):
            el_id = int(node_table[nn,i]-1)

            omega = 1 + lamda_x*(xe[el_id]-xn[nn]) + lamda_y*(ye[el_id]-yn[nn])
            if omega < 0:
                omega = 0
            elif omega > 2:
                omega = 2
            omega_sum = omega_sum + omega
            zn = zn + omega*ze[el_id]

        if abs(omega_sum) > 1e-10:
            zn = zn/omega_sum
        else:
            omega_sum = float(0)
    else:
        omega_sum = float(0)

    # If not successful use inverse distance average
    if omega_sum == 0:
        zn = 0

        for i in range(nelmts):
            el_id = int(node_table[nn,i]-1)

            dx = xe[el_id] - xn[nn]
            dy = ye[el_id] - yn[nn]

            omega = float(1) / np.sqrt(dx*dx+dy*dy)
            omega_sum = omega_sum + omega
            zn = zn + omega*ze[el_id]

        if omega_sum != 0:
            zn = zn/omega_sum
        else:
            zn = float(0)

    return zn


def _map_ele_to_node(node_table, element_coordinates, node_coordinates,
                     element_data):
    """
    Get node data relating to specific element
    """
    xn = node_coordinates[:,0]
    yn = node_coordinates[:,1]
    xe = element_coordinates[:,0]
    ye = element_coordinates[:,1]

    zn = np.zeros(len(xn))

    for i in range(len(xn)):
        zn[i] = _interp_node_z(i,node_table,xe,ye,element_data,xn,yn)

    return zn


"""
dfsu stats
"""


def _item_aggregate_stats(dfsu_object, item_name, item_info, return_max=True,
                          tstep_start=None, tstep_end=None, current_dir=False):
    """
    Return max or min for input item across entire model or specific time range
    """
    item_idx = item_info[item_name]['index'] + 1
    ele_data = np.zeros((item_info['num_elements']))

    # If current_dir provided, get current dir at input item_name max/min
    if current_dir:
        cd_index = item_info['Current direction']['index'] + 1
        cd_ele_data = np.zeros((item_info['num_elements']))

    # Sort time range
    if tstep_start is None:
        tstep_start = 0

    if tstep_end is None:
        # Get from tstep_start to the end
        tstep_end = item_info['num_timesteps']
    else:
        # Add one to include tstep_end in output
        tstep_end += 1

    for tstep in range(tstep_start, tstep_end):
        # Iterate tstep in time range
        item_data = _utils.dotnet_arr_to_ndarr(dfsu_object.ReadItemTimeStep(item_idx, tstep).Data)

        # Determine elements to update
        if return_max:
            comp_boolean = np.greater(item_data, ele_data)
        else:
            comp_boolean = np.less(item_data, ele_data)

        # Update elements which have new extreme
        update_elements = item_data[comp_boolean]
        ele_data[comp_boolean] = update_elements

        # Update current_dir if specified
        if current_dir:
            cd_data = _utils.dotnet_arr_to_ndarr(dfsu_object.ReadItemTimeStep(cd_index, tstep).Data)
            update_cd_elements = cd_data[comp_boolean]
            cd_ele_data[comp_boolean] = update_cd_elements

    if current_dir:
        # Return both item_name data and current_dir data
        return ele_data, cd_ele_data
    else:
        # Else just item_name data
        return ele_data
