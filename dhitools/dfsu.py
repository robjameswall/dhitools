"""DHI MIKE21 dfsu functions
"""

# Author: Robert Wall

import numpy as np
import mesh
from dotenv import load_dotenv, find_dotenv
import os
import clr

load_dotenv(find_dotenv())
sdk_path = os.getenv('MIKE_SDK')
dfs_dll = os.getenv('MIKE_DFS')
clr.AddReference(os.path.join(sdk_path, dfs_dll))
clr.AddReference('System')

import DHI.Generic.MikeZero.DFS as dfs
import System
from System import Array
from System.Runtime.InteropServices import GCHandle, GCHandleType

import ctypes


class Dfsu():

    def __init__(self, filename):

        self.read_dfsu(filename)

    def read_dfsu(self, filename):
        self.filename = filename
        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(filename)
        self.items = _dfsu_info(dfsu_object)
        self.projection = str(dfsu_object.Projection.WKTString)
        self.ele_table = _element_table(dfsu_object)
        self.NtoE = _node_to_element_table(dfsu_object, self.ele_table)
        self.nodes = _node_coordinates(dfsu_object)
        self.elements = _element_coordinates(dfsu_object)

        dfsu_object.Close()

    def summary(self):
        print(self.items)

    def get_element_data(self, item_name, tstep_start=None, tstep_end=None,
                         element_list=None):
        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        ele_data = _element_data(dfsu_object=dfsu_object, item_name=item_name,
                                 item_info=self.items, tstep_start=tstep_start,
                                 tstep_end=tstep_end, element_list=element_list)
        dfsu_object.Close()
        return ele_data


def _dfsu_info(dfsu_object):

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


'''
Underlying dfsu mesh node and element functions
'''


def _element_table(dfsu_object):
    ele_table_object = dfsu_object.ElementTable
    element_table = np.zeros(shape=(dfsu_object.NumberOfElements,3), dtype='int')
    for i, e in enumerate(ele_table_object):
        element_table[i,:] = [e[0], e[1], e[2]]
    return element_table


def _node_to_element_table(dfsu_object, ele_table):
    num_nodes = dfsu_object.NumberOfNodes
    num_elements = dfsu_object.NumberOfElements

    # Set placeholders for constructing node-to-element-table (NtoE)
    e = np.arange(num_elements)
    u = np.ones(num_elements)
    I = np.concatenate((e, e, e))
    J = np.concatenate((ele_table[:,0],ele_table[:,1],ele_table[:,2]))
    K = np.concatenate((u*1, u*2, u*3))

    # Construct NtoE
    count = np.zeros((num_nodes,1))
    for i in range(len(I)):
        count[J[i-1]-1] = count[J[i-1]-1]+1
    num_cols = int(count.max())

    NtoE = np.zeros((num_nodes,num_cols), dtype='int')
    count = np.zeros((num_nodes,1))
    for i in range(len(I)):
        count[J[i-1]-1] = count[J[i-1]-1]+1
        NtoE[J[i-1]-1, int(count[J[i-1]-1])-1] = I[i]

    return NtoE


def _node_coordinates(dfsu_object):
    xn = np.array([point for point in dfsu_object.X])
    yn = np.array([point for point in dfsu_object.Y])
    zn = np.array([point for point in dfsu_object.Z])

    node_coordinates = np.column_stack([xn, yn, zn])
    return node_coordinates


def _element_coordinates(dfsu_object):
    element_coordinates = np.zeros(shape=(dfsu_object.NumberOfElements, 3))

    # Convert nodes to .NET System double to input to method
    xtemp = Array.CreateInstance(System.Double, 0)
    ytemp = Array.CreateInstance(System.Double, 0)
    ztemp = Array.CreateInstance(System.Double, 0)

    # Get element coords
    elemts_temp = dfs.dfsu.DfsuUtil.CalculateElementCenterCoordinates(dfsu_object, xtemp, ytemp, ztemp)

    # Place in array; need to get from .NET Array to numpy array
    for n in range(3):
        ele_coords_temp = []
        for ele in elemts_temp[n+1]:
            ele_coords_temp.append(ele)
        element_coordinates[:, n] = ele_coords_temp

    return element_coordinates


'''
Read item node and element data
'''


def _single_to_ndarray(system_single):
    '''
    Converts data read in with dotNet (i.e. reading dfsu through DHI .NET
    libraries) and efficiently converts System.Single[,] to numpy ndarray

    Inputs:
        - System.Single[,] : .NET Framwork single-precision float

    Outputs:
        - ndarray
    '''

    src_hndl = GCHandle.Alloc(system_single, GCHandleType.Pinned)

    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        bufType = ctypes.c_float*len(system_single)
        cbuf = bufType.from_address(src_ptr)
        ndarray = np.frombuffer(cbuf, dtype=cbuf._type_)
    finally:
        if src_hndl.IsAllocated:
            src_hndl.Free()
    return ndarray


def _element_data(dfsu_object, item_name, item_info,
                  tstep_start=None, tstep_end=None,
                  element_list=None):

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
            ele_data[:,i] = _single_to_ndarray(dfsu_object.ReadItemTimeStep(item_idx, t).Data)[element_list]
        else:
            ele_data[:,i] = _single_to_ndarray(dfsu_object.ReadItemTimeStep(item_idx, t).Data)

    return ele_data
