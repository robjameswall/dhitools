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


class Dfsu():

    def __init__(self, filename):

        self.read_dfsu(filename)

    def read_dfsu(self, filename):
        self.filename = filename
        self._obj = dfs.DfsFileFactory.DfsuFileOpen(filename)
        self.items = _dfsu_info(self._obj)
        self.projection = str(self._obj.Projection.WKTString)
        self.ele_table = _element_table(self._obj)
        self.NtoE = _node_to_element_table(self._obj, self.ele_table)
        self.nodes = _node_coordinates(self._obj)
        self.elements = _element_coordinates(self._obj)

    def summary(self):
        print(self.items)


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
