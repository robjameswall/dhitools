"""DHI MIKE21 mesh functions
"""

# Author: Robert Wall

import numpy as np
import geopandas as gpd
import datetime as dt
from dotenv import load_dotenv, find_dotenv
import os
import clr

# Set path to MIKE SDK
load_dotenv(find_dotenv())
sdk_path = os.getenv('MIKE_SDK')
dfs_dll = os.getenv('MIKE_DFS')
eum_dll = os.getenv('MIKE_EUM')
clr.AddReference(os.path.join(sdk_path, dfs_dll))
clr.AddReference(os.path.join(sdk_path, eum_dll))
clr.AddReference('System')

# Import .NET libraries
import DHI.Generic.MikeZero.DFS as dfs
from DHI.Generic.MikeZero import eumUnit, eumItem, eumQuantity
import System
from System import Array

import _utils


class Mesh(object):
    """
    MIKE21 .mesh

    Parameters
    ----------
    filename : str
        Path to .mesh

    Attributes
    ----------
    filename : str
        Path to .mesh
    nodes : ndarray, shape (num_nodes, 3)
        (x,y,z) coordinate for each node
    elements : ndarray, shape (num_ele, 3)
        (x,y,z) coordinate for each element
    ele_table : ndarray, shape (num_ele, 3)
        Defines for each element the nodes that define the element.
    node_table : ndarray, shape (num_nodes, n)
        Defines for each node the element adjacent to this node. May contain
        padded zeros
    node_ids : ndarray, shape (num_nodes, )
        Ordered node ids
    node_boundary_code : ndarray, shape (num_nodes, )
        Each nodes boundary code
    element_ids : ndarray, shape (num_elements, )
        Ordered element ids
    num_nodes : int
        Number of nodes elements
    num_elements : int
        Number of mesh elements
    projection : str
        .mesh spatial projection string in WKT format
    zUnitKey : int
        EUM unit designating quantity of Z variable
        1000 = metres
        1014 = U.S. feet

    See Also
    ----------
    Many of these methods have been adapated from the DHI MATLAB Toolbox:
        https://www.mikepoweredbydhi.com/download/mike-by-dhi-tools/
        coastandseatools/dhi-matlab-toolbox
    """

    def __init__(self, filename=None):
        self.filename = filename
        self._file_input = False
        self.zUnitKey = 1000  # Default value (1000 = meter)
        self.lyrs = {}  # Dict for model input layers ie. roughness

        if filename is not None:
            self.read_mesh()

    def read_mesh(self, filename=None):
        '''
        Read in .mesh file
        '''
        if filename is None:
            filename = self.filename

        if filename.endswith('.mesh'):
            dfs_obj = dfs.mesh.MeshFile.ReadMesh(filename)
            self.projection = str(dfs_obj.ProjectionString)
            self.zUnitKey = dfs_obj.EumQuantity.Unit
        elif filename.endswith('.dfsu'):
            dfs_obj = dfs.DfsFileFactory.DfsuFileOpen(filename)
            self.projection = str(dfs_obj.Projection.WKTString)
            self.zUnitKey = dfs_obj.get_ZUnit()

        mesh_in = _read_mesh(dfs_obj)

        self.nodes = mesh_in[0]
        self.node_ids = mesh_in[1]
        self.node_boundary_codes = mesh_in[2]
        self.element_table = mesh_in[3]
        self.node_table = mesh_in[4]
        self.elements = mesh_in[5]
        self.element_ids = mesh_in[6]
        self._file_input = True
        self.num_elements = len(self.elements)
        self.num_nodes = len(self.nodes)

        if filename.endswith('.dfsu'):
            dfs_obj.Close()

    def summary(self):
        '''
        Summarise .mesh file
        '''
        if self._file_input:
            print("Input mesh file: {}".format(self.filename))
        else:
            print("No .mesh input file")

        try:
            print("Num. Elmts = {}".format(self.num_elements))
            print("Num. Nodes = {}".format(self.num_nodes))
            print("Mean elevation = {}".format(np.mean(self.nodes[:, 2])))
            print("Projection = \n {}".format(self.projection))
        except AttributeError:
            print("Object has no element or node properties. Read in mesh.")

    def write_mesh(self, output_name):
        '''
        Write new mesh file
        '''
        _write_mesh(filename=output_name,
                    nodes=self.nodes,
                    node_id=self.node_ids,
                    node_boundary_code=self.node_boundary_codes,
                    element_table=self.element_table,
                    element_ids=self.element_ids,
                    proj=self.projection,
                    zUnitKey=self.zUnitKey)

    def interpolate_rasters(self, raster_list, method='nearest'):
        '''
        Interpolate multiple raster elevations to mesh nodes

        Depends on GDAL/OGR
        '''
        import _raster_interpolate as _ri

        for r in raster_list:
            interp_z = _ri.interpolate_from_raster(r, self.nodes[:, :2], method)

            # Just consider nodes that overlay raster
            # Prepend node_ids
            updated_z = np.column_stack((self.node_ids, interp_z))

            # Sort by node_id
            updated_z_sorted = updated_z[updated_z[:, 0].argsort()]

            # Boolean mask for only updated nodes
            updated_bool = ~np.isnan(updated_z_sorted[:, 1])

            # Drop NaN
            only_updated_z = updated_z_sorted[:, 1][~np.isnan(updated_z_sorted[:, 1])]

            # Update mesh obj nodes only where node was interpolated
            self.nodes[:, 2][updated_bool] = only_updated_z

    def to_gpd(self, elements=True, output_shp=None):
        '''
        Export mesh elements or nodes to GeoDataFrame with option to write to
        shape file

        Parameters
        ----------
        elements : boolean
            if True, export element points
            if False, export nodes points
        output_shp : str, optional
            output path to write to .shp file

        Returns
        -------
        mesh_df : GeoDataFrame, shape (nrows, 2)
            Geopandas df with field for element or node id if specified
        '''

        from shapely.geometry import Point
        import pycrs

        # Sort input depending on elements or nodes
        if elements:
            field_name = 'Ele_num'
            point_data = self.elements
            point_id = self.element_ids
        else:
            field_name = 'Node_num'
            point_data = self.nodes
            point_id = self.node_ids

        # Create geometry series from points
        mesh_points = [Point(pt[0], pt[1]) for pt in point_data]
        mesh_series = gpd.GeoSeries(mesh_points)

        # Create GeoDataframe
        mesh_df = gpd.GeoDataFrame(point_id, geometry=mesh_series,
                                   columns=[field_name])

        # Set crs
        proj4_crs = pycrs.parser.from_ogc_wkt(self.projection).to_proj4()
        mesh_df.crs = proj4_crs

        if output_shp is not None:
            mesh_df.to_file(output_shp)

        return mesh_df

    def lyr_from_shape(self, lyr_name, input_shp, field_attribute,
                       output_shp=None):
        """
        Create a model input layer at mesh element coordinates.

        For example, input_shp is a roughness map containing polygons with
        roughness values. A spatial join  is performed for mesh element points
        within input_shp polygons and returns field_attributeat element points.

        Parameters
        ----------
        lyr_name : str
            Layer name as key to `lyrs` attribute dictionary
        input_shp : str
            Path to input shape file
        field_attributes : str
            Attribute in `input_shp` to extract at mesh elements
        output_shp : str, optional
            output path to write to .shp file

        Returns
        -------
        Inserts `lyr_name` into the `lyrs` attribute dictionary as an ndarray,
        shape (num_elements,) with extracted `field_attributes` value for each
        mesh element
        """

        # Load input_shp to GeoDF
        input_df = gpd.read_file(input_shp)

        # Load mesh element points as GeoDF
        mesh_df = self.to_gpd()

        # Perform spatial join
        join_df = gpd.sjoin(mesh_df, input_df, how='left', op='within')
        self.lyrs[lyr_name] = np.array(join_df[field_attribute])

        if output_shp is not None:
            join_df.to_file(output_shp)

    def lyr_to_dfsu(self, lyr_name, output_dfsu):
        """
        Create model layer .dfsu file `lyr` attribute. References `lyrs`
        attribute dictionary as value at element coordinates to write to
        .dfsu file.

        See also `lyr_from_shape`.

        Parameters
        ----------
        lyr_name : str
            Layer name as key to `lyrs` attribute dictionary
        output_dfsu : str
            Path to output .dfsu file

        Returns
        -------
        weights : array, shape (n_components,)

        """
        # Check that lyr_name is correct
        assert self.lyrs[lyr_name].shape[0] == self.num_elements, \
            "Length of input layer must equal number of mesh elements"

        # Load mesh file and mesh object
        mesh_class = dfs.mesh.MeshFile()
        dhi_mesh = mesh_class.ReadMesh(self.filename)

        # Call dfsu builder
        builder = dfs.dfsu.DfsuBuilder.Create(dfs.dfsu.DfsuFileType.Dfsu2D)
        builder.SetFromMeshFile(dhi_mesh)

        # Create arbitrary date and timestep; this is not a dynamic dfsu
        date = dt.datetime(2018, 1, 1, 0, 0, 0, 0)
        time_step = 1.0
        builder.SetTimeInfo(System.DateTime(date.year, date.month, date.day),
                            time_step)

        # Create dfsu attribute
        builder.AddDynamicItem(lyr_name,
                               eumQuantity(eumItem.eumIManningsM,
                                           eumUnit.eumUMeter2One3rdPerSec))

        # Create file
        dfsu_file = builder.CreateFile(output_dfsu)

        # Add lyr_name values
        net_arr = Array.CreateInstance(System.Single, self.num_elements)
        for i, val in enumerate(self.lyrs[lyr_name]):
            net_arr[i] = val
        dfsu_file.WriteItemTimeStepNext(0, net_arr)

        # Close file
        dfsu_file.Close()

    def plot_mesh(self, fill=False, kwargs=None):
        """
        Plot triangular mesh with triplot or tricontourf.

        See matplotlib kwargs for respective additional plot arguments.

        Parameters
        ----------
        fill : boolean
            if True, plots filled contour mesh (tricontourf)
            if False, plots (x, y) triangular mesh (triplot)
        kwargs : dict
            Additional arguments supported by triplot/tricontourf

        Returns
        -------
        fig : matplotlib figure obj
        ax : matplotlib axis obj

        See Also
        --------
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.triplot.html
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tricontourf.html

        """
        if fill:
            fig, ax, tf = _filled_mesh_plot(self.nodes[:,0],
                                            self.nodes[:,1],
                                            self.nodes[:,2],
                                            self.element_table,
                                            kwargs)
            return fig, ax, tf

        else:
            fig, ax = _mesh_plot(self.nodes[:,0], self.nodes[:,1],
                                 self.element_table, kwargs)
            return fig, ax


def _dfsu_builder(mesh_path):
    mesh_class = dfs.mesh.MeshFile()
    dhi_mesh = mesh_class.ReadMesh(mesh_path)

    # Call dfsu builder
    builder = dfs.dfsu.DfsuBuilder.Create(dfs.dfsu.DfsuFileType.Dfsu2D)
    builder.SetFromMeshFile(dhi_mesh)


def _read_mesh(dfs_obj):
    """ See Mesh class description for output details """
    num_nodes = dfs_obj.NumberOfNodes
    num_elements = dfs_obj.NumberOfElements

    # Node coordinates
    nodes = _node_coordinates(dfs_obj)

    # Node ids
    node_ids = _utils.dotnet_arr_to_ndarr(dfs_obj.NodeIds)

    # Node boundary codes
    boundary_code = _utils.dotnet_arr_to_ndarr(dfs_obj.Code)

    # Element table
    ele_table = _element_table(dfs_obj)

    # Node table
    node_table = _node_table(num_nodes, num_elements, ele_table)

    # Element ids
    element_ids = _utils.dotnet_arr_to_ndarr(dfs_obj.ElementIds)

    # Element coordinates
    if dfs_obj.GetType().get_Name() == 'DfsuFile':
        # Use internal MIKE SDK method if dfsu file
        elements = _dfsu_element_coordinates(dfs_obj)

    elif dfs_obj.GetType().get_Name() == 'MeshFile':
        # Else derive coordinates from element table and nodes
        elements = _mesh_element_coordinates(ele_table, nodes)

    return nodes, node_ids, boundary_code, ele_table, node_table, elements, element_ids


def _node_coordinates(dfs_obj):
    """ Read in node (x,y,z) """
    xn = _utils.dotnet_arr_to_ndarr(dfs_obj.X)
    yn = _utils.dotnet_arr_to_ndarr(dfs_obj.Y)
    zn = _utils.dotnet_arr_to_ndarr(dfs_obj.Z)

    return np.column_stack([xn,yn,zn])


def _element_table(dfs_obj):
    """ Defines for each element the nodes that define the element """
    table_obj = dfs_obj.ElementTable
    ele_table = np.zeros((len(table_obj), 3), dtype=int)
    for i, e in enumerate(table_obj):
        ele_table[i, :] = _utils.dotnet_arr_to_ndarr(e)
    return ele_table


def _node_table(num_nodes, num_elements, ele_table):
    """ Create node_table from ele_table """

    # Set placeholders for constructing node-to-element-table (node_table)
    e = np.arange(num_elements)
    u = np.ones(num_elements)
    I = np.concatenate((e, e, e))
    J = np.concatenate((ele_table[:,0],ele_table[:,1],ele_table[:,2]))
    K = np.concatenate((u*1, u*2, u*3))

    # Construct node_table
    count = np.zeros((num_nodes,1))
    for i in range(len(I)):
        count[J[i-1]-1] = count[J[i-1]-1]+1
    num_cols = int(count.max())

    node_table = np.zeros((num_nodes,num_cols), dtype='int')
    count = np.zeros((num_nodes,1))
    for i in range(len(I)):
        count[J[i-1]-1] = count[J[i-1]-1]+1
        node_table[J[i-1]-1, int(count[J[i-1]-1])-1] = I[i]

    return node_table


def _mesh_element_coordinates(element_tables, nodes):
    """ Manual method to calc element coords from ele table and node coords"""
    # Node coords
    xn = nodes[:, 0]
    yn = nodes[:, 1]
    zn = nodes[:, 2]

    # Elmt node index mapping (minus 1 because python indexing)
    node_map = element_tables[:, 1:].astype('int') - 1

    # Take mean of nodes mapped to element
    xe = np.mean(xn[node_map], axis=1)
    ye = np.mean(yn[node_map], axis=1)
    ze = np.mean(zn[node_map], axis=1)

    return np.stack([xe, ye, ze], axis=1)


def _dfsu_element_coordinates(dfsu_object):
    """ Use MIKE SDK method to calc element coords from dfsu_object """

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


def _write_mesh(filename, nodes, node_id,
                node_boundary_code, element_table,
                element_ids, proj, zUnitKey=1000):
    """ See Mesh class description for input details """

    eum_type = 100079  # Specify item type as 'bathymetry' (MIKE convention)
    num_nodes = len(nodes)

    node_write_fmt = np.column_stack([node_id, nodes, node_boundary_code])
    ele_write_fmt = np.column_stack([element_ids, element_table])

    # Open file to write to
    with open(filename, 'w') as target:
        # Format first line
        first_line = '%i  %i  %i  %s\n' % (eum_type, zUnitKey, num_nodes, proj)
        target.write(first_line)

        # Nodes
        np.savetxt(target, node_write_fmt, fmt='%i %-17.15g %17.15g %17.15g %i',
                   newline='\n', delimiter=' ')

        # Element header
        num_elements = len(ele_write_fmt)

        # Specify only triangular elements
        elmt_header = '%i %i %i\n' % (num_elements, 3, 21)
        target.write(elmt_header)

        # Elements
        np.savetxt(target, ele_write_fmt, fmt='%i', newline='\n', delimiter=' ')


def _mesh_plot(x, y, element_table, kwargs=None):
    """ Triplot of the mesh """
    if kwargs is None:
        kwargs = {}

    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    # Subtract 1 from element table to align with Python indexing
    t = tri.Triangulation(x, y, element_table-1)

    fig, ax = plt.subplots()
    ax.triplot(t, **kwargs)

    return fig, ax


def _filled_mesh_plot(x, y, z, element_table, kwargs=None):
    """ Tricontourf of the mesh and input z"""
    if kwargs is None:
        kwargs = {}

    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    # Subtract 1 from element table to align with Python indexing
    t = tri.Triangulation(x, y, element_table-1)

    fig, ax = plt.subplots()
    tf = ax.tricontourf(t, z, **kwargs)

    return fig, ax, tf
