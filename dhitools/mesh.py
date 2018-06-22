"""DHI MIKE21 mesh functions
"""

# Author: Robert Wall

import numpy as np


class Mesh():

    def __init__(self, filename=None):
        self.filename = filename
        self._file_input = False
        self.zUnitKey = 1000  # Default value (1000 = meter)

        if filename is not None:
            self.read_mesh(self.filename)

    def read_mesh(self, filename=None):
        '''
        Read in .mesh file
        '''
        if filename is None:
            filename = self.filename

        elmts, nodes, proj, zunitkey = _read_mesh(filename)

        self.elements = elmts
        self.nodes = nodes
        self.projection = proj
        self.zUnitKey = zunitkey
        self._file_input = True
        self.num_elements = len(self.elements)
        self.num_nodes = len(self.nodes)

    def summary(self):
        '''
        Summarise .mesh file
        '''
        if self._file_input:
            print("Input mesh file: {}".format(self.filename))
        else:
            print("No input file")

        try:
            print("Num. Elmts = {}".format(self.num_elements))
            print("Num. Elmts = {}".format(self.num_nodes))
            print("Mean elevation = {}".format(np.mean(self.nodes[:, 3])))
            print("Projection = \n {}".format(self.projection))
        except AttributeError:
            print("Object has no element or node properties. Read in mesh.")

    def write_mesh(self, output_name):
        '''
        Write new mesh file
        '''
        try:
            _write_mesh(filename=output_name,
                        Elmts=self.elements, Nodes=self.nodes,
                        proj=self.projection, zUnitKey=self.zUnitKey)
            print("Successfully output mesh to: {}".format(output_name))
        except AttributeError:
            print("Error: Object has no element or node properties. Read in mesh.")

    def interpolate_rasters(self, raster_list, method='nearest'):
        '''
        Interpolate multiple raster elevations to mesh nodes
        '''
        import _raster_interpolate as _ri

        for r in raster_list:
            interp_z = _ri.interpolate_from_raster(r, self.nodes[:, 1:3], method)

            # Just consider nodes that overlay raster
            updated_z = np.column_stack((self.nodes[:, 0], interp_z))

            # Sort by node_id
            updated_z_sorted = updated_z[updated_z[:, 0].argsort()]

            # Boolean mask for only updated nodes
            updated_bool = ~np.isnan(updated_z_sorted[:, 1])

            # Drop NaN
            only_updated_z = updated_z_sorted[:, 1][~np.isnan(updated_z_sorted[:, 1])]

            # Update mesh obj nodes only where node was interpolated
            self.nodes[:, 3][updated_bool] = only_updated_z


def _read_mesh(filename):
    '''
    Read nodes and element connectivity from a .mesh file

    Outputs:
        elmt: Element-Node table, for each element list the node number,
                as returned by the delaunay function
        node: Node coordinates having 4 columns, [id, x, y, z, code]
        proj: Projection string of mesh
        zUnitKey: EUM Unit key for Z values in mesh. Common values:
                    1000 = metre
                    1014 = feet (US)
                  Check EUM system for details

    '''

    # Open mesh file
    with open(filename, 'r') as fid:
        # Read all lines
        lines = fid.readlines()

        # First line contains number of nodes, zUnitKey (ie. metres)
        # and projection
        first_line = lines[0]
        split_first_line = first_line.split()

        # Some files might not have a zunitkey and projection
        if len(split_first_line) is 4:
            nnodes = int(split_first_line[2])
            zunitkey = int(split_first_line[1])
            proj = split_first_line[3]
        elif len(split_first_line) is 2:
            nnodes = int(split_first_line[0])
            zunitkey = []
            proj = []
        else:
            print('Check mesh header file')
            return None

        # Sort nodes in to array
        nodes_str = lines[1:nnodes + 1]   # Get list of nodes as srt for each line
        nodes_tmp = [line.split() for line in nodes_str]  # Split str on each line
        nodes_float = [map(float, line) for line in nodes_tmp]  # Convert to floats
        nodes = np.array(nodes_float)

        # Sort elements in to array
        elmt_str = lines[nnodes + 2:]
        elmt_tmp = [line.split() for line in elmt_str]
        elmt_float = [map(float, line) for line in elmt_tmp]
        elmts = np.array(elmt_float)

        return elmts, nodes, proj, zunitkey


def _write_mesh(filename, Elmts, Nodes, proj, zUnitKey=1000):
    '''
    Write a MikeZero .mesh file

    Inputs:
        Elmts    : Element-Node table, for each element list the node
                   number, e.g., as returned by the delaunay function.
        Nodes    : Node coordinates having 4 columns, [x, y, z, code]
        filename : Name of file to write
        proj     : String containing coordinate system
        zUnitKey : EUM Unit key for Z values in mesh. Common values:
                     1000 = meter (default)
                     1014 = feet US
                   Check EUM system for details (EUM.xml). Must be a length
                   unit.

    NOTE: THIS HAS ONLY BEEN TESTED FOR TRIANGLE ELEMENTS

    '''

    eum_type = 100079  # Some number specified by DHI, so just include it
    nnodes = len(Nodes)

    # Open file to write to
    with open(filename, 'w') as target:
        # Format first line
        first_line = '%i  %i  %i  %s\n' % (eum_type, zUnitKey, nnodes, proj)
        target.write(first_line)

        # Nodes
        np.savetxt(target, Nodes, fmt='%i %-17.15g %17.15g %17.15g %i',
                   newline='\n', delimiter=' ')

        # Element header
        nelmts = len(Elmts)
        elmt_header = '%i %i %i\n' % (nelmts, 3, 21)
        target.write(elmt_header)

        # Elements
        np.savetxt(target, Elmts, fmt='%i', newline='\n', delimiter=' ')
