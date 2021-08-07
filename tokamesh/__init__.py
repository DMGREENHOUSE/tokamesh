
from numpy import arange, linspace, searchsorted, stack, zeros, full, log2, floor, unique, atleast_1d
from numpy import int64
from itertools import product
from tokamesh.intersection import edge_rectangle_intersection


class TriangularMesh(object):
    def __init__(self, R, z, triangles):
        """
        Class for calculating geometry matrices over triangular meshes using
        barycentric linear interpolation.

        :param R: \
            The major radius of each mesh vertex as a 1D numpy array.

        :param z: \
            The z-height of each mesh vertex as a 1D numpy array.

        :param triangles: \
            A 2D numpy array of integers specifying the indices of the vertices which form
            each of the triangles in the mesh. The array must have shape `(N,3)` where `N` is
            the total number of triangles.
        """
        self.R = R
        self.z = z
        self.triangle_vertices = triangles
        self.n_vertices = self.R.size
        self.n_triangles = self.triangle_vertices.shape[0]

        # pre-calculate barycentric coordinate coefficients for each triangle
        R1, R2, R3 = [self.R[self.triangle_vertices[:,k]] for k in range(3)]
        z1, z2, z3 = [self.z[self.triangle_vertices[:,k]] for k in range(3)]
        self.area = 0.5*((z2 - z3)*(R1 - R3) + (R3 - R2)*(z1 - z3))
        self.lam1_coeffs = 0.5*stack([z2-z3, R3-R2, R2*z3 - R3*z2], axis=1) / self.area[:,None]
        self.lam2_coeffs = 0.5*stack([z3-z1, R1-R3, R3*z1 - R1*z3], axis=1) / self.area[:,None]

        # number all the edges in the mesh, and store the indices of which edges
        # are part of each triangle
        self.triangle_edges = zeros([self.n_triangles, 3], dtype=int64)
        edge_dict = {}
        for i in range(self.n_triangles):
            s1 = (self.triangle_vertices[i,0], self.triangle_vertices[i,1])
            s2 = (self.triangle_vertices[i,1], self.triangle_vertices[i,2])
            s3 = (self.triangle_vertices[i,0], self.triangle_vertices[i,2])
            for j, edge in enumerate([s1, s2, s3]):
                if edge not in edge_dict:
                    edge_dict[edge] = len(edge_dict)
                self.triangle_edges[i,j] = edge_dict[edge]

        self.n_edges = len(edge_dict)
        self.R_edges = zeros([self.n_edges, 2])
        self.z_edges = zeros([self.n_edges, 2])
        for edge, i in edge_dict.items():
            self.R_edges[i,:] = [self.R[edge[0]], self.R[edge[1]]]
            self.z_edges[i,:] = [self.z[edge[0]], self.z[edge[1]]]

        # store info about the bounds of the mesh
        self.R_limits = [self.R.min(), self.R.max()]
        self.z_limits = [self.z.min(), self.z.max()]

        self.build_binary_trees()

    def build_binary_trees(self):
        # we now divide the bounding rectangle of the mesh into
        # into a rectangular grid, and create a mapping between
        # each grid cell and all triangles which intersect it.

        # find an appropriate depth for each tree
        R_extent = self.R[self.triangle_vertices].ptp(axis=1).mean()
        z_extent = self.z[self.triangle_vertices].ptp(axis=1).mean()
        R_depth = max(int(floor(log2((self.R_limits[1]-self.R_limits[0]) / R_extent))), 2)
        z_depth = max(int(floor(log2((self.z_limits[1]-self.z_limits[0]) / z_extent))), 2)
        # build binary trees for each axis
        self.R_tree = BinaryTree(R_depth, self.R_limits)
        self.z_tree = BinaryTree(z_depth, self.z_limits)

        # now build a map between rectangle centres and a list of
        # all triangles which intersect that rectangle
        self.tree_map = {}
        for i,j in product(range(self.R_tree.nodes), range(self.z_tree.nodes)):
            # limits of the rectangle
            R_lims = self.R_tree.edges[i:i+2]
            z_lims = self.z_tree.edges[j:j+2]
            # find all edges which intersect the rectangle
            edge_inds = edge_rectangle_intersection(R_lims, z_lims, self.R_edges, self.z_edges)
            edge_bools = zeros(self.n_edges, dtype=int64)
            edge_bools[edge_inds] = 1
            # use this to find which triangles intersect the rectangle
            triangle_bools = edge_bools[self.triangle_edges].any(axis=1)
            # add the indices of these triangles to the dict
            if triangle_bools.any():
                self.tree_map[(i,j)] = triangle_bools.nonzero()[0]

    def interpolate(self, R, z, vertex_values):
        """
        Given the values of a function at each vertex of the mesh, use barycentric
        interpolation to approximate the function at a chosen set of points. Any
        points which lie outside the mesh will be assigned a value of zero.

        :param R: \
            The major radius of each interpolation point as a 1D numpy array.

        :param z: \
            The z-height of each interpolation point as a 1D numpy array.

        :param vertex_values: \
            The function value at each mesh vertex as a 1D numpy array.

        :return: \
            The interpolated function values as a 1D numpy array.
        """
        R_vals = atleast_1d(R)
        z_vals = atleast_1d(z)
        # first determine in which cell each point lies using the binary trees
        grid_coords = zeros([R_vals.size,2], dtype=int64)
        grid_coords[:,0] = self.R_tree.lookup_index(R_vals)
        grid_coords[:,1] = self.z_tree.lookup_index(z_vals)
        # find the set of unique grid coordinates
        unique_coords, inverse, counts = unique(grid_coords, axis=0, return_inverse=True, return_counts=True)
        # now create an array of indices which are ordered according to which of the unqiue values they match
        indices = inverse.argsort()
        # build a list of slice objects which addresses those indices which match each unique coordinate
        ranges = counts.cumsum()
        slices = [slice(0, ranges[0])]
        slices.extend([slice(*ranges[i:i + 2]) for i in range(ranges.size - 1)])
        # loop over each unique grid coordinate
        interpolated_values = zeros(R_vals.size)
        for v, slc in zip(unique_coords, slices):
            # only need to proceed if the current coordinate contains triangles
            key = (v[0], v[1])
            if key in self.tree_map:
                search_triangles = self.tree_map[key]  # the triangles intersecting this cell
                cell_indices = indices[slc]  # the indices of points inside this cell
                # get the barycentric coord values of each point, and the index of the triangle which contains them
                coords, container_triangles = self.bary_coords(R_vals[cell_indices], z_vals[cell_indices], search_triangles)
                # get the values of the vertices for the triangles which contain the points
                vals = vertex_values[self.triangle_vertices[container_triangles,:]]
                # take the dot-product of the coordinates and the vertex values to get the interpolated value
                interpolated_values[cell_indices] = (coords*vals).sum(axis=1)
        return interpolated_values

    def bary_coords(self, R, z, search_triangles):
        Q = stack([atleast_1d(R), atleast_1d(z), full(R.size, fill_value=1.)], axis=0)
        lam1 = self.lam1_coeffs[search_triangles,:].dot(Q)
        lam2 = self.lam2_coeffs[search_triangles,:].dot(Q)
        lam3 = 1 - lam1 - lam2
        bools = (lam1 >= 0.) & (lam2 >= 0.) & (lam3 >= 0.)
        i1, i2 = bools.nonzero()

        coords = zeros([R.size,3])
        coords[i2,0] = lam1[i1,i2]
        coords[i2,1] = lam2[i1,i2]
        coords[i2,2] = lam3[i1,i2]
        container_triangles = full(R.size, fill_value=-1)
        container_triangles[i2] = search_triangles[i1]
        return coords, container_triangles

    def draw(self, ax, **kwargs):
        """
        Draw the mesh using a given matplotlib.pyplot axis object.

        :param ax: \
            A matplotlib.pyplot axis object on which the mesh will be drawn by
            calling the 'plot' method of the object.

        :param kwargs: \
            Any valid keyword argument of matplotlib.pyplot.plot may be given in
            order to change the properties of the plot.
        """
        ax.plot(self.R_edges.T, self.z_edges.T, **kwargs)




class BinaryTree:
    """
    divides the range specified by limits into n = 2**layers equal regions,
    and builds a binary tree which allows fast look-up of which of region
    contains a given value.

    :param int layers: number of layers that make up the tree
    :param limits: tuple of the lower and upper bounds of the look-up region.
    """
    def __init__(self, layers, limits):
        self.layers = layers
        self.nodes = 2**self.layers
        self.lims = limits
        self.edges = linspace(limits[0], limits[1], self.nodes + 1)
        self.mids = 0.5*(self.edges[1:] + self.edges[:-1])

        self.indices = full(self.nodes+2, fill_value=-1, dtype=int64)
        self.indices[1:-1] = arange(self.nodes)

    def lookup_index(self, values):
        return self.indices[searchsorted(self.edges, values)]