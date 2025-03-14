import matplotlib.pyplot as plt
import numpy as np
from h5py import Group
from enum import Enum

from scipy.interpolate import interp1d
try:
    from importlib.metadata import version, PackageNotFoundError
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tokamesh")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = ["__version__"]

from numpy import searchsorted, stack, log2, floor, unique, atleast_1d, atleast_2d
from numpy import arange, linspace, int64, full, zeros, meshgrid, ndarray, ones
from numpy import empty, nan, isnan, argwhere, argsort, cross, pi
from numpy import savez, load, array, sqrt, mean, argmin, ones_like, zeros_like, tile
from numpy.linalg import solve
from scipy.sparse import csc_matrix
from scipy.special import factorial
from itertools import product
from scipy.integrate import simpson as simps
from tokamesh.intersection import edge_rectangle_intersection
from tokamesh.geometry import build_edge_map
from tokamesh.tokamaks import mastu_boundary


class TriangularMesh(object):
    """
    Class for performing operations with a triangular mesh, such as
    interpolation and plotting.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where
        ``N`` is the total number of triangles.
    """

    def __init__(self, R, z, triangles):

        for name, obj in [("R", R), ("z", z), ("triangles", triangles)]:
            if not isinstance(obj, ndarray):
                raise TypeError(
                    f"""\n
                    [ TriangularMesh error ]
                    >> The '{name}' argument of TriangularMesh should have type:
                    >> {ndarray}
                    >> but instead has type:
                    >> {type(obj)}
                    """
                )

        for name, obj in [("R", R), ("z", z)]:
            if obj.squeeze().ndim > 1:
                raise ValueError(
                    f"""\n
                    [ TriangularMesh error ]
                    >> The '{name}' argument of TriangularMesh should be
                    >> a 1D array, but given array has shape {obj.shape}.
                    """
                )

        if R.size != z.size:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The 'R' and 'z' arguments of TriangularMesh should be
                >> of equal size, but given arrays have sizes {R.size} and {z.size}.
                """
            )

        if triangles.squeeze().ndim != 2 or triangles.squeeze().shape[1] != 3:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The 'triangles' argument must have shape (num_triangles, 3)
                >> but given array has shape {triangles.shape}.
                """
            )

        self.R = R.squeeze()
        self.z = z.squeeze()
        self.triangle_vertices = triangles.squeeze()
        self.n_vertices = self.R.size
        self.n_triangles = self.triangle_vertices.shape[0]

        # pre-calculate barycentric coordinate coefficients for each triangle
        R1, R2, R3 = [self.R[self.triangle_vertices[:, k]] for k in range(3)]
        z1, z2, z3 = [self.z[self.triangle_vertices[:, k]] for k in range(3)]
        self.area = 0.5 * ((z2 - z3) * (R1 - R3) + (R3 - R2) * (z1 - z3))
        self.lam1_coeffs = (
            0.5
            * stack([z2 - z3, R3 - R2, R2 * z3 - R3 * z2], axis=1)
            / self.area[:, None]
        )
        self.lam2_coeffs = (
            0.5
            * stack([z3 - z1, R1 - R3, R3 * z1 - R1 * z3], axis=1)
            / self.area[:, None]
        )

        # Construct a mapping from triangles to edges, and edges to vertices
        self.triangle_edges, self.edge_vertices, _ = build_edge_map(
            self.triangle_vertices
        )
        self.R_edges = self.R[self.edge_vertices]
        self.z_edges = self.z[self.edge_vertices]
        self.n_edges = self.edge_vertices.shape[0]

        delta = 1.e-3
        # store info about the bounds of the mesh
        self.R_limits = [self.R.min()-delta, self.R.max()+delta]
        self.z_limits = [self.z.min()-delta, self.z.max()+delta]

        self.build_binary_trees()

    def build_binary_trees(self):
        # we now divide the bounding rectangle of the mesh into
        # a rectangular grid, and create a mapping between each
        # grid cell and all triangles which intersect it.

        # find an appropriate depth for each tree
        R_extent = self.R[self.triangle_vertices].ptp(axis=1).mean()
        z_extent = self.z[self.triangle_vertices].ptp(axis=1).mean()
        R_depth = max(
            int(floor(log2((self.R_limits[1] - self.R_limits[0]) / R_extent)))-1, 2
        )
        z_depth = max(
            int(floor(log2((self.z_limits[1] - self.z_limits[0]) / z_extent)))-1, 2
        )
        # build binary trees for each axis
        self.R_tree = BinaryTree(R_depth, self.R_limits)
        self.z_tree = BinaryTree(z_depth, self.z_limits)

        # now build a map between rectangle centres and a list of
        # all triangles which intersect that rectangle
        self.tree_map = {}
        for i, j in product(range(self.R_tree.nodes), range(self.z_tree.nodes)):
            # limits of the rectangle
            R_lims = self.R_tree.edges[i : i + 2]
            z_lims = self.z_tree.edges[j : j + 2]
            # find all edges which intersect the rectangle
            edge_inds = edge_rectangle_intersection(
                R_lims, z_lims, self.R_edges, self.z_edges
            )
            edge_bools = zeros(self.n_edges, dtype=int64)
            edge_bools[edge_inds] = 1
            # use this to find which triangles intersect the rectangle
            triangle_bools = edge_bools[self.triangle_edges].any(axis=1)
            # add the indices of these triangles to the dict
            if triangle_bools.any():
                self.tree_map[(i, j)] = triangle_bools.nonzero()[0]



    def interpolate(self, R, z, vertex_values, extrapolate_val=None, is_verify_vertex_match=False):
        """
        Given the values of a function at each vertex of the mesh, use barycentric
        interpolation to approximate the function at a chosen set of points. Any
        points which lie outside the mesh will be assigned a value of zero.

        :param R: \
            The major-radius of each interpolation point as a numpy array.

        :param z: \
            The z-height of each interpolation point as a numpy array.

        :param vertex_values: \
            The function value at each mesh vertex as a 1D numpy array.

        :return: \
            The interpolated function values as a numpy array.
        """
        vertex_values = self.handle_vertex_values(vertex_values)
        if type(vertex_values) is not ndarray or vertex_values.ndim != 1:
            raise TypeError(
                """\n
                [ TriangularMesh error ]
                >> The 'vertex_values' argument of the TriangularMesh.interpolate
                >> method must have type numpy.ndarray, and have only one dimension.
                """
            )

        if vertex_values.size != self.n_vertices:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The size of 'vertex_values' argument of the TriangularMesh.interpolate
                >> must be equal to the number of mesh vertices.
                >> The mesh has {self.n_vertices} vertices but given array is of size {vertex_values.size}.
                """
            )

        R_vals = atleast_1d(R)
        z_vals = atleast_1d(z)

        if R_vals.shape != z_vals.shape:
            raise ValueError(
                """\n
                [ TriangularMesh error ]
                >> The 'R' and 'z' arrays passed to the TriangularMesh.interpolate
                >> method are of inconsistent shapes - their shapes must be equal.
                """
            )

        input_shape = R_vals.shape
        if len(input_shape) > 1:
            R_vals = R_vals.flatten()
            z_vals = z_vals.flatten()

        # lookup sets of coordinates are in each grid cell
        unique_coords, slices, indices = self.grid_lookup(R_vals, z_vals)
        # loop over each unique grid coordinate
        interpolated_values = ones(R_vals.size)
        if extrapolate_val is None:
            extrapolate_val = 0.
        interpolated_values *= extrapolate_val
        for v, slc in zip(unique_coords, slices):
            # only need to proceed if the current coordinate contains triangles
            key = (v[0], v[1])
            if key in self.tree_map:
                # get triangles intersecting this cell
                search_triangles = self.tree_map[key]
                cell_indices = indices[slc]  # the indices of points inside this cell
                # get the barycentric coord values of each point, and the
                # index of the triangle which contains them
                coords, container_triangles = self.bary_coords(
                    R_vals[cell_indices], z_vals[cell_indices], search_triangles
                )
                # get the values of the vertices for the triangles which contain the points
                vals = vertex_values[self.triangle_vertices[container_triangles, :]]
                # take the dot-product of the coordinates and the vertex
                # values to get the interpolated value
                interpolated_values[cell_indices] = (coords * vals).sum(axis=1)
        if is_verify_vertex_match:
            # due to numerical issues assosciated with interpolating at a mesh vertex, check for existing
            precision = 1.e-3
            R_tile = tile(R, (self.R.size, 1)).T
            z_tile = tile(z, (self.z.size, 1)).T
            matching_inds = argwhere((abs(R_tile-self.R) < precision) & (abs(z_tile-self.z) < precision))
            interpolated_values[matching_inds[:, 0]] = vertex_values[matching_inds[:, 1]]

        if len(input_shape) > 1:
            interpolated_values.resize(input_shape)
        return interpolated_values

    def find_triangle(self, R, z):
        """
        Find the indices of the triangles which contain a given set of points.

        :param R: \
            The major-radius of each point as a numpy array.

        :param z: \
            The z-height of each point as a numpy array.

        :return: \
            The indices of the triangles which contain each point as numpy array.
            Any points which are not inside a triangle are given an index of -1.
        """
        R_vals = atleast_1d(R)
        z_vals = atleast_1d(z)

        if R_vals.shape != z_vals.shape:
            raise ValueError(
                """\n
                [ TriangularMesh error ]
                >> The 'R' and 'z' arrays passed to the TriangularMesh.interpolate
                >> method are of inconsistent shapes - their shapes must be equal.
                """
            )

        input_shape = R_vals.shape
        if len(input_shape) > 1:
            R_vals = R_vals.flatten()
            z_vals = z_vals.flatten()

        # lookup sets of coordinates are in each grid cell
        unique_coords, slices, indices = self.grid_lookup(R_vals, z_vals)
        # loop over each unique grid coordinate
        triangle_indices = full(R_vals.size, fill_value=-1, dtype=int)
        for v, slc in zip(unique_coords, slices):
            # only need to proceed if the current coordinate contains triangles
            key = (v[0], v[1])
            if key in self.tree_map:
                # get triangles intersecting this cell
                search_triangles = self.tree_map[key]
                cell_indices = indices[slc]  # the indices of points inside this cell
                # get the barycentric coord values of each point, and the
                # index of the triangle which contains them
                _, container_triangles = self.bary_coords(
                    R_vals[cell_indices], z_vals[cell_indices], search_triangles
                )
                triangle_indices[cell_indices] = container_triangles
        if len(input_shape) > 1:
            triangle_indices.resize(input_shape)
        return triangle_indices

    def grid_lookup(self, R, z):
        # first determine in which cell each point lies using the binary trees
        grid_coords = zeros([R.size, 2], dtype=int64)
        grid_coords[:, 0] = self.R_tree.lookup_index(R)
        grid_coords[:, 1] = self.z_tree.lookup_index(z)
        # find the set of unique grid coordinates
        unique_coords, inverse, counts = unique(
            grid_coords, axis=0, return_inverse=True, return_counts=True
        )
        # now create an array of indices which are ordered according
        # to which of the unique values they match
        indices = inverse.argsort()
        # build a list of slice objects which addresses those indices
        # which match each unique coordinate
        ranges = counts.cumsum()
        slices = [slice(0, ranges[0])]
        slices.extend([slice(*ranges[i : i + 2]) for i in range(ranges.size - 1)])
        return unique_coords, slices, indices

    def bary_coords(self, R, z, search_triangles):
        Q = stack([atleast_1d(R), atleast_1d(z), full(R.size, fill_value=1.0)], axis=0)
        lam1 = self.lam1_coeffs[search_triangles, :].dot(Q)
        lam2 = self.lam2_coeffs[search_triangles, :].dot(Q)
        lam3 = 1 - lam1 - lam2
        bools = (lam1 >= 0.0) & (lam2 >= 0.0) & (lam3 >= 0.0)
        i1, i2 = bools.nonzero()

        coords = zeros([R.size, 3])
        coords[i2, 0] = lam1[i1, i2]
        coords[i2, 1] = lam2[i1, i2]
        coords[i2, 2] = lam3[i1, i2]
        container_triangles = full(R.size, fill_value=-1)
        container_triangles[i2] = search_triangles[i1]
        return coords, container_triangles

    def draw(self, ax, is_add_numbered_vertices=False,
             x_label="Major Radius, R (m)",
             y_label="Height, z (m)",
             aspect='equal',
             **kwargs):
        """
        Draw the mesh using a given ``matplotlib.pyplot`` axis object.

        :param ax: \
            A ``matplotlib.pyplot`` axis object on which the mesh will be drawn by
            calling the 'plot' method of the object.

        :param kwargs: \
            Any valid keyword argument of ``matplotlib.pyplot.plot`` may be given in
            order to change the properties of the plot.
        """
        if ("color" not in kwargs) and ("c" not in kwargs):
            kwargs["color"] = "black"
        ax.plot(self.R_edges[0, :].T, self.z_edges[0, :].T, **kwargs)
        if "label" in kwargs:
            kwargs["label"] = None
        ax.plot(self.R_edges[1:, :].T, self.z_edges[1:, :].T, **kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect(aspect)
        if is_add_numbered_vertices:
            # Plot text at vertex
            for i, (x, y) in enumerate(zip(self.R, self.z)):
                ax.text(x=x, y=y, s=i, ha='center', va='bottom')

    def get_field_image(self, vertex_values, shape=(250, 250),
                        pad_fraction=0.01,
                        ax=None,
                        is_draw_limit=False,
                        is_draw_separatrix=False,
                        connection_length_args = None,
                        te_front_args = None,
                        is_shade_out_of_mwi=False,
                        R_extent=None,
                        z_extent=None,
                        resolution=None,
                        **kwargs):
        """
        Given the value of a field at each mesh vertex, use interpolation to generate
        an image of the field across the whole mesh.

        :param vertex_values: \
            The value of the field being plotted at each vertex of the mesh as a 1D numpy array.

        :param shape: \
            A tuple of two integers specifying the dimensions of the image.

        :param pad_fraction: \
            The fraction of the mesh width/height used as padding to create a gap between
            the edge of the mesh and the edge of the plot.

        :return R_axis, z_axis, field_image: \
            ``R_axis`` is a 1D array of the major-radius value of each column of the image array.
            ``z_axis`` is a 1D array of the z-height value of each column of the image array.
            ``field_image`` is a 2D array of the interpolated field values. Any points outside
            the mesh are assigned a value of zero.
        """
        vertex_values = self.handle_vertex_values(vertex_values)
        if R_extent is None:
            R_pad = (self.R_limits[1] - self.R_limits[0]) * pad_fraction
            r_min = self.R_limits[0] - R_pad
            r_max = self.R_limits[1] + R_pad
        else:
            r_min, r_max = R_extent
        if z_extent is None:
            z_pad = (self.z_limits[1] - self.z_limits[0]) * pad_fraction
            z_min = self.z_limits[0] - z_pad
            z_max = self.z_limits[1] + z_pad
        else:
            z_min, z_max = z_extent
        if resolution is not None:
            shape = (int((r_max-r_min)/resolution),
                     int((z_max-z_min)/resolution),)

        R_axis = linspace(r_min, r_max, shape[0])
        z_axis = linspace(z_min, z_max, shape[1])

        R_grid, z_grid = meshgrid(R_axis, z_axis)

        image = self.interpolate(
            R_grid.flatten(), z_grid.flatten(), vertex_values=vertex_values
        )
        image.resize((shape[1], shape[0]))
        if ax is None:
            return R_axis, z_axis, image.T
        else:

            if is_shade_out_of_mwi:
                # old coverage
                a = array((0.8, -1.77))
                b = array((1.58, -1.57))
                # new coverage
                a = array((0.6, -1.6))
                b = array((1.73, -1.55))
                points = array([R_grid, z_grid])
                # todo look to speed this up
                bools = np.zeros_like(image)
                for i in range(shape[0]):
                    for j in range(shape[0]):
                        bools[i][j] = cross(points[:, i, j] - a, b - a) >= 0
                alpha = (bools[::-1, :] +1.) / 2.
            else:
                alpha = 1.
            if 'alpha' not in kwargs.keys():
                kwargs['alpha'] = alpha
            im = ax.imshow(image[::-1, :],
                           extent=(R_axis[0], R_axis[-1], z_axis[0], z_axis[-1]),
                           # alpha=alpha,
                           **kwargs,
                           )
            if is_draw_limit:
                boundary_R, boundary_z = mastu_boundary(lower_divertor=True)
                ax.plot(boundary_R[1:], boundary_z[1:], lw=2, color='gray',
                        label="MAST-U Divertor Boundary"
                        )
            if is_draw_separatrix and hasattr(self, 'psi'):
                # sep_inds = argwhere(self.psi == 1.)
                ax.plot(self.R[self.separatrix_inds], self.z[self.separatrix_inds], lw=2, alpha=0.5,
                        color='gray', linestyle='dashed', label='Separatrix')
            if connection_length_args is not None and hasattr(self, 'index_grid'):
                default = {'value': 1.,
                           'field': self.parallel_distance,
                           'plot_params': {'color': 'gray',
                                           'linestyle': 'dotted',
                                           'lw': 2,
                                           'label': '1 (m) Connection Length'},
                           }
                for k, v in default.items():
                    if k not in connection_length_args.keys():
                        connection_length_args[k] = v
                self.plot_front(front_args=connection_length_args, ax=ax)
            if te_front_args is not None and hasattr(self, 'index_grid'):
                default = {'value': 5.,
                           'plot_params': {'color': 'magenta',
                                           'linestyle': 'dotted',
                                           'lw': 2,
                                           'label': '4 (eV) Front'},
                           }
                for k, v in default.items():
                    if k not in te_front_args.keys():
                        te_front_args[k] = v
                self.plot_front(front_args=te_front_args, ax=ax)

            ax.set_xlim((R_axis[0], R_axis[-1]))
            ax.set_ylim((z_axis[0], z_axis[-1]))
            ax.set_xlabel("R (m)")
            ax.set_ylabel("z (m)")
            return im

    def plot_front(self, front_args, ax):
        front = front_args['value']  # '1.
        field = front_args['field']  # self.parallel_distance
        inds = argmin(abs(field[self.index_grid] - front), axis=0)
        inds = array([ig[cl] for ig, cl in zip(self.index_grid.T, inds)])
        in_ranges = np.any(field[self.index_grid] > front, axis=0)
        inds = inds[in_ranges]
        ax.plot(self.R[inds], self.z[inds], **front_args['plot_params'])
                # lwcolor='gray', linestyle='dotted', label=f'{connection_length} (m) Connection Length')

    def build_interpolator_matrix(self, points, is_sparse=False):
        """
        Takes an array of (R, z) points and generates an interpolator matrix
        For each point, finds the triangle that encases the point and returns the Barycentric
             coordinates.
        The final matrix is 2d with each row referring to a requested point and each column
             referring to the mesh's vertex index. The element (value) is the Barycentric
             coordinate for that point according to that mesh vertex.
        """
        points = atleast_2d(points)
        if len(points.shape) > 2:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The expected format of the points variable is an array of r-z pairs.
                >> A two-dimensional array was expected but a the array provided has
                >> shape {points.shape}. 
                """
            )

        if points.shape[1] != 2:
            raise ValueError(
                f"""\n
                [ TriangularMesh error ]
                >> The expected format of the points variable is an array of r-z pairs.
                >> The second dimension was expected to have size two but the array
                >> provided has shape {points.shape}. 
                """
            )

        G = zeros([len(points), self.n_vertices])
        for q, p in enumerate(points):
            unique_coords, slices, _ = self.grid_lookup(p[0], p[1])
            for v, slc in zip(unique_coords, slices):
                # only need to proceed if the current coordinate contains triangles
                key = (v[0], v[1])
                if key in self.tree_map:
                    # get triangles intersecting this cell
                    search_triangles = self.tree_map[key]
                    # get the barycentric coord values of each point, and the
                    # index of the triangle which contains them
                    coords, container_triangles = self.bary_coords(
                        p[0], p[1], search_triangles
                    )
                    inds = self.triangle_vertices[container_triangles, :]
                    inds = inds.flatten()
                    coords = coords.flatten()
                    for i,v in zip(inds, coords):
                        G[q, i] = v
        # if any(G.sum(axis=1) == 0.):
        #     msg = "Geometry matrix may not have been correctly calculated. Values are unreachable"
        if is_sparse:
            G = csc_matrix(G)
        return G

    def umbrella_matrix(self, internal_only = False, inverse_distance_weighting = True, return_distances = False):
        """
        returns a sparse 'umbrella' matrix operator, which finds the difference between
        the value of every internal vertex value and the average value of the other
        vertices with which it is connected.
        """
        # first create a map of the connections between vertices
        connection_map = dict()
        for t in self.triangle_vertices:
            for side_vertex_index in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                ind1, ind2 = side_vertex_index
                if ind1 not in connection_map:
                    connection_map[ind1] = [ind2]
                elif ind2 not in connection_map[ind1]:
                    connection_map[ind1].append(ind2)

                if ind2 not in connection_map:
                    connection_map[ind2] = [ind1]
                elif ind1 not in connection_map[ind2]:
                    connection_map[ind2].append(ind1)

        # Sort the dictionary into ascending order
        # todo: a dictionary is actually unordered - might be best to replace this
        #  dictionary structure. It really depends on whether we truly require order or
        #  whether the recording of i and j indices suffices
        connection_map = dict(sorted(connection_map.items()))

        # find the indices of edge vertices so they can be ignored
        from tokamesh.construction import find_boundaries
        edge_indices = find_boundaries(self.triangle_vertices)

        i_indices = []
        j_indices = []
        values = []
        avg_distances = []
        distance = lambda p0, p1: sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
        distance_connection_map = {}
        weight_connection_map = {}
        for i, j_list in connection_map.items():
            if internal_only is False or i not in edge_indices:
                # set weights for the surrounding vertices
                distances = array( [distance([self.R[i], self.z[i]],
                                             [self.R[j], self.z[j]]) for j in j_list] )
                avg_distances.append( mean(distances) )

                if inverse_distance_weighting:
                    inv_distances = 1./distances
                    weights = inv_distances / inv_distances.sum()
                else:
                    weights = [1./len(j_list)]*len(j_list)

                distance_connection_map[i] = list(distances)
                weight_connection_map[i] = list(weights)
                # add the diagonal value
                values.append(-1)
                i_indices.append(i)
                j_indices.append(i)

                for j, w in zip(j_list, weights):  # add the averaging values
                    values.append(w)
                    i_indices.append(i)
                    j_indices.append(j)

        shape = (self.n_vertices, self.n_vertices)
        U = csc_matrix((array(values), (array(i_indices), array(j_indices))), shape = shape)
        if return_distances:
            return U, array(avg_distances)
        else:
            return U

    def umbrella_matrix_mesh_tools(self, internal_only = False, inverse_distance_weighting = True, return_distances = False):
        """
        returns a sparse 'umbrella' matrix operator, which finds the difference between
        the value of every internal vertex value and the average value of the other
        vertices with which it is connected.
        """

        # first create a map of the connections between vertices
        connection_map = dict()
        for t in self.triangle_vertices:
            for side_vertex_index in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                ind1, ind2 = side_vertex_index
                if ind1 not in connection_map:
                    connection_map[ind1] = [ind2]
                elif ind2 not in connection_map[ind1]:
                    connection_map[ind1].append(ind2)

                if ind2 not in connection_map:
                    connection_map[ind2] = [ind1]
                elif ind1 not in connection_map[ind2]:
                    connection_map[ind2].append(ind1)

        # find the indices of edge vertices so they can be ignored
        from tokamesh.construction import find_boundaries
        edge_indices = find_boundaries(self.triangle_vertices)

        i_indices = []
        j_indices = []
        values = []
        avg_distances = []
        distance = lambda x,y : sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
        for i, j_list in connection_map.items():
            if internal_only is False or i not in edge_indices:
                # set weights for the surrounding vertices
                distances = array( [distance([self.R[i], self.z[i]], [self.R[j], self.z[j]]) for j in j_list] )
                avg_distances.append( mean(distances) )
                if inverse_distance_weighting:
                    inv_distances = 1./distances
                    weights = inv_distances / inv_distances.sum()
                else:
                    weights = [1./len(j_list)]*len(j_list)

                # add the diagonal value
                values.append(-1)
                i_indices.append(i)
                j_indices.append(i)

                for j, w in zip(j_list,weights):  # add the averaging values
                    values.append(w)
                    i_indices.append(i)
                    j_indices.append(j)

        shape = (self.n_vertices, self.n_vertices)
        U = csc_matrix((array(values), (array(i_indices), array(j_indices))), shape = shape)
        if return_distances:
            return U, array(avg_distances)
        else:
            return U

    def save(self, filepath):
        savez(filepath, R=self.R, z=self.z, triangles=self.triangle_vertices)

    @classmethod
    def load(cls, data: (str, Group, dict)):
        if isinstance(data, str):
            mesh = cls._load_from_file_path(data)
        elif isinstance(data, Group):
            mesh = cls._load_from_h5_group(data)
        elif isinstance(data, dict):
            mesh = cls._load_from_dictionary(data)
        else:
            msg = f"Unrecognised data type: {type(data)}"
            raise TypeError(msg)
        return mesh

    @classmethod
    def _load_from_file_path(cls, filepath: str):
        data = load(filepath, allow_pickle=True)
        return cls._load_from_dictionary(data)

    @classmethod
    def _load_from_dictionary(cls, data: dict):
        return cls(R=data['R'], z=data['z'], triangles=data['triangles'])

    @classmethod
    def _load_from_h5_group(cls, file_group):
        keys = ("R", "z", "triangles")
        data = {k: file_group[k][()] for k in keys}
        return cls(**data)

    def triangle_area(self, x, y):
        # Create matrices for x and y coordinates
        x1, x2, x3 = x
        y1, y2, y3 = y

        # Calculate the determinant for each triangle
        areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        return areas

    def integrate_total_area(self,
                             vertex_values, is_total=True):
        vertex_values = self.handle_vertex_values(vertex_values)
        # get the centre of each
        R = self.R[self.triangle_vertices].mean(axis=1)
        z = self.z[self.triangle_vertices].mean(axis=1)
        # note this could be done directly
        centre_vals = self.interpolate(R, z, vertex_values)
        area = self.triangle_area(self.R[self.triangle_vertices].T, self.z[self.triangle_vertices].T)

        if is_total:
            return (centre_vals * area).sum(axis=0)
        else:
            return centre_vals * area

    def integrate_total_volume(self,
                             vertex_values):
        vertex_values = self.handle_vertex_values(vertex_values)
        # get the centre of each
        R = self.R[self.triangle_vertices].mean(axis=1)
        return (2. * np.pi * R * self.integrate_total_area(vertex_values, is_total=False)).sum(axis=0)

    def handle_vertex_values(self, vertex_values):
        return vertex_values


def get_fd_coeffs(points, order=1):
    # check validity of inputs
    if type(points) is not ndarray: points = array(points)
    n = len(points)
    if n <= order: raise ValueError('The order of the derivative must be less than the number of points')
    # build the linear system
    b = zeros(n)
    b[order] = factorial(order)
    A = ones([n, n])
    for i in range(1, n):
        A[i,:] = points**i
    # return the solution
    return solve(A, b)


class XQuantityOptions(Enum):
    R = "R"
    PSI = "psi"
    S_MINUS_S0 = "s-s0"

class YQuantityOptions(Enum):
    Z = "z"
    POLOIDAL_DISTANCE_TO_TARGET = "poloidal distance (to target)"
    POLOIDAL_DISTANCE_FROM_X_POINT = "poloidal distance (from x point)"
    POLOIDAL_DISTANCE_FROM_DIVERTOR_ENTRANCE = "poloidal distance (from divertor entrance)"
    PARALLEL_DISTANCE_TO_TARGET = "parallel distance (to target)"
    PARALLEL_DISTANCE_FROM_X_POINT = "parallel distance (from x point)"
    PARALLEL_DISTANCE_FROM_DIVERTOR_ENTRANCE = "parallel distance (from divertor entrance)"


def coarsen_field_aligned_mesh(base_mesh, step=2):
    """
    Create a new FieldAlignedMesh with half-resolution (or 1/step) in each dimension.
    The first and last rows/columns of the index_grid are always kept to preserve boundaries.

    :param base_mesh: FieldAlignedMesh object (see your code).
    :param step: Subsampling factor for the interior points.
    :return: A new, coarser FieldAlignedMesh.
    """
    # --- 1) Extract relevant arrays from base_mesh ---
    R_old = base_mesh.R
    z_old = base_mesh.z
    psi_old = base_mesh.psi
    pd_old = base_mesh.poloidal_distance
    par_old = base_mesh.parallel_distance
    index_grid_old = base_mesh.index_grid  # shape (nZ, nR)
    triangles_old = base_mesh.triangle_vertices

    # Some boundary logic: we keep the first & last row/column,
    # and step through the interior.
    nZ, nR = index_grid_old.shape
    # Row indices to keep:
    row_inds = [0] + list(range(1, nZ - 1, step)) + [nZ - 1]
    # Column indices to keep:
    col_inds = [0] + list(range(1, nR - 1, step)) + [nR - 1]

    # --- 2) Build the new sub-grid (index_grid_coarse) ---
    index_grid_coarse = index_grid_old[np.ix_(row_inds, col_inds)]
    # shape: (len(row_inds), len(col_inds))

    # --- 3) Collect all unique old vertex IDs that appear in index_grid_coarse ---
    old_ids_kept = np.unique(index_grid_coarse.ravel())

    # --- 4) Build a mapping old_id -> new_id (0..M-1) ---
    old_to_new = {old_id: i for i, old_id in enumerate(old_ids_kept)}

    # --- 5) Create new arrays R_coarse, z_coarse, etc. by selecting only old_ids_kept ---
    # We'll reorder them in ascending order of the old ID (this is arbitrary but consistent).
    # To do that, we can sort old_ids_kept if needed, or rely on the dict.
    # Let's define an array of new length M:
    M = len(old_ids_kept)
    R_coarse = np.zeros(M, dtype=R_old.dtype)
    z_coarse = np.zeros(M, dtype=z_old.dtype)
    psi_coarse = np.zeros(M, dtype=psi_old.dtype)
    pd_coarse = np.zeros(M, dtype=pd_old.dtype)
    par_coarse = np.zeros(M, dtype=par_old.dtype)

    # Sort old_ids_kept so we fill arrays in ascending ID order
    old_ids_sorted = np.sort(old_ids_kept)
    # We'll store old_id -> "array index" as well
    for new_i, old_id in enumerate(old_ids_sorted):
        R_coarse[new_i] = R_old[old_id]
        z_coarse[new_i] = z_old[old_id]
        psi_coarse[new_i] = psi_old[old_id]
        pd_coarse[new_i] = pd_old[old_id]
        par_coarse[new_i] = par_old[old_id]
    if base_mesh.B is not None:
        B_coarse = np.zeros((2, M), dtype=par_old.dtype)
        B_coarse[0] = base_mesh.B[0, old_ids_sorted]
        B_coarse[1] = base_mesh.B[1, old_ids_sorted]
    else:
        B_coarse = None
    # But note: old_to_new might not map in sorted order.
    # It's okay as long as we are consistent. We'll fix that by building:
    sorted_old_to_new = {}
    for new_i, old_id in enumerate(old_ids_sorted):
        sorted_old_to_new[old_id] = new_i

    # Remap index_grid_coarse to these new IDs
    idx_coarse_remapped = np.vectorize(sorted_old_to_new.__getitem__)(index_grid_coarse)
    nZc, nRc = idx_coarse_remapped.shape

    # --- 3) Re-Triangulate the Coarse Grid ---
    # We assume a structured rectangular grid, with 2 triangles per cell:
    #   (i, j)   -- (i, j+1)
    #     |   \         |
    #   (i+1,j) -- (i+1,j+1)
    # so each cell => triangles:
    #  t1 = (v00, v01, v11)
    #  t2 = (v00, v11, v10)

    tri_list = []
    for i in range(nZc - 1):
        for j in range(nRc - 1):
            v00 = idx_coarse_remapped[i, j]
            v01 = idx_coarse_remapped[i, j + 1]
            v10 = idx_coarse_remapped[i + 1, j]
            v11 = idx_coarse_remapped[i + 1, j + 1]

            # Triangle 1
            tri_list.append([v00, v01, v11])
            # Triangle 2
            tri_list.append([v00, v11, v10])

    triangles_coarse = np.array(tri_list, dtype=int)
    # --- 7) Build a new FieldAlignedMesh with these coarser arrays ---
    # Also pass along B, B_x_point, etc. if needed, in the same manner. For brevity, we skip them here.
    # We do handle the poloidal_distance_upstream_to_x_point, etc. by reusing base_mesh's attribute.
    # Up to you how to handle. Let's do a simple version:

    # We'll define a dictionary of all needed arguments:
    new_kwargs = dict(
        R=R_coarse,
        z=z_coarse,
        triangles=triangles_coarse,
        index_grid=idx_coarse_remapped,
        psi=psi_coarse,
        poloidal_distance=pd_coarse,
        parallel_distance=par_coarse,
        poloidal_distance_upstream_to_x_point=base_mesh.poloidal_distance_upstream_to_x_point,
        parallel_distance_upstream_to_x_point=base_mesh.parallel_distance_upstream_to_x_point,
        divertor_entrance_z=base_mesh.divertor_entrance_z,
        B=B_coarse,
        B_x_point=base_mesh.B_x_point,
        B_mid_plane=base_mesh.B_mid_plane,
        coarse_mapping=old_ids_sorted,
    )
    # If you have more or fewer attributes in your base_mesh,
    # add them to new_kwargs as desired. Or use a loop to copy any relevant ones.

    # Finally, construct a new FieldAlignedMesh
    # from copy import copy
    new_mesh = FieldAlignedMesh(**new_kwargs)
    # Copy over any extra attributes from base_mesh if needed
    # e.g., new_mesh.some_attribute = copy(base_mesh.some_attribute)
    return new_mesh


class FieldAlignedMesh(TriangularMesh):

    def __init__(self, R, z, triangles, index_grid,
                 psi, poloidal_distance, parallel_distance,
                 poloidal_distance_upstream_to_x_point=None,
                 parallel_distance_upstream_to_x_point=None,
                 divertor_entrance_z = -1.5,
                 B = None,  # (poloidal, toroidal)
                 B_x_point = None,  # (poloidal, toroidal)
                 B_mid_plane = None,  # (poloidal, toroidal)
                 coarse_mapping=None,
                 **kwargs
    ):
        """
        Poloidal Distance to Target
        Parallel Distance to Target
        :param R:
        :param z:
        :param triangles:
        :param index_grid:
        :param psi:
        :param poloidal_distance:
        :param parallel_distance:
        :param seperatrix_flux_expansion:
        :param upstream_separatrix_parallel_distance_to_xp:
        :param upstream_separatrix_poloidal_distance_to_xp:
        :param connection_length_upstream_sep_to_x_point:
        :param distance_to_strike_point:
        :param corrected_psi:
        :param poloidal_distance_to_target:
        :param connection_length_to_target:
        :param kwargs:
        """
        self.coarse_mapping = coarse_mapping
        for key, value in kwargs.items():
            setattr(self, key, value)
        super(FieldAlignedMesh, self).__init__(R, z, triangles)
        self.verify_unique()
        assert parallel_distance[index_grid[0]].mean() > parallel_distance[index_grid[-1]].mean()

        self.index_grid = index_grid
        self.inverse_index_grid = self.find_inverse_index_grid(index_grid)

        self.psi = psi
        self.poloidal_distance = poloidal_distance
        self.parallel_distance = parallel_distance
        self.divertor_entrance_z = divertor_entrance_z
        self.B = B
        self.B_x_point = B_x_point
        self.B_mid_plane = B_mid_plane
        if poloidal_distance_upstream_to_x_point is None:
            poloidal_distance_upstream_to_x_point = 0.
        if parallel_distance_upstream_to_x_point is None:
            parallel_distance_upstream_to_x_point = 0.
        self.poloidal_distance_upstream_to_x_point = poloidal_distance_upstream_to_x_point
        self.parallel_distance_upstream_to_x_point = parallel_distance_upstream_to_x_point

        if self.B is None:
            self.poloidal_component = None
            self.poloidal_component = None
        else:
            self.poloidal_component = abs(self.B[0] / sqrt(self.B[0]**2 + self.B[1]**2))
        self.upstream_indices = self.index_grid[0]
        self.downstream_indices = self.index_grid[-1]
        self.private_boundary_indices = self.index_grid[:, 0]
        self.common_boundary_indices = self.index_grid[:, -1]
        self.partnered_upstream_downstream_indexes = self.get_upstream_downstream_pairs()


        self.separatrix_inds = self.get_separatrix_inds()

        self.poloidal_distance_from_x_point = self.find_distance_from_x_point(
            self.poloidal_distance, poloidal_distance_upstream_to_x_point
        )
        self.parallel_distance_from_x_point = self.find_distance_from_x_point(
            self.parallel_distance, parallel_distance_upstream_to_x_point
        )
        self.poloidal_distance_from_divertor_entrance = self.find_distance_from_divertor_entrance(
            self.poloidal_distance
        )
        self.parallel_distance_from_divertor_entrance = self.find_distance_from_divertor_entrance(
            self.parallel_distance
        )

        self.poloidal_mesh = None
        self.s_minus_s0_grid = self.find_s_minus_s0_on_grid()
        self.s_minus_s0 = self.flatten(self.s_minus_s0_grid)
        self.vertex_width = self.find_vertex_width()
        self.vertex_height = self.find_vertex_height()
        self.vertex_parallel_height = self.find_vertex_parallel_height()

        self.x_quantities = {
            XQuantityOptions.R: {
                'Name': 'Major Radius, R',
                'Unit': '(m)',
                'Label': 'Major Radius, R (m)',
                'Quantities': self.R
            },
            XQuantityOptions.PSI: {
                'Name': 'Normalised Poloidal Flux, $\psi$',
                'Unit': '',
                'Label': 'Normalised Poloidal Flux, $\psi$',
                'Quantities': self.psi
            },
            XQuantityOptions.S_MINUS_S0: {
                'Name': 'Distance to Separatrix, $s-s_0$',
                'Unit': '(m)',
                'Label': 'Distance to Separatrix, $s-s_0$ (m)',
                'Quantities': self.s_minus_s0
            },
        }
        self.y_quantities = {
            YQuantityOptions.Z: {
                'Name': 'Height, z',
                'Unit': '(m)',
                'Label': 'Height, z (m)',
                'Quantities': self.z
            },
            YQuantityOptions.POLOIDAL_DISTANCE_TO_TARGET: {
                'Name': 'Poloidal Distance to Target',
                'Unit': '(m)',
                'Label': 'Poloidal Distance to Target (m)',
                'Quantities': self.poloidal_distance
            },
            YQuantityOptions.POLOIDAL_DISTANCE_FROM_X_POINT: {
                'Name': 'Poloidal Distance from X-Point',
                'Unit': '(m)',
                'Label': 'Poloidal Distance from X-Point (m)',
                'Quantities': self.poloidal_distance_from_x_point
            },
            YQuantityOptions.POLOIDAL_DISTANCE_FROM_DIVERTOR_ENTRANCE: {
                'Name': 'Poloidal Distance from Divertor Entrance',
                'Unit': '(m)',
                'Label': 'Poloidal Distance from Divertor Entrance (m)',
                'Quantities': self.poloidal_distance_from_divertor_entrance
            },
            YQuantityOptions.PARALLEL_DISTANCE_TO_TARGET: {
                'Name': 'Parallel Distance to Target',
                'Unit': '(m)',
                'Label': 'Parallel Distance to Target (m)',
                'Quantities': self.parallel_distance
            },
            YQuantityOptions.PARALLEL_DISTANCE_FROM_X_POINT: {
                'Name': 'Parallel Distance from X-Point',
                'Unit': '(m)',
                'Label': 'Parallel Distance from X-Point (m)',
                'Quantities': self.parallel_distance_from_x_point
            },
            YQuantityOptions.PARALLEL_DISTANCE_FROM_DIVERTOR_ENTRANCE: {
                'Name': 'Parallel Distance from Divertor Entrance',
                'Unit': '(m)',
                'Label': 'Parallel Distance from Divertor Entrance (m)',
                'Quantities': self.parallel_distance_from_divertor_entrance
            },
        }

    def handle_vertex_values(self, vertex_values):
        if (self.coarse_mapping is None) or (len(vertex_values) == len(self.coarse_mapping)):
            return vertex_values
        elif len(vertex_values) >= self.coarse_mapping.max():
            return vertex_values[self.coarse_mapping]
        else:
            msg = f"Unrecognised vertex_values shape: {vertex_values.shape}. Expected ({self.n_vertices}, ) or ({len(self.coarse_mapping)}, )."
            raise ValueError(msg)

    def interpolate_psi(self, R, z, **kwargs):
        return self.interpolate(R, z, self.psi, **kwargs)


    def get_separatrix_poloidal_flux_expansion_relative_x_point(self):
        if (self.B_x_point is None) or (self.B is None):
            msg = "Please provide magnetic field details"
            raise ValueError(msg)
        return (self.B_x_point[0] * self.B[1][self.separatrix_inds]) / (self.B[0][self.separatrix_inds] * self.B_x_point[1])

    def get_separatrix_poloidal_flux_expansion_relative_mid_plane(self):
        if (self.B_mid_plane is None) or (self.B is None):
            msg = "Please provide magnetic field details"
            raise ValueError(msg)
        return (self.B_mid_plane[0] * self.B[1][self.separatrix_inds]) / (self.B[0][self.separatrix_inds] * self.B_mid_plane[1])

    def get_separatrix_total_flux_expansion_relative_x_point(self):
        if (self.B_x_point is None) or (self.B is None):
            msg = "Please provide magnetic field details"
            raise ValueError(msg)
        return sqrt(self.B_x_point[0]**2 +  self.B_x_point[1]**2) / \
               sqrt(self.B[0][self.separatrix_inds]**2 + self.B[1][self.separatrix_inds]**2)

    def get_separatrix_total_flux_expansion_relative_mid_plane(self):
        if (self.B_mid_plane is None) or (self.B is None):
            msg = "Please provide magnetic field details"
            raise ValueError(msg)
        return sqrt(self.B_mid_plane[0]**2 +  self.B_mid_plane[1]**2) / \
               sqrt(self.B[0][self.separatrix_inds]**2 + self.B[1][self.separatrix_inds]**2)

    def verify_unique(self, limit = 1.e-7):

        R_diff = self.R[:, np.newaxis] - self.R[np.newaxis, :]
        z_diff = self.z[:, np.newaxis] - self.z[np.newaxis, :]
        dist_matrix = np.sqrt(R_diff**2 + z_diff**2)
        # ignore the lower diagonal
        dist_matrix[np.tril_indices_from(dist_matrix, k=0)] = 2. * limit
        below = np.argwhere(dist_matrix < limit)
        if dist_matrix.min() < limit:
            for b in below:
                print(f"There is a repeated point at ({self.R[b[0]]}, {self.z[b[0]]})")
            raise AssertionError

    def find_distance_from_x_point(self, distances, distance_to_x_point):
        distances_from_x_point = distance_to_x_point + distances[self.upstream_indices] - distances[self.index_grid]
        return self.flatten(distances_from_x_point)

    def find_distance_from_divertor_entrance(self, distances):
        # find the separatrix row closest
        sep_z = self.z[self.separatrix_inds]
        entrance_row = np.argmin(abs(sep_z - self.divertor_entrance_z))
        # base everything from this row (it is zero here)
        pd_grid = distances[self.index_grid[entrance_row]] - distances[self.index_grid]
        return self.flatten(pd_grid)

    def find_vertex_width(self):
        """
        Use the index grid to find the width associated with each mesh vertex
        :return:
        """

        R_grid = self.R[self.index_grid]
        z_grid = self.z[self.index_grid]
        # get the mid_points between all vertices
        R_mid =  0.5 * (R_grid[:, 1:] + R_grid[:, :-1])
        z_mid =  0.5 * (z_grid[:, 1:] + z_grid[:, :-1])

        # the width assosciated to each cell is the gap between midpoints
        R_width = np.zeros_like(R_grid)
        z_width = np.zeros_like(R_grid)
        R_width[:, 1:-1] = R_mid[:, 1:] - R_mid[:, :-1]
        z_width[:, 1:-1] = z_mid[:, 1:] - z_mid[:, :-1]

        # the first and final column are the gap from the edge to the midpoint
        R_width[:, 0] = R_mid[:, 0] - R_grid[:, 0]
        z_width[:, 0] = z_mid[:, 0] - z_grid[:, 0]
        R_width[:, -1] = R_grid[:, -1] - R_mid[:, -1]
        z_width[:, -1] = z_grid[:, -1] - z_mid[:, -1]

        width = np.sqrt(R_width**2 + z_width**2)

        return self.flatten(width)

    def find_vertex_height(self):
        """
        Use the index grid to find the height associated with each mesh vertex
        :return:
        """

        R_grid = self.R[self.index_grid]
        z_grid = self.z[self.index_grid]
        # get the mid_points between all vertices
        R_mid =  0.5 * (R_grid[1:, :] + R_grid[:-1, :])
        z_mid =  0.5 * (z_grid[1:, :] + z_grid[:-1, :])

        # the height assosciated to each cell is the gap between midpoints
        R_height = np.zeros_like(R_grid)
        z_height = np.zeros_like(R_grid)
        R_height[1:-1, :] = R_mid[1:, :] - R_mid[:-1, :]
        z_height[1:-1, :] = z_mid[1:, :] - z_mid[:-1, :]

        # the first and final column are the gap from the edge to the midpoint
        R_height[0, :] = R_mid[0, :] - R_grid[0, :]
        z_height[0, :] = z_mid[0, :] - z_grid[0, :]
        R_height[-1, :] = R_grid[-1, :] - R_mid[-1, :]
        z_height[-1, :] = z_grid[-1, :] - z_mid[-1, :]

        height = np.sqrt(R_height**2 + z_height**2)

        return self.flatten(height)

    def find_vertex_parallel_height(self):
        """
        Use the index grid to find the height associated with each mesh vertex
        :return:
        """

        parallel_grid = self.parallel_distance[self.index_grid]
        # get the mid_points between all vertices
        parallel_mid =  0.5 * (parallel_grid[1:, :] + parallel_grid[:-1, :])

        # the height assosciated to each cell is the gap between midpoints
        parallel_height = np.zeros_like(parallel_grid)
        parallel_height[1:-1, :] = parallel_mid[1:, :] - parallel_mid[:-1, :]

        # the first and final column are the gap from the edge to the midpoint
        parallel_height[0, :] = parallel_mid[0, :] - parallel_grid[0, :]
        parallel_height[-1, :] = parallel_grid[-1, :] - parallel_mid[-1, :]

        return self.flatten(parallel_height)

    def find_s_minus_s0_on_grid(self):
        R = self.R[self.index_grid]
        z = self.z[self.index_grid]
        R_sep = self.R[self.separatrix_inds]
        z_sep = self.z[self.separatrix_inds]
        s_minus_s0_grid = []
        for _R, _z, si, ig in zip(R, z, self.separatrix_inds, self.index_grid):
            s_minus_s0 = [0.]
            gaps = np.sqrt((_R[1:] - _R[:-1])**2 + (_z[1:] - _z[:-1])**2)
            dists = np.array([np.sum(gaps[:i]) for i in range(len(_R))])
            sep_ind = np.argwhere(ig==si)[0]
            # sep_dist = dists[sep_ind]
            dists -= dists[sep_ind]
            s_minus_s0_grid.append(dists)

        return np.array(s_minus_s0_grid)

    def flatten(self, grid):
        return array([grid[i[0], i[1]] for i in self.inverse_index_grid])

    def verify_poloidal_distance(self):
        z_diff = mean(self.z[self.index_grid][0]) - mean(self.z[self.index_grid][-1])
        poloidal_distance_diff = mean(self.poloidal_distance[self.index_grid][0]) - mean(self.poloidal_distance[self.index_grid][-1])
        if z_diff > 0.:
            # start of grid is upstream
            if poloidal_distance_diff < 0.:
                # poloidal distance is lower upstream -> change this
                corrected_grid = self.poloidal_distance[self.index_grid][::-1]
                corrected_grid -= self.poloidal_distance[self.index_grid][0]
                self.poloidal_distance = corrected_grid.flatten()
        else:
            # start of grid is at target
            if poloidal_distance_diff > 0.:
                # poloidal distance is higher upstream -> change this
                corrected_grid = self.poloidal_distance[self.index_grid][::-1]
                corrected_grid -= self.poloidal_distance[self.index_grid][0]

                self.poloidal_distance = corrected_grid.flatten()

    def get_separatrix_inds(self):
        target = 1.
        inds = []
        for row in self.index_grid:
            inds.append(row[argmin(abs(self.psi[row]-target))])
        return inds

    def find_inverse_index_grid(self, index_grid):
        map = []
        for i, row in enumerate(index_grid):
            for j, val in enumerate(row):
               map.append([val, i, j])
        map = array(map)
        # sort
        sort_inds = argsort(map[:, 0])
        map = map[sort_inds]
        return map[:, 1:]

    def get_alternative_mesh_representation(self, x_quantity: XQuantityOptions, y_quantity: YQuantityOptions):
        return TriangularMesh(R=self.x_quantities[x_quantity]['Quantities'],
                              z=self.y_quantities[y_quantity]['Quantities'],
                              triangles=self.triangle_vertices)

    def plot_mesh_representations(self, is_show=True):

        mesh_combinations = ((XQuantityOptions.R, YQuantityOptions.Z),
                             (XQuantityOptions.PSI, YQuantityOptions.POLOIDAL_DISTANCE_FROM_X_POINT),
                             # (XQuantityOptions.PSI, YQuantityOptions.POLOIDAL_DISTANCE_TO_TARGET),
                             (XQuantityOptions.S_MINUS_S0, YQuantityOptions.POLOIDAL_DISTANCE_FROM_DIVERTOR_ENTRANCE),
                             (XQuantityOptions.PSI, YQuantityOptions.POLOIDAL_DISTANCE_TO_TARGET),
                             )
        fig, axs = plt.subplots(2, 2)
        axs = axs.flatten()
        for ax, combination in zip(axs, mesh_combinations):
            if self.x_quantities[combination[0]]['Unit'] == self.y_quantities[combination[1]]['Unit']:
                aspect = 'equal'
            else:
                aspect = 'auto'
            self.get_alternative_mesh_representation(*combination).draw(ax,
                                                                        x_label=self.x_quantities[combination[0]]['Label'],
                                                                        y_label=self.y_quantities[combination[1]]['Label'],
                                                                        aspect=aspect,
                                                                        color='C0',
                                                                        alpha=1., linewidth=1.,
                                                                        )
        if is_show:
            plt.show()

    def save(self, filepath):
        savez(
            filepath,
            R=self.R,
            z=self.z,
            triangles=self.triangle_vertices,
            index_grid=self.index_grid,
            psi=self.psi,
            poloidal_distance=self.poloidal_distance,
            parallel_distance=self.parallel_distance,
        )

    def save_to_h5_group(self, file_group):
        file_group.create_dataset("R", data=self.R)
        file_group.create_dataset("z", data=self.z)
        file_group.create_dataset("triangles", data=self.triangle_vertices)
        file_group.create_dataset("index_grid", data=self.index_grid)
        file_group.create_dataset("psi", data=self.psi)
        file_group.create_dataset("poloidal_distance", data=self.poloidal_distance)
        file_group.create_dataset("parallel_distance", data=self.parallel_distance)
        file_group.create_dataset("B_x_point", data=self.B_x_point)
        file_group.create_dataset("B_mid_plane", data=self.B_mid_plane)
        file_group.create_dataset("B", data=self.B)
        file_group.create_dataset("poloidal_distance_upstream_to_x_point", data=self.poloidal_distance_upstream_to_x_point)
        file_group.create_dataset("parallel_distance_upstream_to_x_point", data=self.parallel_distance_upstream_to_x_point)
        file_group.create_dataset("poloidal_grazing_angle", data=self.poloidal_grazing_angle)
        file_group.create_dataset("toroidal_grazing_angle", data=self.toroidal_grazing_angle)
        file_group.create_dataset("grazing_angle", data=self.grazing_angle)

    @classmethod
    def load(cls, data: (str, Group, dict), coarse_step: int=1):
        if isinstance(data, str):
            mesh = cls._load_from_file_path(data)
        elif isinstance(data, Group):
            mesh = cls._load_from_h5_group(data)
        elif isinstance(data, dict):
            mesh = cls._load_from_dictionary(data)
        else:
            msg = f"Unrecognised data type: {type(data)}"
            raise TypeError(msg)
        if coarse_step > 1:
            mesh = coarsen_field_aligned_mesh(mesh, step=coarse_step)

        return mesh

    @classmethod
    def _load_from_file_path(cls, filepath: str):
        D = load(filepath, allow_pickle=True)
        return cls(**D)

    @classmethod
    def _load_from_h5_group(cls, file_group: Group):
        data = {k: v[()] for k, v in file_group.items()}
        return cls(**data)

    @classmethod
    def _load_from_dictionary(cls, data: dict):
        return cls(**data)

    def get_upstream_downstream_pairs(self):
        # Index grid: each column represents a flux tube
        upstream_indexes = self.index_grid[0]
        downstream_indexes = self.index_grid[-1]
        # Partner them up
        partnered_indexes = [(up, down) for up, down in zip(upstream_indexes, downstream_indexes)]
        return array(partnered_indexes)

    def get_field_image_parallel_distance(self, vertex_values, shape=(150, 150), pad_fraction=0.01, ax=None,
                        is_draw_limit=True, is_draw_separatrix=True, **kwargs):

        psi_par_mesh = TriangularMesh(R=self.psi, z=self.parallel_distance, triangles=self.triangle_vertices)
        # use this mesh to interpolate
        psi_axis, par_axis, image = psi_par_mesh.get_field_image(vertex_values, shape=shape, pad_fraction=pad_fraction)
        if ax is None:
            return psi_axis, par_axis, image
        else:
            im = ax.imshow(image.T[::-1, :],  # .T[::-1, :],
                           extent=(psi_axis[0], psi_axis[-1], par_axis[0], par_axis[-1]),
                           aspect='auto',
                           **kwargs
                           )
            if is_draw_separatrix and hasattr(self, 'psi'):
                sep_inds = argwhere(self.psi == 1.)
                ax.plot(self.psi[sep_inds], self.parallel_distance[sep_inds], lw=1, color='gray', linestyle='dashed')
            # ax.set_xlim(*psi_limits)
            # ax.set_ylim(*par_limits)
            ax.set_xlabel("Connection length to target (m)")
            ax.set_ylabel("Normalised poloidal magnetic flux, $\psi_N$")
            return im
        return array(partnered_indexes)

    def get_field_image_poloidal_distance(self, vertex_values, shape=(150, 150), pad_fraction=0.01, ax=None,
                        is_draw_limit=True, is_draw_separatrix=True, **kwargs):

        if self.poloidal_mesh is None:
            self.poloidal_mesh = TriangularMesh(R=self.psi,
                                                    z=self.poloidal_distance,
                                                    triangles=self.triangle_vertices)
        # use this mesh to interpolate
        psi_axis, par_axis, image = self.poloidal_mesh.get_field_image(vertex_values, shape=shape, pad_fraction=pad_fraction)
        if ax is None:
            return psi_axis, par_axis, image
        else:
            im = ax.imshow(image.T[::-1, :],  # .T[::-1, :],
                           extent=(psi_axis[0], psi_axis[-1], par_axis[0], par_axis[-1]),
                           aspect='auto',
                           **kwargs
                           )
            if is_draw_separatrix and hasattr(self, 'psi'):
                sep_inds = argwhere(self.psi == 1.)
                ax.plot(self.psi[sep_inds], self.poloidal_distance[sep_inds], lw=1, color='gray', linestyle='dashed')
            # ax.set_xlim(*psi_limits)
            # ax.set_ylim(*par_limits)
            ax.set_xlabel("Poloidal distance from target (m)")
            ax.set_ylabel("Normalised poloidal magnetic flux, $\psi_N$")
            return im

    def integrate_constant_poloidal_distance_chord(self,
                                          vertex_values: ndarray,
                                          poloidal_distance: float,
                                          psi_range: (ndarray, str) = "all"):
        """
        Take a chord at a specified poloidal distance to the target.

        :return:
        """
        if psi_range == "all":
            psi_range = (self.psi.min(), self.psi.max())
        psi = linspace(*psi_range, num=100)
        # psi-=0.1
        poloidal_distance = poloidal_distance * ones_like(psi)
        # use the poloidal mesh to interpolate the field
        if self.poloidal_mesh is None:
            self.poloidal_mesh = TriangularMesh(R=self.psi,
                                                    z=self.poloidal_distance,
                                                    triangles=self.triangle_vertices)
        chord_values = self.poloidal_mesh.interpolate(psi, poloidal_distance, vertex_values=vertex_values)
        # use the poloidal mesh to recover the R and z positions
        extrapolation_val = 0.
        chord_Rs = self.poloidal_mesh.interpolate(psi, poloidal_distance, vertex_values=self.R, extrapolate_val=extrapolation_val)
        chord_zs = self.poloidal_mesh.interpolate(psi, poloidal_distance, vertex_values=self.z, extrapolate_val=extrapolation_val)
        # only take the in-bounds values
        valid_args = np.argwhere((chord_Rs!=extrapolation_val) & (chord_zs!=extrapolation_val)).T[0]
        chord_Rs = chord_Rs[valid_args]
        chord_zs = chord_zs[valid_args]
        chord_values = chord_values[valid_args]
        # take the far left as 0. distance and increase from there
        poloidal_spacing = np.hstack([[0.], np.linalg.norm(np.vstack([
            chord_Rs[1:] - chord_Rs[:-1],
            chord_zs[1:] - chord_zs[:-1]
        ]), axis=0)])

        poloidal_distance = [sum(np.hstack([[0.], poloidal_spacing[:i]])) for i in range(len(poloidal_spacing))]

        # now we integrate over the range
        if len(chord_values) == 0:
            integrated_selection = 0.
        else:
            integrated_selection = simps(y=chord_values, x=poloidal_distance)

        return integrated_selection

    def integrate_constant_psi_chord(self,
                                          vertex_values: ndarray,
                                          psi: float,
                                          poloidal_distance_range: (ndarray, str) = "all"):
        """
        Take a chord at a specified psi.

        :return:
        """
        if poloidal_distance_range == "all":
            poloidal_distance_range = (self.poloidal_distance.min(), self.poloidal_distance.max())
        poloidal_distance = linspace(*poloidal_distance_range, num=100)
        psi = psi * ones_like(poloidal_distance)
        # use the poloidal mesh to interpolate the field
        if self.poloidal_mesh is None:
            self.poloidal_mesh = TriangularMesh(R=self.psi,
                                                    z=self.poloidal_distance,
                                                    triangles=self.triangle_vertices)
        chord_values = self.poloidal_mesh.interpolate(psi, poloidal_distance, vertex_values=vertex_values)
        # use the poloidal mesh to recover the R and z positions
        extrapolation_val = 0.
        chord_Rs = self.poloidal_mesh.interpolate(psi, poloidal_distance, vertex_values=self.R, extrapolate_val=extrapolation_val)
        chord_zs = self.poloidal_mesh.interpolate(psi, poloidal_distance, vertex_values=self.z, extrapolate_val=extrapolation_val)
        # only take the in-bounds values
        valid_args = np.argwhere((chord_Rs!=extrapolation_val) & (chord_zs!=extrapolation_val)).T[0]
        chord_Rs = chord_Rs[valid_args]
        chord_zs = chord_zs[valid_args]
        chord_values = chord_values[valid_args]
        # take the far left as 0. distance and increase from there
        poloidal_spacing = np.hstack([[0.], np.linalg.norm(np.vstack([
            chord_Rs[1:] - chord_Rs[:-1],
            chord_zs[1:] - chord_zs[:-1]
        ]), axis=0)])

        poloidal_distance = [sum(np.hstack([[0.], poloidal_spacing[:i]])) for i in range(len(poloidal_spacing))]

        # now we integrate over the range
        if len(chord_values) == 0:
            integrated_selection = 0.
        else:
            integrated_selection = simps(y=chord_values, x=poloidal_distance)

        # import matplotlib.pyplot as plt
        # plt.title(f"Integrated Value: {integrated_selection:.2e}")
        # plt.plot(poloidal_distance, chord_values)
        # plt.show()
        return integrated_selection

    def integrate_area(self,
                       vertex_values: ndarray,
                       psi_range: (ndarray, str) = "all",
                       poloidal_distance_range: (ndarray, str) = "all"):
        """
        Integrate over an area spanning psi_range and poloidal_distance_range

        :return:
        """
        if poloidal_distance_range == "all":
            poloidal_distance_range = (self.poloidal_distance.min(), self.poloidal_distance.max())
        if psi_range == "all":
            psi_range = (self.psi.min(), self.psi.max())
        poloidal_distance = linspace(*poloidal_distance_range, num=100)
        psi = linspace(*psi_range, num=101)
        psi_grid, poloidal_distance_grid = meshgrid(psi, poloidal_distance)
        # use the poloidal mesh to interpolate the field
        if self.poloidal_mesh is None:
            self.poloidal_mesh = TriangularMesh(R=self.psi,
                                                    z=self.poloidal_distance,
                                                    triangles=self.triangle_vertices)
        grid_values = self.poloidal_mesh.interpolate(psi_grid, poloidal_distance_grid, vertex_values=vertex_values)
        # use the poloidal mesh to recover the R and z positions
        extrapolation_val = 0.
        grid_Rs = self.poloidal_mesh.interpolate(psi_grid, poloidal_distance_grid, vertex_values=self.R, extrapolate_val=extrapolation_val)
        grid_zs = self.poloidal_mesh.interpolate(psi_grid, poloidal_distance_grid, vertex_values=self.z, extrapolate_val=extrapolation_val)
        psi_int_values = []
        # loop over each psi
        for chord_values, chord_Rs, chord_zs in zip(grid_values, grid_Rs, grid_zs):
            # only take the in-bounds values
            valid_args = np.argwhere((chord_Rs!=extrapolation_val) & (chord_zs!=extrapolation_val)).T[0]
            chord_Rs = chord_Rs[valid_args]
            chord_zs = chord_zs[valid_args]
            chord_values = chord_values[valid_args]
            # take the far left as 0. distance and increase from there
            poloidal_spacing = np.hstack([[0.], np.linalg.norm(np.vstack([
                chord_Rs[1:] - chord_Rs[:-1],
                chord_zs[1:] - chord_zs[:-1]
            ]), axis=0)])

            poloidal_distance_along_chord = [sum(np.hstack([[0.], poloidal_spacing[:i]])) for i in range(len(poloidal_spacing))]
            # now we integrate over the range
            # if len(chord_values) !=len(poloidal_distance_along_chord):
            #     print(chord_values)
            #     print(poloidal_distance_along_chord)

            if len(chord_values) == 0:
                integrated_chord = 0.
            else:
                integrated_chord = simps(y=chord_values, x=poloidal_distance_along_chord)
            psi_int_values.append(integrated_chord)
        # Finally we integrate over poloidal distance to the target
        integrated_selection = simps(y=psi_int_values, x=poloidal_distance)
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3)
        # self.get_field_image(vertex_values, ax=axs[0])
        # self.get_field_image_poloidal_distance(vertex_values, ax=axs[1])
        # axs[2].plot(poloidal_distance, psi_int_values)
        # plt.suptitle(f"Integrated Value: {integrated_selection:.2e}")
        # plt.show()
        return integrated_selection

    def integrate_volume(self,
                             vertex_values):
        toroidal_factor = 2. * pi * self.R
        poloidal_plane = self.vertex_width * self.vertex_height
        return vertex_values * toroidal_factor * poloidal_plane
    
    def get_fx_vertices(self):
        # plt.plot(self.Seperatrix_Poloidal_Distance_to_Target, self.Seperatrix_Flux_Expansion)
        # plt.show()
        interp_fx = interp1d(x=self.Seperatrix_Poloidal_Distance_to_Target,
                             y=self.Seperatrix_Flux_Expansion,
                             # assume_sorted=True,
                             kind='linear')
        partners = self.poloidal_distance_to_target_at_accompanying_separatrix.copy()
        # limit
        partners[partners < self.Seperatrix_Poloidal_Distance_to_Target.min()] = self.Seperatrix_Poloidal_Distance_to_Target.min()
        partners[partners > self.Seperatrix_Poloidal_Distance_to_Target.max()] = self.Seperatrix_Poloidal_Distance_to_Target.max()
        return interp_fx(partners)


    def integrate(self, values, is_parallel=True, is_include_flux_expansion=True,
                  is_use_corrected_psi=True,
                  upstream_boundary_condition=None, downstream_boundary_condition=None):
        values = values[self.index_grid]
        n_r, n_c = self.index_grid.shape
        if is_parallel:
            pd = self.parallel_distance_from_x_point[self.index_grid]
        else:
            pd = self.poloidal_distance_from_x_point[self.index_grid]
        dl = pd[1:] - pd[:-1]
        mid_vals = 0.5 * (values[1:] + values[:-1])
        elements =  mid_vals * dl
        if is_include_flux_expansion:
            # total flux expansion
            B_tot = sqrt(self.B[0]**2 + self.B[1]**2)[self.index_grid]
            mid_B = 0.5 * (B_tot[1:] + B_tot[:-1])
            elements /= mid_B

            # start assuming boundary_condition
            cummulative_integral = zeros(shape=(n_r, n_c))
            for i in range(n_r-1):
                cummulative_integral[i+1] = B_tot[i+1] * (cummulative_integral[i]/B_tot[i] + elements[i])

        else:
            cummulative_integral = np.vstack((
                # np.flip(np.cumsum(np.flip(elements, axis=0), axis=0), axis=0),
                np.zeros((1, dl.shape[1])),
                np.cumsum(elements, axis=0)
            ))

        if (upstream_boundary_condition is not None) and (downstream_boundary_condition is not None):
            msg = "Upstream and Downstream boundary conditions cannot both be specified."
            raise ValueError(msg)

        if upstream_boundary_condition is not None:
            cummulative_integral += upstream_boundary_condition
        if downstream_boundary_condition is not None:
            cummulative_integral += downstream_boundary_condition - cummulative_integral[-1]

        return self.flatten(cummulative_integral)




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
        self.nodes = 2 ** self.layers
        self.lims = limits
        self.edges = linspace(limits[0], limits[1], self.nodes + 1)
        self.mids = 0.5 * (self.edges[1:] + self.edges[:-1])

        self.indices = full(self.nodes + 2, fill_value=-1, dtype=int64)
        self.indices[1:-1] = arange(self.nodes)

    def lookup_index(self, values):
        return self.indices[searchsorted(self.edges, values)]

