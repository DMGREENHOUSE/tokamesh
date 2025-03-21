import matplotlib.pyplot as plt
from numpy import ndarray, finfo
from numpy import sqrt, log, pi, tan, dot, cross, identity, isnan
from numpy import absolute, nan, isfinite, minimum, maximum
from numpy import vstack, ndarray, linspace, full, zeros, stack, savez, int64, float64
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from time import perf_counter
import sys
from scipy.sparse import csc_matrix
from numpy.linalg import norm
from numpy import sqrt, arccos, sign, array, cos, sin, pi, allclose

from tokamesh.utilities import build_edge_map, partition_triangles

class RayToCone:
    def __init__(self, start, end):
        start = array(start)
        end = array(end)

        self.r, self.theta, self.phi = self.convert_to_spherical(start, end)
        self.start, self.end = start, end

    def convert_to_spherical(self, start, end):
        displacement = end - start
        r = norm(displacement)
        theta = arccos(displacement[2] / r)
        phi = sign(displacement[1]) * arccos(displacement[0] / norm(displacement[:2]))

        return r, theta, phi

    def convert_to_cartesian(self, origin, r, theta, phi):
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)
        return array((x, y, z)) + origin

    def calc_angles_between_vectors(self, a, b, is_degrees=True):
        cos_theta = dot(a, b) / (norm(a) * norm(b))
        theta_radians = arccos(cos_theta)
        if is_degrees:
            return theta_radians * 180 / pi
        else:
            return theta_radians

    def calc_angles_between_points(self, A, B, C, is_degrees=True):
        AB = B - A
        BC = C - B
        CA = A - C
        AC = C - A
        angles =array((self.calc_angles_between_vectors(AB, BC, is_degrees=is_degrees),
                  self.calc_angles_between_vectors(BC, CA, is_degrees=is_degrees),
                  self.calc_angles_between_vectors(AB, AC, is_degrees=is_degrees)))
        return angles

    def __call__(self,
                 alpha=(0.05,),  #  4),
                 num_points=(6,),  #  12),
                 is_plot_cone=False,
                 is_verify=True):
        """

        :param alpha: given in degrees.
        :param num_points:
        :param is_plot_cone:
        :return:
        """
        # convert to radians
        alpha_rad = array(alpha) * pi / 180.

        # include the centre
        new_coordinates = self.end.copy().reshape((1, 3))  # vstack([new_coordinates, ])

        for a, np in zip(alpha_rad, num_points):
            if np > 0:
                spacing = (2 * pi / np)
                spacings = array([[cos(i * spacing), sin(i * spacing)] for i in range(np)]).T
                new_thetas = self.theta + a * spacings[0]
                new_phis = self.phi + a * spacings[1]
                new_coordinates_contribution = array([self.convert_to_cartesian(self.start, self.r, t, p) for t, p in zip(new_thetas, new_phis)])

                new_coordinates = vstack([new_coordinates, new_coordinates_contribution])
        if is_verify:
            if len(num_points) > 1:
                raise ValueError("Verification only for single cone currently")
            cone_angles  = []
            for new_coord in new_coordinates[1:]:
                cone_angles.append(self.calc_angles_between_points(self.start, self.end, new_coord)[-1])
            point_separation_angles  = []
            for new_coord_a, new_coord_b in zip(new_coordinates[1:-1], new_coordinates[2:]):
                point_separation_angles.append(self.calc_angles_between_points(self.end, new_coord_a, new_coord_b)[-1])
            try:
                assert allclose(alpha[0], cone_angles, rtol=0.3)
            except:
                print(f"Failed: target: {alpha[0]}")
                print("Found: ", cone_angles)
            if num_points[0] > 0:
                try:
                    assert allclose(360/num_points[0], point_separation_angles, rtol=0.3)
                except:
                    print(f"Failed: target: {360/num_points[0]}")
                    print("Found: ", point_separation_angles)

        # check angles around the circle
        if is_plot_cone:
            self.plot(new_coordinates)
        return new_coordinates

    def plot(self, coords):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        # Plot the points
        ax.scatter(*self.start, c='r', marker='o', label='Start')
        ax.scatter(*self.end, c='g', marker='o', label='End')
        for coord in coords:
            ax.scatter(*coord, c='b', marker='^')

        # Set labels for the axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Add a legend
        ax.legend()

        # Set the plot title
        plt.title('3D Scatter Plot of Three Points')

        # Show the plot
        plt.show()


class GeometryCalculator:
    """
    Class for calculating geometry matrices over triangular meshes using
    barycentric linear interpolation.

    :param R: \
        The major radius of each mesh vertex as a 1D numpy array.

    :param z: \
        The z-height of each mesh vertex as a 1D numpy array.

    :param triangles: \
        A 2D numpy array of integers specifying the indices of the vertices which form
        each of the triangles in the mesh. The array must have shape ``(N,3)`` where
        ``N`` is the total number of triangles.

    :param ray_origins: \
        The ``(x,y,z)`` position vectors of the origin of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param ray_ends: \
        The ``(x,y,z)`` position vectors of the end-points of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.
    """

    def __init__(
        self,
        R: ndarray,
        z: ndarray,
        triangles: ndarray,
        ray_origins: ndarray,
        ray_ends: ndarray,
    ):
        # first check the validity of the data
        self.check_geometry_data(R, z, triangles, ray_origins, ray_ends)

        # store the mesh data
        self.R = R
        self.z = z
        self.triangle_vertices = triangles
        self.n_vertices = self.R.size
        self.n_triangles = self.triangle_vertices.shape[0]
        self.GeomFacs = GeometryFactors()

        # calculate the ray data
        diffs = ray_ends - ray_origins
        self.lengths = sqrt((diffs**2).sum(axis=1))
        self.rays = diffs / self.lengths[:, None]
        self.pixels = ray_origins
        self.n_rays = self.lengths.size

        # coefficients for the quadratic representation of the ray radius
        self.q0 = self.pixels[:, 0] ** 2 + self.pixels[:, 1] ** 2
        self.q1 = 2 * (
            self.pixels[:, 0] * self.rays[:, 0] + self.pixels[:, 1] * self.rays[:, 1]
        )
        self.q2 = self.rays[:, 0] ** 2 + self.rays[:, 1] ** 2
        self.sqrt_q2 = sqrt(self.q2)

        # calculate terms used in the linear inequalities
        self.L_tan = -0.5 * self.q1 / self.q2  # ray-distance of the tangency point
        self.R_tan_sqr = self.q0 + 0.5 * self.q1 * self.L_tan
        self.R_tan = sqrt(self.R_tan_sqr)  # major radius of the tangency point
        # z-height of the tangency point
        self.z_tan = self.pixels[:, 2] + self.rays[:, 2] * self.L_tan
        # gradient of the hyperbola asymptote line
        self.m = self.rays[:, 2] / sqrt(self.q2)

        # Construct a mapping from triangles to edges, and edges to vertices
        self.triangle_edges, self.edge_vertices, _ = build_edge_map(
            self.triangle_vertices
        )
        self.R_edges = self.R[self.edge_vertices]
        self.z_edges = self.z[self.edge_vertices]
        self.n_edges = self.edge_vertices.shape[0]

        # pre-calculate the properties of each edge
        self.R_edge_mid = self.R_edges.mean(axis=1)
        self.z_edge_mid = self.z_edges.mean(axis=1)
        self.edge_lengths = sqrt(
            (self.R_edges[:, 0] - self.R_edges[:, 1]) ** 2
            + (self.z_edges[:, 0] - self.z_edges[:, 1]) ** 2
        )
        self.edge_drn = zeros([self.n_edges, 2])
        self.edge_drn[:, 0] = self.R_edges[:, 1] - self.R_edges[:, 0]
        self.edge_drn[:, 1] = self.z_edges[:, 1] - self.z_edges[:, 0]
        self.edge_drn /= self.edge_lengths[:, None]

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

        # for edges in the mesh which are very close to horizontal, the regular
        # intersection calculation becomes inaccurate. Here we set a lower-limit
        # on the allowed size of the z-component of the edge unit vector. Edges
        # with a z-component below this limit will instead use a different
        # intersection calculation for horizontal edges.
        self.min_z_component = 1e-12

    def calculate(self, save_file=None) -> dict[str, ndarray]:
        """
        Calculate the geometry matrix.

        :keyword str save_file: \
            A string specifying a file path to which the geometry matrix data will be
            saved using the numpy ``.npz`` format. If not specified, the geometry matrix
            data is still returned as a dictionary, but is not saved.

        :return: \
            The geometry matrix data as a dictionary of numpy arrays. The structure of
            the dictionary is as follows: ``entry_values`` is a 1D numpy array containing
            the values of all non-zero matrix entries. ``row_indices`` is a 1D numpy
            array containing the row-index of each of the non-zero entries. ``col_indices``
            is a 1D numpy array containing the column-index of each of the non-zero entries.
            ``shape`` is a 1D numpy array containing the dimensions of the matrix. The
            arrays defining the mesh are also stored as ``R``, ``z`` and ``triangles``.
        """
        # clear the geometry factors in case they contain data from a previous calculation
        self.GeomFacs.vertex_map.clear()
        # process the first triangle so we can estimate the run-time
        t_start = perf_counter()
        self.process_triangle(0)
        dt = perf_counter() - t_start

        # use the estimate to break the evaluation into groups
        group_size = max(min(int(1.0 / dt), (self.n_triangles - 1) // 4), 1)
        remainder = (self.n_triangles - 1) % group_size
        ranges = zip(
            range(1, self.n_triangles, group_size),
            range(1 + group_size, self.n_triangles, group_size),
        )

        # calculate the contribution to the matrix for each triangle
        for start, end in ranges:
            [self.process_triangle(i) for i in range(start, end)]

            # print the progress
            f_complete = (end + 1) / self.n_triangles
            eta = int((perf_counter() - t_start) * (1 - f_complete) / f_complete)
            sys.stdout.write(
                f"\r >> Calculating geometry matrix:  [ {f_complete:.1%} complete   ETA: {eta} sec ]           "
            )
            sys.stdout.flush()

        # clean up any remaining triangles
        if remainder != 0:
            [
                self.process_triangle(i)
                for i in range(self.n_triangles - remainder, self.n_triangles)
            ]

        t_elapsed = perf_counter() - t_start
        mins, secs = divmod(t_elapsed, 60)
        hrs, mins = divmod(mins, 60)
        time_taken = f"{int(hrs)}:{int(mins):02d}:{int(secs):02d}"
        sys.stdout.write(
            f"\r >> Calculating geometry matrix:  [ completed in {time_taken} ]           "
        )
        sys.stdout.flush()
        sys.stdout.write("\n")

        # convert the calculated matrix elements to a form appropriate for sparse matrices
        data_vals, vertex_inds, ray_inds = self.GeomFacs.get_sparse_matrix_data()

        data_dict = {
            "entry_values": data_vals,
            "row_indices": ray_inds,
            "col_indices": vertex_inds,
            "shape": array([self.n_rays, self.n_vertices]),
            "R": self.R,
            "z": self.z,
            "triangles": self.triangle_vertices,
        }

        # save the matrix data
        if save_file is not None:
            savez(save_file, **data_dict)

        return data_dict

    def inequality_checks(self, R, z):
        dz = z[:, None] - self.z_tan[None, :]
        mR = R[:, None] * self.m[None, :]
        R_check = R[:, None] > self.R_tan[None, :]
        t = self.m[None, :] * (R[:, None] - self.R_tan[None, :])
        rgn_A = (dz > -mR).all(axis=0)
        rgn_B = (dz < mR).all(axis=0)
        rgn_C = ((t < dz) & (dz < -t) & R_check).all(axis=0)
        rgn_D = (~R_check).all(axis=0)
        return ~(rgn_A | rgn_B | rgn_C | rgn_D)

    def edge_hyperbola_intersections(
        self, R0: float, z0: float, uR: float, uz: float, w: float
    ) -> ndarray:
        u_ratio = uR / uz
        alpha = R0 + (self.pixels[:, 2] - z0) * u_ratio
        beta = self.rays[:, 2] * u_ratio

        # calculate the quadratic coefficients
        a = self.q2 - beta**2
        b = self.q1 - 2 * alpha * beta
        c = self.q0 - alpha**2

        # use the discriminant to check for the existence of the roots
        D = b**2 - 4 * a * c
        real_roots = (D >= 0).nonzero()

        # where roots exists, calculate them
        intersections = full([self.n_rays, 2], nan)
        sqrt_D = sqrt(D[real_roots])
        twice_a = 2 * a[real_roots]
        intersections[real_roots, 0] = -(b[real_roots] + sqrt_D) / twice_a
        intersections[real_roots, 1] = -(b[real_roots] - sqrt_D) / twice_a

        # convert the ray-distances of the intersections to side-displacements
        side_displacements = (
            intersections * self.rays[:, 2, None] + self.pixels[:, 2, None] - z0
        ) / uz
        # reject any intersections which don't occur on the edge itself
        invalid_intersections = absolute(side_displacements) > 0.5 * w
        intersections[invalid_intersections] = nan
        return intersections

    def horizontal_hyperbola_intersections(
        self, R0: float, z0: float, uR: float, uz: float, w: float
    ) -> ndarray:
        """
        Numerically stable calculation of edge-hyperbola intersections for the special
        case where the edge is almost exactly horizontal.
        """
        # find the intersections
        intersections = (z0 - self.pixels[:, 2]) / self.rays[:, 2]
        # convert the ray-distances of the intersections to side-displacements
        R_intersect = sqrt(self.q2 * (intersections - self.L_tan) ** 2 + self.R_tan_sqr)
        side_displacements = (R_intersect - R0) / uR
        # reject any intersections which don't occur on the edge itself
        invalid_intersections = absolute(side_displacements) > 0.5 * w
        intersections[invalid_intersections] = nan
        return intersections

    def process_triangle(self, tri_index: int):
        # a hyperbola can at most intersect a triangle six times, so we create space for this.
        intersections = zeros([self.n_rays, 6])
        # loop over each triangle edge and check for intersections
        edges = self.triangle_edges[tri_index, :]
        for j, edge in enumerate(edges):
            R0 = self.R_edge_mid[edge]
            z0 = self.z_edge_mid[edge]
            uR, uz = self.edge_drn[edge, :]
            w = self.edge_lengths[edge]

            # if the edge is horizontal, a simplified method can be used
            if abs(uz) < self.min_z_component:
                intersections[:, 2 * j] = self.horizontal_hyperbola_intersections(
                    R0, z0, uR, uz, w
                )
                intersections[:, 2 * j + 1] = nan
            else:  # else we need the general intersection calculation
                intersections[:, 2 * j : 2 * j + 2] = self.edge_hyperbola_intersections(
                    R0, z0, uR, uz, w
                )

        # clip all the intersections so that they lie in the allowed range
        maximum(intersections, 0.0, out=intersections)
        minimum(intersections, self.lengths[:, None], out=intersections)
        # now sort the intersections for each ray in order of distance
        intersections.sort(axis=1)

        # After sorting the intersections by distance along the ray, we now have (up to)
        # three pairs of distances where the ray enters and then leaves the triangle.
        # After the clipping operation, if any of these pairs contain the same value,
        # then that intersection occurs outside the allowed range and must be discarded.

        # loop over each of the three pairs:
        for j in range(3):
            equal = intersections[:, 2 * j] == intersections[:, 2 * j + 1]
            if equal.any():  # discard any pairs with equal distance values
                intersections[equal, 2 * j : 2 * j + 2] = nan

        # re-sort the intersections
        intersections.sort(axis=1)
        # check where valid intersections exist, and count how many there are per ray
        valid_intersections = isfinite(intersections)
        intersection_count = valid_intersections.sum(axis=1)
        # At this point, each ray should have an even number of intersections, if any
        # have an odd number then something has gone wrong, so raise an error.
        if (intersection_count % 2 == 1).any():
            raise ValueError(
                f"""\n\n
                \r[ GeometryCalculator error ]
                \r>> One or more rays has an odd number of intersections with
                \r>> triangle {tri_index}. This is typically caused by insufficient
                \r>> floating-point precision in the intersection calculations.
                """
            )

        max_intersections = intersection_count.max()
        for j in range(max_intersections // 2):
            indices = (intersection_count >= 2 * (j + 1)).nonzero()[0]
            # calculate the integrals of the barycentric coords over the intersection path
            L1_int, L2_int, L3_int = self.barycentric_coord_integral(
                l1=intersections[:, 2 * j],
                l2=intersections[:, 2 * j + 1],
                inds=indices,
                tri_index=tri_index,
            )

            # update the vertices with the integrals
            v1, v2, v3 = self.triangle_vertices[tri_index, :]
            self.GeomFacs.update_vertex(
                vertex_ind=v1, ray_indices=indices, integral_vals=L1_int
            )
            self.GeomFacs.update_vertex(
                vertex_ind=v2, ray_indices=indices, integral_vals=L2_int
            )
            self.GeomFacs.update_vertex(
                vertex_ind=v3, ray_indices=indices, integral_vals=L3_int
            )

    def barycentric_coord_integral(
        self, l1: ndarray, l2: ndarray, inds: ndarray[int], tri_index: int
    ) -> tuple[ndarray, ndarray, ndarray]:
        l1_slice = l1[inds]
        l2_slice = l2[inds]
        dl = l2_slice - l1_slice

        R_coeff = radius_hyperbolic_integral(
            l1=l1_slice,
            l2=l2_slice,
            l_tan=self.L_tan[inds],
            R_tan_sqr=self.R_tan_sqr[inds],
            sqrt_q2=self.sqrt_q2[inds],
        )

        z_coeff = self.pixels[inds, 2] * dl + 0.5 * self.rays[inds, 2] * (
            l2_slice**2 - l1_slice**2
        )
        lam1_int = (
            self.lam1_coeffs[tri_index, 0] * R_coeff
            + self.lam1_coeffs[tri_index, 1] * z_coeff
            + self.lam1_coeffs[tri_index, 2] * dl
        )
        lam2_int = (
            self.lam2_coeffs[tri_index, 0] * R_coeff
            + self.lam2_coeffs[tri_index, 1] * z_coeff
            + self.lam2_coeffs[tri_index, 2] * dl
        )
        lam3_int = dl - lam1_int - lam2_int
        return lam1_int, lam2_int, lam3_int

    @staticmethod
    def check_geometry_data(
        R: ndarray,
        z: ndarray,
        triangle_inds: ndarray,
        ray_starts: ndarray,
        ray_ends: ndarray,
    ):
        """
        Check that all the data have the correct shapes / types
        """
        if not all(
            isinstance(arg, ndarray)
            for arg in [R, z, triangle_inds, ray_starts, ray_ends]
        ):
            raise TypeError(
                """\n\n
                \r[ GeometryCalculator error ]
                \r>> All arguments must be of type numpy.ndarray.
                """
            )

        if R.ndim != 1 or z.ndim != 1 or R.size != z.size:
            raise ValueError(
                """\n\n
                \r[ GeometryCalculator error ]
                \r>> 'R' and 'z' arguments must be 1-dimensional arrays of equal length.
                """
            )

        if triangle_inds.ndim != 2 or triangle_inds.shape[1] != 3:
            raise ValueError(
                """\n\n
                \r[ GeometryCalculator error ]
                \r>> 'triangle_inds' argument must be a 2-dimensional array of shape (N,3)
                \r>> where 'N' is the total number of triangles.
                """
            )

        if (
            ray_starts.ndim != 2
            or ray_ends.ndim != 2
            or ray_starts.shape[1] != 3
            or ray_ends.shape[1] != 3
            or ray_ends.shape[0] != ray_starts.shape[0]
        ):
            raise ValueError(
                """\n\n
                \r[ GeometryCalculator error ]
                \r>> 'ray_starts' and 'ray_ends' arguments must be 2-dimensional arrays
                \r>> of shape (M,3), where 'M' is the total number of rays.
                """
            )

        for tag, arr in [
            ("R", R),
            ("z", z),
            ("ray_starts", ray_starts),
            ("ray_ends", ray_ends),
        ]:
            float_precision = finfo(arr.dtype).precision
            if float_precision < 15:
                raise ValueError(
                    f"""\n\n
                    \r[ GeometryCalculator error ]
                    \r>> The '{tag}' argument array has a data-type of {arr.dtype}
                    \r>> with a decimal precision of {float_precision}.
                    \r>> Arrays should use at least 64-bit floats, such that the
                    \r>> decimal precision is 15 or above.
                    """
                )



# class BarycentricGeometryMatrix(object):
#     """
#     Class for calculating geometry matrices over triangular meshes using
#     barycentric linear interpolation.

#     :param R: \
#         The major radius of each mesh vertex as a 1D numpy array.

#     :param z: \
#         The z-height of each mesh vertex as a 1D numpy array.

#     :param triangles: \
#         A 2D numpy array of integers specifying the indices of the vertices which form
#         each of the triangles in the mesh. The array must have shape ``(N,3)`` where
#         ``N`` is the total number of triangles.

#     :param ray_origins: \
#         The ``(x,y,z)`` position vectors of the origin of each ray (i.e. line-of-sight)
#         as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
#         total number of rays.

#     :param ray_ends: \
#         The ``(x,y,z)`` position vectors of the end-points of each ray (i.e. line-of-sight)
#         as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
#         total number of rays.
#     """

#     def __init__(self, R, z, triangles, ray_origins, ray_ends):

#         # first check the validity of the data
#         self.check_geometry_data(R, z, triangles, ray_origins, ray_ends)

#         # store the mesh data
#         self.R = R
#         self.z = z
#         self.triangle_vertices = triangles
#         self.n_vertices = self.R.size
#         self.n_triangles = self.triangle_vertices.shape[0]
#         self.GeomFacs = GeometryFactors()

#         # calculate the ray data
#         diffs = ray_ends - ray_origins
#         self.lengths = sqrt((diffs**2).sum(axis=1))
#         self.rays = diffs / self.lengths[:, None]
#         self.pixels = ray_origins
#         self.n_rays = self.lengths.size

#         # coefficients for the quadratic representation of the ray radius
#         self.q0 = self.pixels[:, 0] ** 2 + self.pixels[:, 1] ** 2
#         self.q1 = 2 * (
#             self.pixels[:, 0] * self.rays[:, 0] + self.pixels[:, 1] * self.rays[:, 1]
#         )
#         self.q2 = self.rays[:, 0] ** 2 + self.rays[:, 1] ** 2
#         self.sqrt_q2 = sqrt(self.q2)

#         # calculate terms used in the linear inequalities
#         self.L_tan = -0.5 * self.q1 / self.q2  # ray-distance of the tangency point
#         self.R_tan_sqr = self.q0 + 0.5 * self.q1 * self.L_tan
#         self.R_tan = sqrt(self.R_tan_sqr)  # major radius of the tangency point
#         # z-height of the tangency point
#         self.z_tan = self.pixels[:, 2] + self.rays[:, 2] * self.L_tan
#         # gradient of the hyperbola asymptote line
#         self.m = self.rays[:, 2] / sqrt(self.q2)

#         # Construct a mapping from triangles to edges, and edges to vertices
#         self.triangle_edges, self.edge_vertices, _ = build_edge_map(
#             self.triangle_vertices
#         )
#         self.R_edges = self.R[self.edge_vertices]
#         self.z_edges = self.z[self.edge_vertices]
#         self.n_edges = self.edge_vertices.shape[0]

#         # pre-calculate the properties of each edge
#         self.R_edge_mid = self.R_edges.mean(axis=1)
#         self.z_edge_mid = self.z_edges.mean(axis=1)
#         self.edge_lengths = sqrt(
#             (self.R_edges[:, 0] - self.R_edges[:, 1]) ** 2
#             + (self.z_edges[:, 0] - self.z_edges[:, 1]) ** 2
#         )
#         self.edge_drn = zeros([self.n_edges, 2])
#         self.edge_drn[:, 0] = self.R_edges[:, 1] - self.R_edges[:, 0]
#         self.edge_drn[:, 1] = self.z_edges[:, 1] - self.z_edges[:, 0]
#         self.edge_drn /= self.edge_lengths[:, None]

#         # pre-calculate barycentric coordinate coefficients for each triangle
#         R1, R2, R3 = [self.R[self.triangle_vertices[:, k]] for k in range(3)]
#         z1, z2, z3 = [self.z[self.triangle_vertices[:, k]] for k in range(3)]
#         self.area = 0.5 * ((z2 - z3) * (R1 - R3) + (R3 - R2) * (z1 - z3))
#         self.lam1_coeffs = (
#             0.5
#             * stack([z2 - z3, R3 - R2, R2 * z3 - R3 * z2], axis=1)
#             / self.area[:, None]
#         )
#         self.lam2_coeffs = (
#             0.5
#             * stack([z3 - z1, R1 - R3, R3 * z1 - R1 * z3], axis=1)
#             / self.area[:, None]
#         )

#     def calculate(self, save_file=None, precision=float64):
#         """
#         Calculate the geometry matrix.

#         :keyword str save_file: \
#             A string specifying a file path to which the geometry matrix data will be
#             saved using the numpy ``.npz`` format. If not specified, the geometry matrix
#             data is still returned as a dictionary, but is not saved.

#         :return: \
#             The geometry matrix data as a dictionary of numpy arrays. The structure of
#             the dictionary is as follows: ``entry_values`` is a 1D numpy array containing
#             the values of all non-zero matrix entries. ``row_indices`` is a 1D numpy
#             array containing the row-index of each of the non-zero entries. ``col_indices``
#             is a 1D numpy array containing the column-index of each of the non-zero entries.
#             ``shape`` is a 1D numpy array containing the dimensions of the matrix. The
#             arrays defining the mesh are also stored as ``R``, ``z`` and ``triangles``.
#         """
#         # clear the geometry factors in case they contain data from a previous calculation
#         self.GeomFacs.vertex_map.clear()
#         # process the first triangle so we can estimate the run-time
#         t_start = perf_counter()
#         self.process_triangle(0)
#         dt = perf_counter() - t_start

#         # use the estimate to break the evaluation into groups
#         group_size = max(min(int(1.0 / dt), (self.n_triangles - 1) // 4), 1)
#         remainder = (self.n_triangles - 1) % group_size
#         ranges = zip(
#             range(1, self.n_triangles, group_size),
#             range(1 + group_size, self.n_triangles, group_size),
#         )

#         # calculate the contribution to the matrix for each triangle
#         for start, end in ranges:
#             [self.process_triangle(i) for i in range(start, end)]

#             # print the progress
#             f_complete = (end + 1) / self.n_triangles
#             eta = int((perf_counter() - t_start) * (1 - f_complete) / f_complete)
#             sys.stdout.write(
#                 f"\r >> Calculating geometry matrix:  [ {f_complete:.1%} complete   ETA: {eta} sec ]           "
#             )
#             sys.stdout.flush()

#         # clean up any remaining triangles
#         if remainder != 0:
#             [
#                 self.process_triangle(i)
#                 for i in range(self.n_triangles - remainder, self.n_triangles)
#             ]

#         t_elapsed = perf_counter() - t_start
#         mins, secs = divmod(t_elapsed, 60)
#         hrs, mins = divmod(mins, 60)
#         time_taken = f"{int(hrs)}:{int(mins):02d}:{int(secs):02d}"
#         sys.stdout.write(
#             f"\r >> Calculating geometry matrix:  [ completed in {time_taken} ]           "
#         )
#         sys.stdout.flush()
#         sys.stdout.write("\n")

#         # convert the calculated matrix elements to a form appropriate for sparse matrices
#         data_vals, vertex_inds, ray_inds = self.GeomFacs.get_sparse_matrix_data()
#         data_vals = array(data_vals, dtype=precision)
#         data_dict = {
#             "entry_values": data_vals,
#             "row_indices": ray_inds,
#             "col_indices": vertex_inds,
#             "shape": array([self.n_rays, self.n_vertices]),
#             "R": self.R,
#             "z": self.z,
#             "triangles": self.triangle_vertices,
#         }

#         # save the matrix data
#         if save_file is not None:
#             savez(save_file, **data_dict)

#         return data_dict

#     def inequality_checks(self, R, z):
#         dz = z[:, None] - self.z_tan[None, :]
#         mR = R[:, None] * self.m[None, :]
#         R_check = R[:, None] > self.R_tan[None, :]
#         t = self.m[None, :] * (R[:, None] - self.R_tan[None, :])
#         rgn_A = (dz > -mR).all(axis=0)
#         rgn_B = (dz < mR).all(axis=0)
#         rgn_C = ((t < dz) & (dz < -t) & R_check).all(axis=0)
#         rgn_D = (~R_check).all(axis=0)
#         return ~(rgn_A | rgn_B | rgn_C | rgn_D)

#     def edge_hyperbola_intersections(
#         self, R0: float, z0: float, uR: float, uz: float, w: float
#     ) -> ndarray:
#         u_ratio = uR / uz
#         alpha = R0 + (self.pixels[:, 2] - z0) * u_ratio
#         beta = self.rays[:, 2] * u_ratio

#         # calculate the quadratic coefficients
#         a = self.q2 - beta**2
#         b = self.q1 - 2 * alpha * beta
#         c = self.q0 - alpha**2

#         # use the discriminant to check for the existence of the roots
#         D = b**2 - 4 * a * c
#         real_roots = (D >= 0).nonzero()

#         # where roots exists, calculate them
#         intersections = full([self.n_rays, 2], nan)
#         sqrt_D = sqrt(D[real_roots])
#         twice_a = 2 * a[real_roots]
#         intersections[real_roots, 0] = -(b[real_roots] + sqrt_D) / twice_a
#         intersections[real_roots, 1] = -(b[real_roots] - sqrt_D) / twice_a

#         # convert the ray-distances of the intersections to side-displacements
#         side_displacements = (
#             intersections * self.rays[:, 2, None] + self.pixels[:, 2, None] - z0
#         ) / uz
#         # print(side_displacements)
#         # print(self.rays.shape)
#         # print(self.pixels.shape)
#         # print(self.rays[:, 2, None])
#         # print(self.pixels[:, 2, None])
#         # quit()
#         # reject any intersections which don't occur on the edge itself
#         invalid_intersections = absolute(side_displacements) > 0.5 * w
#         intersections[invalid_intersections] = nan
#         return intersections

#     def horizontal_hyperbola_intersections(
#         self, R0: float, z0: float, uR: float, uz: float, w: float
#     ) -> ndarray:
#         """
#         Numerically stable calculation of edge-hyperbola intersections for the special
#         case where the edge is almost exactly horizontal.
#         """
#         # find the intersections
#         intersections = (z0 - self.pixels[:, 2]) / self.rays[:, 2]
#         # convert the ray-distances of the intersections to side-displacements
#         R_intersect = sqrt(self.q2 * (intersections - self.L_tan) ** 2 + self.R_tan_sqr)
#         side_displacements = (R_intersect - R0) / uR
#         # reject any intersections which don't occur on the edge itself
#         invalid_intersections = absolute(side_displacements) > 0.5 * w
#         intersections[invalid_intersections] = nan
#         return intersections

#     def process_triangle(self, tri_index: int):
#         # a hyperbola can at most intersect a triangle six times, so we create space for this.
#         intersections = zeros([self.n_rays, 6])
#         # loop over each triangle edge and check for intersections
#         edges = self.triangle_edges[tri_index, :]
#         for j, edge in enumerate(edges):
#             R0 = self.R_edge_mid[edge]
#             z0 = self.z_edge_mid[edge]
#             uR, uz = self.edge_drn[edge, :]
#             w = self.edge_lengths[edge]

#             # if the edge is horizontal, a simplified method can be used
#             if abs(uz) < self.min_z_component:
#                 intersections[:, 2 * j] = self.horizontal_hyperbola_intersections(
#                     R0, z0, uR, uz, w
#                 )
#                 intersections[:, 2 * j + 1] = nan
#             else:  # else we need the general intersection calculation
#                 intersections[:, 2 * j : 2 * j + 2] = self.edge_hyperbola_intersections(
#                     R0, z0, uR, uz, w
#                 )

#             # each time there is an intersection, find the length of the ray to reach
#             # the intersection

#         prev_intersections = intersections.copy()
#         # clip all the intersections so that they lie in the allowed range
#         maximum(intersections, 0.0, out=intersections)
#         minimum(intersections, self.lengths[:, None], out=intersections)
#         # now sort the intersections for each ray in order of distance
#         intersections.sort(axis=1)

#         # After sorting the intersections by distance along the ray, we now have (up to)
#         # three pairs of distances where the ray enters and then leaves the triangle.
#         # After the clipping operation, if any of these pairs contain the same value,
#         # then that intersection occurs outside the allowed range and must be discarded.

#         # loop over each of the three pairs:
#         for j in range(3):
#             equal = intersections[:, 2 * j] == intersections[:, 2 * j + 1]
#             if equal.any():  # discard any pairs with equal distance values
#                 intersections[equal, 2 * j : 2 * j + 2] = nan

#         # re-sort the intersections
#         intersections.sort(axis=1)
#         # check where valid intersections exist, and count how many there are per ray
#         valid_intersections = isfinite(intersections)
#         intersection_count = valid_intersections.sum(axis=1)
#         # At this point, each ray should have an even number of intersections, if any
#         # have an odd number then something has gone wrong, so raise an error.
#         # if (intersection_count % 2 == 1).any():
#         #     raise ValueError("One or more rays has an odd number of intersections")

#         max_intersections = intersection_count.max()
#         for j in range(max_intersections // 2):
#             indices = (intersection_count >= 2 * (j + 1)).nonzero()[0]
#             # calculate the integrals of the barycentric coords over the intersection path
#             L1_int, L2_int, L3_int = self.barycentric_coord_integral(
#                 l1=intersections[:, 2 * j],
#                 l2=intersections[:, 2 * j + 1],
#                 inds=indices,
#                 tri_index=tri_index,
#             )

#             # update the vertices with the integrals
#             v1, v2, v3 = self.triangle_vertices[tri_index, :]
#             self.GeomFacs.update_vertex(
#                 vertex_ind=v1, ray_indices=indices, integral_vals=L1_int
#             )
#             self.GeomFacs.update_vertex(
#                 vertex_ind=v2, ray_indices=indices, integral_vals=L2_int
#             )
#             self.GeomFacs.update_vertex(
#                 vertex_ind=v3, ray_indices=indices, integral_vals=L3_int
#             )

#     def barycentric_coord_integral(
#         self, l1: ndarray, l2: ndarray, inds: ndarray[int], tri_index: int
#     ) -> tuple[ndarray, ndarray, ndarray]:
#         l1_slice = l1[inds]
#         l2_slice = l2[inds]
#         dl = l2_slice - l1_slice

#         R_coeff = radius_hyperbolic_integral(
#             l1=l1_slice,
#             l2=l2_slice,
#             l_tan=self.L_tan[inds],
#             R_tan_sqr=self.R_tan_sqr[inds],
#             sqrt_q2=self.sqrt_q2[inds],
#         )

#         z_coeff = self.pixels[inds, 2] * dl + 0.5 * self.rays[inds, 2] * (
#             l2_slice**2 - l1_slice**2
#         )
#         lam1_int = (
#             self.lam1_coeffs[tri_index, 0] * R_coeff
#             + self.lam1_coeffs[tri_index, 1] * z_coeff
#             + self.lam1_coeffs[tri_index, 2] * dl
#         )
#         lam2_int = (
#             self.lam2_coeffs[tri_index, 0] * R_coeff
#             + self.lam2_coeffs[tri_index, 1] * z_coeff
#             + self.lam2_coeffs[tri_index, 2] * dl
#         )
#         lam3_int = dl - lam1_int - lam2_int
#         return lam1_int, lam2_int, lam3_int

#     @staticmethod
#     def check_geometry_data(
#         R: ndarray,
#         z: ndarray,
#         triangle_inds: ndarray,
#         ray_starts: ndarray,
#         ray_ends: ndarray,
#     ):
#         """
#         Check that all the data have the correct shapes / types
#         """
#         if not all(
#             isinstance(arg, ndarray)
#             for arg in [R, z, triangle_inds, ray_starts, ray_ends]
#         ):
#             raise TypeError(
#                 """\n\n
#                 \r[ GeometryCalculator error ]
#                 \r>> All arguments must be of type numpy.ndarray.
#                 """
#             )

#         if R.ndim != 1 or z.ndim != 1 or R.size != z.size:
#             raise ValueError(
#                 """\n\n
#                 \r[ GeometryCalculator error ]
#                 \r>> 'R' and 'z' arguments must be 1-dimensional arrays of equal length.
#                 """
#             )

#         if triangle_inds.ndim != 2 or triangle_inds.shape[1] != 3:
#             raise ValueError(
#                 """\n\n
#                 \r[ GeometryCalculator error ]
#                 \r>> 'triangle_inds' argument must be a 2-dimensional array of shape (N,3)
#                 \r>> where 'N' is the total number of triangles.
#                 """
#             )

#         if (
#             ray_starts.ndim != 2
#             or ray_ends.ndim != 2
#             or ray_starts.shape[1] != 3
#             or ray_ends.shape[1] != 3
#             or ray_ends.shape[0] != ray_starts.shape[0]
#         ):
#             raise ValueError(
#                 """
#                 [ BarycentricGeometryMatrix error ]
#                 >> 'ray_starts' and 'ray_ends' arguments must be 2-dimensional arrays
#                 >> of shape (M,3), where 'M' is the total number of rays.
#                 """
#             )
        
BarycentricGeometryMatrix = GeometryCalculator

class BarycentricGeometryMatrixCone:
    def __init__(self, R, z, triangles, ray_origins, ray_ends, ray_cone_angles, **ray_cone_kwargs):
        assert len(ray_origins.shape) == 2
        assert len(ray_ends.shape) == 2
        assert ray_origins.shape[1] == 3
        assert ray_ends.shape[1] == 3
        if not hasattr(ray_cone_angles, '__iter__'):
            ray_cone_angles = [ray_cone_angles] * len(ray_origins)
        ray_cone_ends = []
        for r_o, r_e, r_a in zip(ray_origins, ray_ends, ray_cone_angles):
            ray_cone = RayToCone(start=r_o, end=r_e)
            cone_ray_ends = ray_cone(alpha=(r_a, ), **ray_cone_kwargs)
            ray_cone_ends.append(cone_ray_ends)
        ray_cone_ends = array(ray_cone_ends)
        ray_cone_ends = ray_cone_ends.transpose([1, 0, 2])
        # make multiple instances
        general_info = None
        BGMs_data = []
        for cone_ray_end in ray_cone_ends:
            BGM = GeometryCalculator(R=R, z=z,
                                 triangles=triangles,
                                 ray_origins=ray_origins,
                                 ray_ends=cone_ray_end,
                                 )
            cone_ray_data = BGM.calculate()
            if general_info is None:
                requested = ("shape", "R", "z", "triangles")
                general_info = {r: cone_ray_data[r] for r in requested}
            gm_requested = ("entry_values", "row_indices", "col_indices", "shape")
            BGMs_data.append(
                {r: cone_ray_data[r] for r in gm_requested}
            )
        self.general_info = general_info
        self.BGMs_data = BGMs_data


    def calculate(self, save_file=None, ignore_nan=False):
        # make the geometry matrices and average over them.
        BG_matrices = []
        for BGM in self.BGMs_data:
            args = (BGM['entry_values'],
                    (BGM['row_indices'],
                     BGM['col_indices']))
            shape = BGM['shape']
            matrix = csc_matrix(args, shape)
            BG_matrices.append(matrix)

        BGM_mean = sum(BG_matrices)  / len(BG_matrices)
        if ((not ignore_nan) and any(isnan(BGM_mean.data))):
            msg = "Geometry matrix contains NaN points, verify mesh."
            raise ValueError(msg)
        # return to sparse format for saving
        data_dict = {
            "data": BGM_mean.data,
            "indices": BGM_mean.indices,
            "indptr": BGM_mean.indptr,
            **self.general_info}

        # save the matrix data
        if save_file is not None:
            savez(save_file, **data_dict)

        return data_dict

def radius_hyperbolic_integral(l1, l2, l_tan, R_tan_sqr, sqrt_q2):
    u1 = sqrt_q2 * (l1 - l_tan)
    u2 = sqrt_q2 * (l2 - l_tan)
    R1 = sqrt(u1**2 + R_tan_sqr)
    R2 = sqrt(u2**2 + R_tan_sqr)

    ratio = (u2 + R2) / (u1 + R1)
    return 0.5 * (u2 * R2 - u1 * R1 + log(ratio) * R_tan_sqr) / sqrt_q2


class GeometryFactors:
    def __init__(self):
        self.vertex_map = defaultdict(float)

    def update_vertex(self, vertex_ind, ray_indices, integral_vals):
        for ray_idx, value in zip(ray_indices, integral_vals):
            self.vertex_map[(vertex_ind, ray_idx)] += value

    def get_sparse_matrix_data(self):
        vertex_inds = array([key[0] for key in self.vertex_map.keys()], dtype=int)
        ray_inds = array([key[1] for key in self.vertex_map.keys()], dtype=int)
        data_vals = array([v for v in self.vertex_map.values()])
        return data_vals, vertex_inds, ray_inds


def triangle_process(
    calculator: GeometryCalculator,
    triangle_indices: ndarray,
    connection: Connection,
):
    [calculator.process_triangle(i) for i in triangle_indices]
    connection.send(calculator.GeomFacs.vertex_map)


def linear_geometry_matrix(
    R: ndarray, ray_origins: ndarray, ray_ends: ndarray
) -> ndarray:
    """
    Calculates a geometry matrix using 1D linear-interpolation basis functions
    assuming that the emission varies only as a function of major-radius.

    :param R: \
        The major-radius position of each basis function in ascending order.

    :param ray_origins: \
        The ``(x,y,z)`` position vectors of the origin of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.

    :param ray_ends: \
        The ``(x,y,z)`` position vectors of the end-points of each ray (i.e. line-of-sight)
        as a 2D numpy array. The array must have shape ``(M,3)`` where ``M`` is the
        total number of rays.
    """
    # verify the inputs are all numpy arrays
    for name, var in [("R", R), ("ray_origins", ray_origins), ("ray_ends", ray_ends)]:
        if type(var) is not ndarray:
            raise TypeError(
                f"""\n
                \r[ linear_geometry_matrix error ]
                \r>> '{name}' argument must have type: 
                \r>> {ndarray}
                \r>> but instead has type:
                \r>> {type(var)}
                """
            )

    # check all shapes of the inputs
    n_points = R.size
    if R.ndim != 1 or R.size < 3:
        raise ValueError(
            f"""\n
            \r[ linear_geometry_matrix error ]
            \r>> 'R' argument must have one dimension and at least 3 elements,
            \r>> but instead has {R.ndim} dimensions and {R.size} elements.
            """
        )

    if (R[1:] - R[:-1] <= 0).any():
        raise ValueError(
            """\n
            \r[ linear_geometry_matrix error ]
            \r>> 'R' argument be sorted in ascending order, and contain only unique values.
            """
        )

    good_rays = (
        ray_origins.ndim == ray_ends.ndim == 2
        and ray_origins.shape[0] == ray_origins.shape[0]
        and ray_origins.shape[1] == ray_origins.shape[1] == 3
    )
    if not good_rays:
        raise ValueError(
            f"""\n
            \r[ linear_geometry_matrix error ]
            \r>> 'ray_origins' and 'ray_ends' must both have shape (N, 3)
            \r>> where 'N' is the number of rays. Instead, they have shapes
            \r>> {ray_origins.shape}, {ray_ends.shape}
            \r>> respectively.
            """
        )

    # calculate linear basis function coefficients
    grads = zeros([n_points - 1, 2])
    grads[:, 1] = 1.0 / (R[1:] - R[:-1])
    grads[:, 0] = -grads[:, 1]
    offsets = zeros([n_points - 1, 2])
    offsets[:, 0] = -grads[:, 0] * R[1:]
    offsets[:, 1] = -grads[:, 1] * R[:-1]

    # calculate the ray data
    diffs = ray_ends - ray_origins
    lengths = sqrt((diffs**2).sum(axis=1))
    rays = diffs / lengths[:, None]
    n_rays = lengths.size

    # coefficients for the quadratic representation of the ray radius
    q0 = ray_origins[:, 0] ** 2 + ray_origins[:, 1] ** 2
    q1 = 2 * (ray_origins[:, 0] * rays[:, 0] + ray_origins[:, 1] * rays[:, 1])
    q2 = rays[:, 0] ** 2 + rays[:, 1] ** 2
    sqrt_q2 = sqrt(q2)

    # pre-calculate quantities used in intersection and integral
    L_tan = -0.5 * q1 / q2  # ray-distance of the tangency point
    L_tan_sqr = L_tan**2
    R_tan_sqr = q0 + 0.5 * q1 * L_tan  # major radius of the tangency point squared

    # each possible pairing of ray and radius can have up to two intersections
    intersections = zeros([n_rays, n_points, 2])
    # solve for the intersections via quadratic formula
    c = q0[:, None] - R[None, :] ** 2
    discriminant = L_tan_sqr[:, None] - c / q2[:, None]
    discriminant[discriminant < 0] = nan
    roots = sqrt(discriminant)
    intersections[:, :, 0] = L_tan[:, None] - roots
    intersections[:, :, 1] = L_tan[:, None] + roots

    # clip all the intersections so that they lie in the allowed range
    maximum(intersections, 0.0, out=intersections)
    minimum(intersections, lengths[:, None, None], out=intersections)
    intersections[isnan(intersections)] = 0.0

    # now loop over each cell, and add the contribution to the geometry matrix
    G = zeros([n_rays, n_points])
    for cell in range(n_points - 1):
        cell_intersects = concatenate(
            [intersections[:, cell, :], intersections[:, cell + 1, :]], axis=1
        )
        cell_intersects.sort(axis=1)

        for i, j in [(0, 1), (2, 3)]:
            integral = radius_hyperbolic_integral(
                l1=cell_intersects[:, i],
                l2=cell_intersects[:, j],
                l_tan=L_tan,
                R_tan_sqr=R_tan_sqr,
                sqrt_q2=sqrt_q2,
            )
            dl = cell_intersects[:, j] - cell_intersects[:, i]
            G[:, cell] += integral * grads[cell, 0] + offsets[cell, 0] * dl
            G[:, cell + 1] += integral * grads[cell, 1] + offsets[cell, 1] * dl
    return G
