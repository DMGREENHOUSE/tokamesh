import pytest
from numpy import arange, array, sin, cos, pi, isclose, ones
from numpy.random import uniform, seed
from tokamesh import TriangularMesh, BinaryTree
from tokamesh.construction import equilateral_mesh
from hypothesis import given, strategies as st



def triangles_from_grid(R, z, separatrix_ind):
    # build a mesh by splitting SOLPS grid cells
    # flatten out the R and z
    # get the triangle vertices index positions in the list
    triangles = []
    flat_R = R.flatten()
    flat_z = z.flatten()
    is_switch_direction_at_sep = False
    for i in range(R.shape[0]-1):
        for j in range(R.shape[1]-1):
            p1_indexes = (i, j)
            p2_indexes = (i+1, j)
            p3_indexes = (i, j+1)
            p4_indexes = (i+1, j+1)
            if is_switch_direction_at_sep:
                if j >= separatrix_ind:
                    triangles.append([p1_indexes, p2_indexes, p3_indexes])
                    triangles.append([p2_indexes, p3_indexes, p4_indexes])
                else:
                    triangles.append([p1_indexes, p2_indexes, p4_indexes])
                    triangles.append([p1_indexes, p3_indexes, p4_indexes])
            else:
                triangles.append([p1_indexes, p2_indexes, p3_indexes])
                triangles.append([p2_indexes, p3_indexes, p4_indexes])
    return flat_R, flat_z, array(triangles)

@pytest.fixture
def mesh():
    # create a test mesh
    # Get the equilibirum data
    equilibirum_data_fn = 'mastu_fiesta_equilibrium_5.npz'
    equilibirum_data_fp = case_direc + equilibirum_data_fn
    D = load(equilibirum_data_fp)
    R_psi = D['R']
    z_psi = D['z']
    psi_grid = D['psi']

    psi = Equilibrium(R_psi, z_psi, psi_grid, machine='mast-u')
    psi.get_separatrix()

    is_plot = False

    """
    Specify the flux axes for the 4 separatrix-bounded regions
    """
    core_flux_grid = linspace(0.96, 1, 5)

    pfr_flux_grid = linspace(0.95, 1, 6)
    pfr_flux_grid = concatenate([pfr_flux_grid, midpoints(pfr_flux_grid[-3:])])
    pfr_flux_grid.sort()

    outer_sol_flux_grid = linspace(1, 1.19, 20)
    outer_sol_flux_grid = concatenate([outer_sol_flux_grid, midpoints(outer_sol_flux_grid[:5])])
    outer_sol_flux_grid.sort()
    outer_sol_flux_grid = outer_sol_flux_grid[1:]

    inner_sol_flux_grid = linspace(1, 1.03, 4)[1:]

    outer_leg_distance_axis = linspace(0, 1, 30)
    outer_leg_distance_axis = concatenate([outer_leg_distance_axis, midpoints(outer_leg_distance_axis[-3:])])
    outer_leg_distance_axis.sort()

    inner_leg_distance_axis = linspace(0, 1, 6)

    outer_edge_distance_axis = linspace(0, 1, 10)
    inner_edge_distance_axis = linspace(0, 1, 10)

    GG = GridGenerator(
        equilibrium=psi,
        core_flux_grid=core_flux_grid,
        pfr_flux_grid=pfr_flux_grid,
        outer_sol_flux_grid=outer_sol_flux_grid,
        inner_sol_flux_grid=inner_sol_flux_grid,
        inner_leg_distance_axis=inner_leg_distance_axis,
        outer_leg_distance_axis=outer_leg_distance_axis,
        inner_edge_distance_axis=inner_edge_distance_axis,
        outer_edge_distance_axis=outer_edge_distance_axis,
        machine='mast-u'
    )
    is_plot = False
    if is_plot:
        psi.plot_equilibrium(flux_range=(0.95, 1.15))
        psi.plot_stationary_points()
        GG.plot_grids()

    grid = GG.outer_leg_grid
    print('TYPE:')
    print(type(grid))

    is_old_method = False
    if is_old_method:
        # remove rows that contain no point inside the desired region
        R_boundary, z_boundary = mastu_boundary(lower_divertor=True)
        # trim the throat
        relevant_r = hstack([[0.616],
                             R_boundary[4:-1],
                             [0.616]])
        relevant_z = hstack([[-1.587],
                             z_boundary[4:-1],
                             [-1.587]])
        # relevant_r = [1.2, 1.3, 1.3, 1.2]
        # relevant_z = [-1.7, -1.7, -1.8, -1.8]

        divertor_polygon = Polygon([(R, z) for R, z in zip(relevant_r, relevant_z)])
        # make a 2d array of points
        grid_R, grid_z, grid_distance = grid.R.copy(), grid.z.copy(), grid.distance.copy()
        points = []
        for R_row, z_row in zip(grid_R, grid_z):
            row_points = []
            for r, z in zip(R_row, z_row):
                row_points.append(Point(r, z))
            points.append(row_points)
        points = array(points, dtype=object)
        # loop each row and delete as necessary
        rows_to_delete = []
        for i, point_row in enumerate(points):
            row_contains = [divertor_polygon.contains(point) for point in point_row]
            if sum(row_contains) == 0:
                rows_to_delete.append(i)
        points = delete(points, rows_to_delete, 0)
        # grid.psi = delete(grid.psi, rows_to_delete, 0)

        # loop each col and delete as necessary
        cols_to_delete = []
        for i, point_col in enumerate(points.T):
            col_contains = [divertor_polygon.contains(point) for point in point_col]
            if sum(col_contains) == 0:
                cols_to_delete.append(i)
        grid.R = delete(grid.R, rows_to_delete, 0)
        grid.z = delete(grid.z, rows_to_delete, 0)
        grid.R = delete(grid.R, cols_to_delete, 1)
        grid.z = delete(grid.z, cols_to_delete, 1)
        points = delete(points, cols_to_delete, 1)
        # grid_distance = delete(grid_distance, cols_to_delete, 1)

        grid.distance = delete(grid.distance, rows_to_delete, 0)
        grid.psi = delete(grid.psi, cols_to_delete, 0)
        grid.lcfs_index = grid.lcfs_index - sum(cols_to_delete < grid.lcfs_index)
        # grid.lcfs_index = searchsorted(grid.psi, 1.0, side='left')
    row_ind = 10
    slc = slice(row_ind, -1)
    grid.R, grid.z = grid.R[slc, :], grid.z[slc, :]
    print(grid.psi.shape)
    grid.distance = grid.distance[slc]
    grid.lcfs_index = searchsorted(grid.psi, 1.0, side='left')
    print(grid.R.shape)
    print(grid.z.shape)
    R, z, triangles = triangles_from_grid(grid.R, grid.z, grid.lcfs_index)

    mesh = FieldAlignedMesh(triangles=triangles, R=grid.R, z=grid.z, psi=grid.psi)

    # print('Perp derivative:')
    # print(mesh.perpendicular_derivative(use_flux=False))
    # todo continue here, make sure that the flux usage lines up and then double check parallel is working as expected
    #     once this has been achieved, procede to writing priors
    # may want to also think about
    p_deriv = mesh.perpendicular_derivative(use_flux=True)
    p_deriv_old = mesh.perpendicular_derivative_old(use_flux=True)
    ind = 31
    print(mesh.rz_memory)
    print('Perp derivative:')
    print(p_deriv)
    # print(mesh.R[31], mesh.z[31])
    # print(mesh.R[32], mesh.z[32])
    # print(mesh.R[61], mesh.z[61])
    print('Perp derivative Old:')
    print(p_deriv_old)

    import matplotlib.pyplot as plt
    mesh.draw(plt, color='navy')
    plt.plot(*mastu_boundary(), c='black')
    plt.axis('equal')
    plt.xlabel('R (m)')
    plt.ylabel('z (m)')
    plt.show()
    scale = 0.1
    R, z, triangles = equilateral_mesh(
        R_range=(0.0, 1.0), z_range=(0.0, 1.0), resolution=scale
    )
    # perturb the mesh to make it non-uniform
    seed(1)
    L = uniform(0.0, 0.3 * scale, size=R.size)
    theta = uniform(0.0, 2 * pi, size=R.size)
    R += L * cos(theta)
    z += L * sin(theta)
    # build the mesh
    return TriangularMesh(R=R, z=z, triangles=triangles)

def test_interpolate(mesh):
    # As barycentric interpolation is linear, if we use a plane as the test
    # function, it should agree nearly exactly with interpolation result.
    def plane(x, y):
        return 0.37 - 5.31 * x + 2.09 * y

    vertex_values = plane(mesh.R, mesh.z)
    # create a series of random test-points
    R_test = uniform(0.2, 0.8, size=50)
    z_test = uniform(0.2, 0.8, size=50)
    # check the exact and interpolated values are equal
    interpolated = mesh.interpolate(R_test, z_test, vertex_values)
    assert isclose(interpolated, plane(R_test, z_test)).all()

    # now test multi-dimensional inputs
    R_test = uniform(0.2, 0.8, size=[12, 3, 8])
    z_test = uniform(0.2, 0.8, size=[12, 3, 8])
    # check the exact and interpolated values are equal
    interpolated = mesh.interpolate(R_test, z_test, vertex_values)
    assert isclose(interpolated, plane(R_test, z_test)).all()

    # now test giving just floats
    interpolated = mesh.interpolate(0.31, 0.54, vertex_values)
    assert isclose(interpolated, plane(0.31, 0.54)).all()


