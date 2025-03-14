import numpy as np
from numpy.linalg import norm
from numpy import sqrt, arccos, sign, array, cos, sin, pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class RayToCone:
    def __init__(self, start, end):
        start = np.array(start)
        end = np.array(end)

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

    def __call__(self, alpha=5, num_points=10):
        alpha = 10./180.
        spacing = (2 * pi / num_points)
        spacings = array([[cos(i * spacing), sin(i * spacing)] for i in range(num_points)]).T
        new_thetas = self.theta + alpha * spacings[0]
        new_phis = self.phi + alpha * spacings[1]
        new_coordinates = np.array([self.convert_to_cartesian(self.start, self.r, t, p) for t, p in zip(new_thetas, new_phis)])
        self.plot(new_coordinates)

    def plot(self, coords):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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


# define a central coordinate
A = np.array((0., 0., 0.))

# define a point on sphere surface
B = np.array((1., 1., 1.))
rc = RayToCone(start=A, end=B)

rc()