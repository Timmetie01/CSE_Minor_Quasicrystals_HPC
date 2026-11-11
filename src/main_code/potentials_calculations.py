import numpy as np
import pytest

def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    '''
    Calculate the Lennard-Jones potential for a given distance.

    :param r: float or array_like, 
        Distance between two particles.
    :param epsilon: float, optional, 
        Depth of the potential well.
    :param sigma: float, optional, 
        Finite distance at which the inter-particle potential is zero.
    :returns: float or array_like,
        Lennard-Jones potential at distance r.
    '''

    r6 = (sigma / r) ** 6
    return 4 * epsilon * (r6 * r6 - r6)


def cluster_potential(coords: np.ndarray, potential_func=lennard_jones_potential):
    '''
    Calculate the total potential for a cluster of particles with any arbitrary potential function

    :param coords: array_like, shape (N, 3), 
        Coordinates of N particles in 3D space.
    :param potential_func: callable, optional,
        Function to calculate the potential between two particles.
    :returns: float,
        Total potential of the cluster.
    :raises ValueError: 
        if the input coordinates are not a 2D array with shape (N, 3).
    :raises RuntimeWarning:
        if any two particles are at the same position, resulting in a distance of zero.
    '''

    #Enforce that the input coordinates are a 2D array with shape (N, 3)
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (N, 3).")

    #Stacks the coordinates into a 3D matrix by repeating the coordinates along a new axis. (Used for pairwise distance calculations)
    coord_matrix = np.broadcast_to(coords[:, None, :], (len(coords), len(coords), 3)).copy()

    #The distances between all pairs of particles are calculated by subtracting the transposed coordinate matrix from the original coordinate matrix.
    #The resulting matrix is antisymmetric (negative transpose of itself) and the diagonal is 0 (since the diagonal is the distance of a particle to itself).
    distances = coord_matrix - coord_matrix.transpose(1, 0, 2)

    #Change from 3D distances to norms, discarding the 3rd axis of the array. This makes it symmetric.
    distances = np.linalg.norm(distances, axis=-1)

    #Since the matrix is symmetric, only consider the upper triangle. Also discard the diagonal. This way each pair is only calculated once, is should be done for potentials
    total_potential = np.sum(potential_func(distances[np.triu_indices_from(distances, k=1)]))

    #print(distances[np.triu_indices_from(distances, k=1)])
    return total_potential





    