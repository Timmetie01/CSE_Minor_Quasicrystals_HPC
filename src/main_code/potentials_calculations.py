from matplotlib.pylab import norm
import numpy as np
from numba import njit
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

def make_lennard_jones_potential(epsilon=1.0, sigma=1.0):
    '''
    Create a Lennard-Jones potential function with specified parameters.

    :param epsilon: float, optional, 
        Depth of the potential well.
    :param sigma: float, optional, 
        Finite distance at which the inter-particle potential is zero.
    :returns: callable,
        A function that calculates the Lennard-Jones potential for a given distance.
    '''
    def potential(r):
        return lennard_jones_potential(r, epsilon, sigma)
    return potential

def lennard_jones_derivative(r, epsilon=1.0, sigma=1.0):
    '''
    Calculate the derivative of the Lennard-Jones potential for a given distance w.r.t. r.

    :param r: float or array_like, 
        Distance between two particles.
    :param epsilon: float, optional, 
        Depth of the potential well.
    :param sigma: float, optional, 
        Finite distance at which the inter-particle potential is zero.
    :returns: float or array_like,
        Lennard-Jones potential's derivative at distance r.
    '''

    r1 = sigma / r
    return 24 * epsilon / sigma * (r1 ** 7 - 2 * r1 ** 13)

def make_lennard_jones_derivative(epsilon=1.0, sigma=1.0):
    '''
    Create a Lennard-Jones potential derivative function with specified parameters.

    :param epsilon: float, optional, 
        Depth of the potential well.
    :param sigma: float, optional, 
        Finite distance at which the inter-particle potential is zero.
    :returns: callable,
        A function that calculates the Lennard-Jones potential's derivative for a given distance.
    '''
    def potential_derivative(r):
        return lennard_jones_derivative(r, epsilon, sigma)
    return potential_derivative


def oscillating_pair_potential(r, frequency=1.0, phase_shift=0.0):
    '''
    Calculate an oscillating pair potential for a given distance.
    https://www.nature.com/articles/nmat4152.pdf

    :param r: float or array_like, 
        Distance between two particles.
    :param frequency: float, optional, 
        Frequency of the oscillation.
    :param phase_shift: float, optional, 
        Phase shift of the oscillation.
    :returns: float or array_like,
        Oscillating pair potential at distance r.   
    '''

    return (r ** -15) + (r ** -3) * np.cos(frequency * (r - 1.25) - phase_shift)

def make_oscillating_pair_potential(frequency=1.0, phase_shift=0.0):
    '''
    Create an oscillating pair potential function with specified parameters.
    https://www.nature.com/articles/nmat4152.pdf

    :param frequency: float, optional, 
        Frequency of the oscillation.
    :param phase_shift: float, optional, 
        Phase shift of the oscillation.
    :returns: callable,
        A function that calculates the oscillating pair potential for a given distance.
    '''
    def potential(r):
        return oscillating_pair_potential(r, frequency, phase_shift)
    return potential

def oscillating_pair_derivative(r, frequency=1.0, phase_shift=0.0):
    '''
    Calculate the derivative of an oscillating pair potential for a given distance.
    https://www.nature.com/articles/nmat4152.pdf

    :param r: float or array_like, 
        Distance between two particles.
    :param frequency: float, optional, 
        Frequency of the oscillation.
    :param phase_shift: float, optional, 
        Phase shift of the oscillation.
    :returns: float or array_like,
        Oscillating pair potential's derivative at distance r.   
    '''

    trig_input = frequency * (r - 1.25) - phase_shift
    return -15 * (r ** -16) - 3 * (r ** -4) * np.cos(trig_input) - frequency * (r ** -3) * np.sin(trig_input)

def make_oscillating_pair_derivative(frequency=1.0, phase_shift=0.0):
    '''
    Create a oscllating pair potential derivative function with specified parameters.

    :param frequency: float, optional, 
        Frequency of the oscillation.
    :param phase_shift: float, optional, 
        Phase shift of the oscillation.
    :returns: callable,
        A function that calculates the oscillating pair potential's derivative for a given distance.
    '''
    def potential_derivative(r):
        return oscillating_pair_derivative(r, frequency, phase_shift)
    return potential_derivative


def single_atom_potential(coords: np.ndarray, idx: int, potential_func=lennard_jones_potential):
    '''
    Calculate the total potential of one atom with the rest for any arbitrary potential function

    :param coords: array_like, shape (N, 3), 
        Coordinates of N particles in 3D space.
    :param idx: int,
        Index of the atom of which the potential will be calculated.
    :param potential_func: callable, optional,
        Function to calculate the potential between two particles.
    :returns: float,
        Total potential of all other atoms with atom number idx.
    '''

    diff = coords - coords[idx]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    return float(np.sum(potential_func(dist[dist>0]))) #Dont include the 0 distance from idx with itself


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
    #The distances between all pairs of particles are calculated by subtracting the original coordinate matrix from the transposed coordinate matrix.
    #The resulting matrix is antisymmetric (negative transpose of itself) and the diagonal is 0 (since the diagonal is the distance of a particle to itself).
    relative_distances = coords[None, :, :] - coords[:, None, :]

    #Change from 3D distances to norms, discarding the 3rd axis of the array. This makes it symmetric.
    relative_distances = np.linalg.norm(relative_distances, axis=-1)

    #Since the matrix is symmetric, only consider the upper triangle. Also discard the diagonal. This way each pair is only calculated once, is should be done for potentials
    return np.sum(potential_func(relative_distances[np.triu_indices_from(relative_distances, k=1)]))

def cluster_potential_gradient(coords, potential_derivative=lennard_jones_derivative):
    '''
    Calculate gradient of the potential for a cluster of particles with any arbitrary potential function derivative

    :param coords: array_like, shape (N, 3), 
        Coordinates of N particles in 3D space.
    :param potential_derivative: callable, optional,
        Function to calculate the derivative of the potential w.r.t. inter-particle distance r
    :returns: array_like, shape (N, 3),
        Gradient of the potential for each particle, similar to coords
    :raises ValueError: 
        if the input coordinates are not a 2D array with shape (N, 3).
    '''

    #Enforce that the input coordinates are a 2D array with shape (N, 3)
    coords = np.asarray(coords)
    num_particles = len(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (N, 3).")

    #Stacks the coordinates into a 3D matrix by repeating the coordinates along a new axis. (Used for pairwise distance calculations)
    #The distances between all pairs of particles are calculated by subtracting the original coordinate matrix from the transposed coordinate matrix.
    #The resulting matrix is antisymmetric (negative transpose of itself) and the diagonal is 0 (since the diagonal is the distance of a particle to itself).
    relative_distances = coords[:, None, :] - coords[None, :, :]

    #removing the diagonal elements to form an NxN-1x3 array. These distances are zero and the distance from an atom to itself should not be counted. Keeping it would also give a ZeroDivisionError.
    relative_distances = relative_distances[~np.eye(num_particles, dtype=bool)].reshape(num_particles, num_particles - 1, 3)

    #Change from 3D distances to norms, discarding the 3rd axis of the array. This makes it symmetric.
    #We must keep the dimensions to allow for division and multiplication in the next line.
    norm_distances = np.linalg.norm(relative_distances, axis=-1, keepdims=True)

    #We now divide the distance vector by its norm to get the unit vector pointing towards each other atom. Multiply by the derivative w.r.t. r and sum the contributions of each particle for each particle to get the gradient.
    return np.sum(relative_distances / norm_distances * potential_derivative(norm_distances), 1)


@njit
def lennard_jones_potential_njit(r, epsilon=1.0, sigma=1.0):
    '''
    Calculate the Lennard-Jones potential for a given distance inside an njit loop.

    :param r: float or array_like, 
        Distance between two particles.
    :param epsilon: float, optional, 
        Depth of the potential well.
    :param sigma: float, optional, 
        Finite distance at which the inter-particle potential is zero.
    :returns: float or array_like,
        Lennard-Jones potential at distance r.
    '''
    inv_r = sigma / r
    inv_r2 = inv_r * inv_r
    r6 = inv_r2 * inv_r2 * inv_r2
    return 4.0 * epsilon * (r6 * r6 - r6)

@njit
def lennard_jones_derivative_njit(r: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    """
    Calculate the derivative of the Lennard-Jones potential for a given distance w.r.t. r.

    :param r: float
        Distance between two particles.
    :param epsilon: float, optional
        Depth of the potential well.
    :param sigma: float, optional
        Finite distance at which the inter-particle potential is zero.
    :returns: float
        Lennard-Jones potential's derivative at distance r.
    """
    r1 = sigma / r
    r2 = r1 * r1
    r6 = r2 * r2 * r2
    return 24.0 * epsilon / sigma * r6 * r1 * (1.0 - 2.0 * r6)

@njit
def single_atom_potential_njit(coords: np.ndarray, idx: int, potential_func=lennard_jones_potential_njit) -> float:
    """
    Calculate the total potential of atom `idx` with all other atoms.
    Compiled with Numba: zero temporary array allocations.
    """
    n_atoms = coords.shape[0]
    xi, yi, zi = coords[idx, 0], coords[idx, 1], coords[idx, 2]
    total = 0.0

    for j in range(n_atoms):
        if j == idx:
            continue
        dx = xi - coords[j, 0]
        dy = yi - coords[j, 1]
        dz = zi - coords[j, 2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        total += potential_func(r)

    return total


@njit
def cluster_potential_njit(
    coords: np.ndarray,
    potential_func=lennard_jones_potential_njit,
) -> float:
    """
    Calculate the total potential for a cluster of particles in 3D space.

    :param coords: ndarray, shape (N, 3)
        Coordinates of N particles in 3D space.
    :param potential_func: @njit callable, optional
        Scalar function calculating the pair potential V(r) between two particles.
    :returns: float
        Total potential energy of the cluster.
    """
    n_atoms = coords.shape[0]
    total_energy = 0.0

    for i in range(n_atoms - 1):
        xi = coords[i, 0]
        yi = coords[i, 1]
        zi = coords[i, 2]
        for j in range(i + 1, n_atoms):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r == 0.0:
                raise ZeroDivisionError("Two particles occupy the exact same position (r = 0).")
            total_energy += potential_func(r)

    return total_energy


@njit
def cluster_potential_gradient_njit(
    coords: np.ndarray,
    potential_derivative=lennard_jones_derivative_njit,
) -> np.ndarray:
    """
    Calculate the gradient of the potential energy for a cluster of particles.

    :param coords: ndarray, shape (N, 3)
        Coordinates of N particles in 3D space.
    :param potential_derivative: @njit callable, optional
        Scalar function calculating dV/dr with respect to inter-particle distance r.
    :returns: ndarray, shape (N, 3)
        Gradient of the potential for each particle.
    """
    n_atoms = coords.shape[0]
    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    for i in range(n_atoms - 1):
        xi = coords[i, 0]
        yi = coords[i, 1]
        zi = coords[i, 2]
        for j in range(i + 1, n_atoms):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r == 0.0:
                raise ZeroDivisionError("Two particles occupy the exact same position (r = 0).")

            # Unit vector (r_i - r_j) / r scaled by dV/dr
            scale = potential_derivative(r) / r
            gx = scale * dx
            gy = scale * dy
            gz = scale * dz

            grad[i, 0] += gx
            grad[i, 1] += gy
            grad[i, 2] += gz

            grad[j, 0] -= gx
            grad[j, 1] -= gy
            grad[j, 2] -= gz

    return grad