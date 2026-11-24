from matplotlib.pylab import norm
import numpy as np

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


def get_oscillating_pair_spline_coefficients(frequency=6.0, phase_shift=0.59):
    '''
    Calculate the parameters needed to build a cubic spline to smoothen an oscillating pair potential and derivative dV/dr, 
    such that they stop oscillating after three wells.

    :param frequency: float, optional,
        Frequency of the oscillation (5.5 =< frequency =< 9.5).
    :param phase_shift: float, optional,
        Phase shift of the oscillation (0.36 =< phase shift =< 0.82).
    :returns: cutoff, width, V0, dV0, D0, c2, c3,
        Respectively the distance at which the spline begins, the width of the spline and the spline's four coefficients.
    '''

    if (frequency < 5.5) or (frequency > 9.5):
        raise ValueError("frequency outside of permitted region.")

    if (phase_shift < 0.36) or (phase_shift > 0.82):
            raise ValueError("phase shift outside of permitted region.")
    
    cutoff = 1.25 + (phase_shift + 3.3 * np.pi) / frequency

    width = 0.5 * np.pi / abs(frequency)

    V0 = float((cutoff**-15) + (cutoff**-3) * np.cos(frequency * (cutoff - 1.25) - phase_shift))
    dV0 = float(-15.0 * (cutoff**-16) - 3.0 * (cutoff**-4) * np.cos(frequency * (cutoff - 1.25) - phase_shift) - frequency * (cutoff**-3) * np.sin(frequency * (cutoff - 1.25) - phase_shift))
    D0 = width * dV0
    c2 = -3.0 * V0 - 2.0 * D0
    c3 = 2.0 * V0 + D0

    return cutoff, width, V0, D0, c2, c3


def make_tapered_oscillating_pair_potential(frequency=6.0, phase_shift=0.59):
    '''
    Calculate an oscillating pair potential for a given distance, 
    tapered using a cubic spline to stop oscillating after three wells.
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
    cutoff, width, V0, D0, c2, c3 = get_oscillating_pair_spline_coefficients(frequency, phase_shift)
    r_end = cutoff + width
    def potential(r):
        r_arr = np.asarray(r, dtype=float)
        out = np.zeros_like(r_arr)

        core = r_arr < cutoff
        if np.any(core):
            r_core = r_arr[core]
            out[core] = (r_core**-15) + (r_core**-3) * np.cos(frequency * (r_core - 1.25) - phase_shift)
        
        tail = (~core) & (r_arr < r_end)
        if np.any(tail):
            t = (r_arr[tail] - cutoff) / width
            out[tail] = V0 + t * (D0 + t * (c2 + t * c3))

        return out
    return potential


def make_tapered_oscillating_pair_derivative(frequency=6.0, phase_shift=0.59):
    '''
    Calculate an oscillating pair derivative dV/dr for a given distance, 
    tapered using a cubic spline to stop oscillating after three wells.
    https://www.nature.com/articles/nmat4152.pdf

    :param r: float or array_like,
        Distance between two particles.
    :param frequency: float, optional,
        Frequency of the oscillation.
    :param phase_shift: float, optional,
        Phase shift of the oscillation.
    :returns: float or array_like,
        Oscillating pair's derivative dV/dr at distance r and it's derivative.
    '''
    cutoff, width, V0, D0, c2, c3 = get_oscillating_pair_spline_coefficients(frequency, phase_shift)
    r_end = cutoff + width
    def potential_derivative(r):
        r_arr = np.asarray(r, dtype=float)
        out = np.zeros_like(r_arr)

        core = r_arr < cutoff
        if np.any(core):
            r_core = r_arr[core]
            trig_input = frequency * (r_core - 1.25) - phase_shift
            out[core] = (-15.0 * (r_core**-16) - 3.0 * (r_core**-4) * np.cos(trig_input) - frequency * (r_core**-3) * np.sin(trig_input))

        tail = (~core) & (r_arr < r_end)
        if np.any(tail):
            t = (r_arr[tail] - cutoff) / width
            out[tail] = (D0 + t * (2.0 * c2 + 3.0 * c3 * t)) / width
        
        return out
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

    coords = np.asarray(coords)
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


