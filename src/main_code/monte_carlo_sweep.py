import numpy as np
import matplotlib.pyplot as plt
from src.main_code.potentials_calculations import cluster_potential, lennard_jones_potential

def mc_step(coords: np.ndarray, energy: float, beta: float, delta: float, potential_func=lennard_jones_potential):
    '''
    Pertrubs a single atom acording to the metropolis condition

    :param coords: array_like, shape (N, 3), 
        Coordinates of N particles in 3D space.
    :param energy: float,
        Potential energy of the cluster before pertrubation.
    :param beta: float,
        A constant in the Metropolis condition (1/(k_b*T)).
    :param delta: float,
        Maximum displacement of the atom.
    :param potential_func: callable, optional,
        Function to calculate the potential between two particles.
    :returns: tuple of (float, int)
        Potential energy of the cluster after the attempted perturbation, and 1 if the move was accepted or 0 if rejected.
    :raises ValueError: 
        if the input coordinates are not a 2D array with shape (N, 3).
    '''
    #Enforce that the input coordinates are a 2D array with shape (N, 3)
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (N, 3).")

    n_atoms = len(coords)
    idx = np.random.randint(n_atoms) #choose what atom to pertrube randomly

    old_pos = coords[idx].copy()

    move = np.random.uniform(-delta, delta, 3)
    coords[idx] += move

    energy_new = cluster_potential(coords, potential_func)
    #Naive implementation calculating potential for full cluster, can be sped up by reusing potentials
    # and only recalculating for the atom that actually moves
    delta_energy = energy_new - energy
    accepted = 0

    if delta_energy <= 0.0 or np.random.rand() < np.exp(-beta * (delta_energy)): #Metropolis criterion
        pot_energy = energy_new
        accepted +=1
    else:
        coords[idx] = old_pos #revert move if not accepted
        pot_energy = energy
    return pot_energy, accepted


def mc_sweep(coords: np.ndarray, T: float, n_moves: int, delta: float, potential_func=lennard_jones_potential):
    '''
    Performs n_moves Monte Carlo steps and tracks the best found potential energy.

    :param coords: array_like, shape (N, 3), 
        Coordinates of N particles in 3D space.
    :param T: float,
        Temperature of the system.
    :param n_moves: int,
        Amount of Monte Carlo steps to perform for the sweep.
    :param delta: float,
        Maximum displacement of the atom in a single step, should be chosen based on the desired acceptance rate.
    :param potential_func: callable, optional,
        Function to calculate the potential between two particles.
    :returns: tuple of (float, float, np.ndarray, int)
        Final potential energy, lowest potential energy found, coordinates of the lowest-energy state, and number of accepted steps.
    :raises ValueError: 
        if the input coordinates are not a 2D array with shape (N, 3).
    '''

    #Enforce that the input coordinates are a 2D array with shape (N, 3)
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (N, 3).")
    
    energy = cluster_potential(coords, potential_func)
    best_energy = energy
    best_coords = coords.copy()
    n_accepted_steps = 0 #To tune delta later keep track of how often we accept

    k_b = 1 #boltzmann constant, 1 is for standard units in LJ potential
    beta = 1/(k_b*T)

    for i in range(n_moves):
        energy, accepted = mc_step(coords, energy, beta, delta, potential_func)
        n_accepted_steps += accepted
        if energy < best_energy:
            best_energy = energy
            best_coords = coords.copy()
    return energy, best_energy, best_coords, n_accepted_steps