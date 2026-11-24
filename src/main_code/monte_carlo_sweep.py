import numpy as np
import matplotlib.pyplot as plt
from src.main_code.potentials_calculations import cluster_potential, lennard_jones_potential, single_atom_potential

#More explanation can be found in chapter 3 of https://drive.google.com/file/d/1pxCWOTs2BM8Df1EvQv6NlNiXXUVjdwx8/view?usp=sharing


def mc_step(coords: np.ndarray, energy: float, beta: float, delta: float, R_c: float, potential_func=lennard_jones_potential):
    '''
    Pertrubs a single atom according to the Metropolis condition. Updates coords in place.

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
    :param R_c: float,
        Maximum radius from the origin that atoms may move.
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
    old_energy_idx = single_atom_potential(coords, idx, lennard_jones_potential)

    move = np.random.uniform(-delta, delta, 3)
    coords[idx] += move
    accepted = 0

    if old_pos[0]**2 + old_pos[1]**2 + old_pos[2]**2 < R_c**2: #dont move too far from the centre or atoms will escape
        coords[idx] = old_pos #revert move if not accepted
        pot_energy = energy
        return pot_energy, accepted
    new_energy_idx = single_atom_potential(coords, idx, lennard_jones_potential)
    energy_new = energy -old_energy_idx +new_energy_idx
    delta_energy = energy_new - energy

    if delta_energy <= 0.0 or np.random.rand() < np.exp(-beta * (delta_energy)): #Metropolis criterion
        pot_energy = energy_new
        accepted +=1
    else:
        coords[idx] = old_pos #revert move if not accepted
        pot_energy = energy
    return pot_energy, accepted

import math

def mc_sweep(coords: np.ndarray, energy: float, beta: float, n_moves: int, delta: float, R_c: float, potential_func=lennard_jones_potential):
    '''
    Performs n_moves Monte Carlo steps and tracks the best found potential energy. Updates coords in place.

    :param coords: array_like, shape (N, 3), 
        Coordinates of N particles in 3D space.
    :param energy: float,
        Potential energy of the system.
    :param beta: float,
        Temperature dependeent constant of the system.
    :param n_moves: int,
        Amount of Monte Carlo steps to perform for the sweep.
    :param delta: float,
        Maximum displacement of the atom in a single step, should be chosen based on the desired acceptance rate.
    :param potential_func: callable, optional,
        Function to calculate the potential between two particles.
    :param R_c: float,
        Maximum radius from the origin that atoms may move.
    :returns: tuple of (float, float, np.ndarray, int)
        Final potential energy, lowest potential energy found, coordinates of the lowest-energy state, and number of accepted steps.
    :raises ValueError: 
        if the input coordinates are not a 2D array with shape (N, 3).
    '''
    n_atoms = len(coords)
    R_c_sq = R_c * R_c
    best_energy = energy
    best_coords = coords.copy()
    n_accepted_steps = 0 #To tune delta later keep track of how often we accept

    # Pre-generate all rng
    indices = np.random.randint(0, n_atoms, size=n_moves)
    moves = np.random.uniform(-delta, delta, size=(n_moves, 3))
    rand_u = np.random.rand(n_moves)

    for i in range(n_moves):
        idx = indices[i] #choose what atom to pertrube randomly
        old_pos = coords[idx].copy()
        new_pos = old_pos + moves[i]

        if new_pos[0]**2 + new_pos[1]**2 + new_pos[2]**2 > R_c_sq: #dont move too far from the centre or atoms will escape
            continue

        old_energy_idx = single_atom_potential(coords, idx, potential_func)
        coords[idx] = new_pos
        new_energy_idx = single_atom_potential(coords, idx, potential_func)
        delta_energy = new_energy_idx - old_energy_idx

        if delta_energy <= 0.0 or rand_u[i] < np.exp(-beta * delta_energy): #Metropolis criterion
            energy += delta_energy
            n_accepted_steps += 1
            if energy < best_energy:
                best_energy = energy
                best_coords = coords.copy()
        else:
            coords[idx] = old_pos #revert move if not accepted

    return energy, best_energy, best_coords, n_accepted_steps