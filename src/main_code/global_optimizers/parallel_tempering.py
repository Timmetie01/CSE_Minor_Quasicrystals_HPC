import numpy as np
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential
from src.main_code.monte_carlo_sweep  import mc_step, mc_sweep

#Theory behind the code is partly in chapter 14 of https://drive.google.com/file/d/1pxCWOTs2BM8Df1EvQv6NlNiXXUVjdwx8/view?usp=sharing

def attempt_temp_swaps(M, replica_at, betas, energies, attempt_count):
    '''
    Attempts to swap configurations between adjacent temperatures using the parallel tempering Metropolis criterion.

    :param M: int,
        Number of replicas.
    :param replica_at: np.ndarray, shape (M),
        Mapping array where ``replica_at[t]`` gives the index of the replica currently at temperature slot ``t``.
    :param betas: np.ndarray, shape (M),
        Inverse temperatures (1 / (k_B * T)) for each temperature slot, ordered from lowest to highest temperature.
    :param energies: np.ndarray, shape (M),
        Current potential energies indexed by replica ID.
    :param attempt_count: int,
        Number of attempted temperature swaps already performed (used to alternate between even and odd pairs).
    :returns: None
    '''
    start = attempt_count % 2 #Vary checking even and odd pairs of temperatures
    # instead of varying this you could also just choose random pairs to check
    for t in range(start, M - 1, 2):
        r_i, r_j = replica_at[t], replica_at[t + 1]   # replicas at adjacent only temperatures
        delta = (betas[t] - betas[t + 1]) * (energies[r_i] - energies[r_j])
        if delta >= 0 or np.random.rand() < np.exp(delta):
            replica_at[t], replica_at[t + 1] = r_j, r_i
    return


def parallel_tempering_run(start_coords, start_energies, M, temps, deltas, swap_every, N_sweeps):
    '''
    Runs a parallel tempering Monte Carlo simulation across M temperature replicas to search for the global minimum potential energy.

    :param start_coords: array_like, shape (M, N, 3),
        Initial 3D coordinates of N particles for each of the M replicas.
    :param start_energies: array_like, shape (M),
        Initial potential energies corresponding to start_coords for each replica.
    :param M: int,
        Number of temperature replicas.
    :param temps: array_like, shape (M),
        Temperature ladder ordered from lowest to highest temperature.
    :param deltas: array_like, shape (M),
        Maximum single-atom displacement for each MC step for each temperature slot.
    :param swap_every: int,
        Frequency at which adjacent temperature swaps are attempted.
    :param N_sweeps: int,
        Total number of Monte Carlo sweeps to perform per replica.
    :returns: tuple of (float, np.ndarray)
        - best_energy -- Lowest potential energy found across all replicas during the simulation.
        - best_coords -- Coordinates corresponding to best_energy, shape (N, 3).
    '''
    replica_at = np.arange(M) #Which replica is at which temperature
    k_b = 1 #boltzmann constant, 1 is for standard units in LJ potential
    betas = 1/(k_b*temps)
    n_moves = len(start_coords[0]) #Each monte carlo sweep tries N moves
    attempt_count = 0

    best_idx = int(np.argmin(start_energies))
    best_coords = start_coords[best_idx].copy()
    best_energy = start_energies[best_idx]

    coords_lst = np.array(start_coords, dtype=float, copy=True)
    energies_lst = start_energies.copy()

    for i in range(N_sweeps):
        for j in range(M): #Here we would parallelise
            r = replica_at[j] #Replica number r sits at temp number j, so index as below
            coords = coords_lst[r]
            energy = energies_lst[r]
            beta = betas[j].copy()
            delta = deltas[j].copy()
            energy, low_en, low_en_coords, num_accepted = mc_sweep(coords, energy, beta, n_moves, delta, lennard_jones_potential)
            energies_lst[r] = energy
            if low_en < best_energy:
                best_energy = low_en
                best_coords = low_en_coords.copy()
        if (i+1) % swap_every == 0: 
            attempt_temp_swaps(M, replica_at, betas, energies_lst, attempt_count)
            attempt_count += 1
    return best_energy, best_coords

M = 10 #Number of replicas
temps = np.arange(M) +1 #Temperature ladder