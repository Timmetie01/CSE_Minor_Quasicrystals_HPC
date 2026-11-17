import numpy as np
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential
from src.main_code.monte_carlo_sweep  import mc_step, mc_sweep

#Theory behind the code is partly in chapter 14 of https://drive.google.com/file/d/1pxCWOTs2BM8Df1EvQv6NlNiXXUVjdwx8/view?usp=sharing

def attempt_swaps(attempt_count):
    '''
    Attempts to swap configurations between adjacent temperatures
    :param attempt_count: int,
        Number of attempted temperature swaps already performed.
    :returns: tuple of (float, float, np.ndarray, int)
        Final potential energy, lowest potential energy found, coordinates of the lowest-energy state, and number of accepted steps.
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
    replica_at = np.arange(M) #Which replica is at which temperature
    k_b = 1 #boltzmann constant, 1 is for standard units in LJ potential
    betas = 1/(k_b*temps)
    n_moves = len(start_coords[0]) #Each monte carlo sweep tries N moves

    coords_lst = start_coords.copy()
    energies_lst = start_energies.copy()
    for i in range(N_sweeps):
        for j in range(M): #Here we would parallelise
            coords = coords_lst[j].copy()
            energy = energies_lst[j].copy()
            beta = betas[j].copy()
            delta = deltas[j].copy()
            mc_sweep(coords, energy, beta, n_moves, delta, lennard_jones_potential)


M = 10 #Number of replicas
temps = np.arange(M) #Temperature ladder