import numpy as np
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential
from src.main_code.monte_carlo_sweep  import mc_step, mc_sweep

#Theory behind the code is partly in chapter 14 of https://drive.google.com/file/d/1pxCWOTs2BM8Df1EvQv6NlNiXXUVjdwx8/view?usp=sharing


k_b = 1 #boltzmann constant, 1 is for standard units in LJ potential

M = 10 #Number of replicas
replica_at = np.arange(M) #Which replica is at which temperature
temps = np.arange(M) #Temperature ladder
betas = 1/(k_b*temps)
energies = np.zeros(M)

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