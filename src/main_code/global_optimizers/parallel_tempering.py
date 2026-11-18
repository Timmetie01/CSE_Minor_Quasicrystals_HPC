import numpy as np
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential, lennard_jones_derivative
from src.main_code.monte_carlo_sweep  import mc_step, mc_sweep
from src.main_code.local_optimizers.fire import FIRE
#Theory behind the code is partly in chapter 14 of https://drive.google.com/file/d/1pxCWOTs2BM8Df1EvQv6NlNiXXUVjdwx8/view?usp=sharing

def attempt_temp_swaps(M, replica_at, betas, energies, attempt_count):
    '''
    Attempts to swap configurations between adjacent temperatures using the parallel tempering Metropolis criterion. Modifies replica_at in place.

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
        #The idea behind this swapping rule: Each replica follows the boltzmann distribution i.e.
        #the probabilty distrobution of being in a state is proportional to e^(-beta*E). This property
        #should be kept by the swaps in the long run. To satisfy this, a restriction is placed on the
        #chance of swapping from one configuration of replicas to the other.
        if delta >= 0 or np.random.rand() < np.exp(delta):
            replica_at[t], replica_at[t + 1] = r_j, r_i
    return


def parallel_tempering_run(start_coords, start_energies, M, temps, deltas, R_c, swap_every, N_sweeps):
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
    :param R_c: float,
        Maximum radius of the cluster.
    :param swap_every: int,
        Frequency at which adjacent temperature swaps are attempted.
    :param N_sweeps: int,
        Total number of Monte Carlo sweeps to perform per replica.
    :returns: tuple of (float, np.ndarray)
        - best_energy -- Lowest potential energy found across all replicas during the simulation.
        - best_coords -- Coordinates corresponding to best_energy, shape (N, 3).
    '''
    # Validate M
    if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)):
        raise TypeError("M must be an integer")

    if M <= 0:
        raise ValueError("M must be a positive integer")

    # Validate R_c
    if isinstance(R_c, (bool, np.bool_)) or not isinstance(R_c, (int, float, np.integer, np.floating)):
        raise TypeError("R_c must be a numeric value")

    if not np.isfinite(R_c) or R_c <= 0:
        raise ValueError("R_c must be finite and positive")

    # Validate swap_every
    if isinstance(swap_every, (bool, np.bool_)) or not isinstance(swap_every, (int, np.integer)):
        raise TypeError("swap_every must be an integer")

    if swap_every <= 0:
        raise ValueError("swap_every must be a positive integer")

    # Validate N_sweeps
    if isinstance(N_sweeps, (bool, np.bool_)) or not isinstance(N_sweeps, (int, np.integer)):
        raise TypeError("N_sweeps must be an integer")

    if N_sweeps <= 0:
        raise ValueError("N_sweeps must be a positive integer")

    # Validate temps
    temps_arr = np.asarray(temps)
    if temps_arr.dtype == bool or not np.issubdtype(temps_arr.dtype, np.number):
        raise TypeError("temps must contain numeric values")

    temps_arr = temps_arr.astype(float, copy=False)
    if temps_arr.ndim != 1 or temps_arr.shape[0] != M:
        raise ValueError("temps must be a 1D array with shape (M)")

    if not np.all(np.isfinite(temps_arr)) or np.any(temps_arr <= 0):
        raise ValueError("temps must contain only finite, positive values")

    if np.any(np.diff(temps_arr) < 0):
        raise ValueError("temps must be ordered from lowest to highest temperature")

    # Validate deltas
    deltas_arr = np.asarray(deltas)
    if deltas_arr.dtype == bool or not np.issubdtype(deltas_arr.dtype, np.number):
        raise TypeError("deltas must contain numeric values")

    deltas_arr = deltas_arr.astype(float, copy=False)
    if deltas_arr.ndim != 1 or deltas_arr.shape[0] != M:
        raise ValueError("deltas must be a 1D array with shape (M)")

    if not np.all(np.isfinite(deltas_arr)) or np.any(deltas_arr <= 0):
        raise ValueError("deltas must contain only finite, positive values")

# Convert coordinates to a floating-point array and protect original input
    coords_lst = np.array(start_coords)
    if coords_lst.dtype == bool or not np.issubdtype(coords_lst.dtype, np.number):
        raise TypeError("start_coords must contain numeric values")

    coords_lst = coords_lst.astype(float, copy=False)
    if coords_lst.ndim != 3 or coords_lst.shape[0] != M or coords_lst.shape[2] != 3:
        raise ValueError("start_coords must have shape (M, N, 3)")

    if coords_lst.shape[1] == 0:
        raise ValueError("At least one particle is required")

    if not np.all(np.isfinite(coords_lst)):
        raise ValueError("start_coords must contain only finite values")

    # Validate start_energies and protect original input
    energies_lst = np.array(start_energies)
    if energies_lst.dtype == bool or not np.issubdtype(energies_lst.dtype, np.number):
        raise TypeError("start_energies must contain numeric values")

    energies_lst = energies_lst.astype(float, copy=False)
    if energies_lst.ndim != 1 or energies_lst.shape[0] != M:
        raise ValueError("start_energies must be a 1D array with shape (M,)")

    if not np.all(np.isfinite(energies_lst)):
        raise ValueError("start_energies must contain only finite values")

    replica_at = np.arange(M)  # Which replica is at which temperature
    k_b = 1.0  # Boltzmann constant, 1 is for standard units in LJ potential
    betas = 1.0 / (k_b * temps_arr)
    n_moves = coords_lst.shape[1]  # Each Monte Carlo sweep tries N moves
    attempt_count = 0

    best_idx = int(np.argmin(energies_lst))
    best_coords = coords_lst[best_idx].copy()
    best_energy = float(energies_lst[best_idx])

    for i in range(N_sweeps):
        for j in range(M):  # Here we would parallelise
            r = replica_at[j]  # Replica number r sits at temp number j
            coords = coords_lst[r]
            energy = energies_lst[r]
            beta = betas[j]
            delta = deltas_arr[j]
            energy, low_en, low_en_coords, num_accepted = mc_sweep(
                coords, energy, beta, n_moves, delta, R_c, lennard_jones_potential
            )
            energies_lst[r] = energy
            if low_en < best_energy:
                best_energy = float(low_en)
                best_coords = low_en_coords.copy()
        if (i + 1) % swap_every == 0:
            attempt_temp_swaps(M, replica_at, betas, energies_lst, attempt_count)
            attempt_count += 1
    return best_energy, best_coords

def make_deltas(temps, d_ref=0.1, T_ref=0.05):
    """Initial guess for delta ~ sqrt(T), anchored at 0.1 sigma for T = 0.05.
    :param temps: array_like, shape (M),
        Initial temperature ladder.
    :param d_ref: float,
        Value for delta at T_ref to base the rest of the values off of.
    :param T_ref: float,
        Temperature at which delta takes value d_ref.
    :returns: array_like, shape (M),
        Initial guess for delta.
    """
    return d_ref * np.sqrt(np.asarray(temps) / T_ref)

def random_cluster(N, R_start, min_dist=0.8, max_tries=10000):
    """N random points inside a sphere of radius R_start, no pair closer than min_dist.
    The minimum distance avoids huge LJ energies from overlapping atoms.
    :param N: int,
        Number of atoms.
    :param R_start: float,
        Maximum distance the atoms may start from the center.
    :param min_dist: float,
        Minimum atoms should start from each other.
    :param max_tries: int,
        Maximum number of attempts to randomly place an atom in a starting position.
    :returns: array_like, shape (N, 3),
        Set of coordinates for the randomly places atoms.
    """
    pos = np.empty((N, 3))
    n = 0
    tries = 0
    while n < N:
        p = np.random.uniform(-R_start, R_start, 3)
        if np.linalg.norm(p) <= R_start and (n == 0 or np.all(np.sum((pos[:n] - p)**2, axis=1) > min_dist**2)):
            pos[n] = p
            n += 1
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Could not place atoms; increase R_start or lower min_dist")
    return pos

def make_start(N, M, R_c):
    """Makes the random starting setup for the PT run.
    :param N: int,
        Number of atoms.
    :param M: int,
        Number of replicas.
    :param R_c: float,
        Maximum distance of atoms from centre for the PT run.
   :returns: tuple of (array_like, array_like)
        - start_energies -- Potential energies of the starting clusters for each replica, shape(M).
        - best_coords -- Starting coordinates for each replica, shape (M, N, 3).
    """
    R_start = R_c - 0.5   # start well inside the wall
    start_coords = np.array([random_cluster(N, R_start) for _ in range(M)])  # (M, N, 3)
    start_energies = np.array([cluster_potential(c) for c in start_coords])                   # (M,)
    return start_energies, start_coords


if __name__ == "__main__":
    np.random.seed(42)

    M = 10 #Number of replicas
    N = 11 #Number of atoms
    R_c = 2.5 #Max radius of cluster to prevent atoms from escaping
    swap_every = 10
    N_sweeps = 2000
    temps = np.geomspace(0.05, 0.40, M)
    deltas = make_deltas(temps)
    energies, coords = make_start(N, M, R_c)

    best_e, best_c =parallel_tempering_run(coords, energies, M, temps, deltas, R_c, swap_every, N_sweeps)
    print("Unquenched global minimum:", best_e)
    #Then we need to quench as this found value will be for T>0 and the minima in the table are at T=0:
    best_c_q = FIRE(best_c, lennard_jones_derivative,
            Nmax=10_000, Nmin=5, f_inc=1.1, f_dec=0.5, 
            alpha0=0.1, f_alpha=0.99, err_tol=1e-5, 
            dt=0.002, max_step=0.05)
    best_e_q = cluster_potential(best_c_q)
    print("Quenched global minimum:" , best_e_q)

def test_pt_escape_local_minima_lj7_cluster():
    """
    Verify parallel tempering escapes the capped-octahedron local minimum (-15.935043)
    of a 7-atom LJ cluster and locates the global minimum (-16.505384) after quenching.
    """
    np.random.seed(42)
    a = 2 ** (1 / 6)
    initial_coords = np.array([
        [a, 0.0, 0.0],
        [-a, 0.0, 0.0],
        [0.0, a, 0.0],
        [0.0, -a, 0.0],
        [0.0, 0.0, a],
        [0.0, 0.0, -a],
        [a, a, a],
    ])

    # Relax into the known local minimum (-15.935043) first
    local_min_coords = FIRE(
        initial_coords,
        lennard_jones_derivative,
        Nmax=10_000,
        Nmin=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha0=0.1,
        f_alpha=0.99,
        err_tol=1e-5,
        dt=0.002,
        max_step=0.05,
    )
    local_min_energy = cluster_potential(local_min_coords, lennard_jones_potential)
    assert np.isclose(local_min_energy, -15.935043, atol=1e-2)

    # Initialize all M replicas trapped in that exact local minimum
    M = 8
    R_c = 3.0
    temps = np.geomspace(0.05, 0.45, M)
    deltas = make_deltas(temps)
    start_coords = np.repeat(local_min_coords[np.newaxis, :, :], M, axis=0)
    start_energies = np.full(M, local_min_energy)

    _, best_c = parallel_tempering_run(
        start_coords=start_coords,
        start_energies=start_energies,
        M=M,
        temps=temps,
        deltas=deltas,
        R_c=R_c,
        swap_every=10,
        N_sweeps=1500,
    )

    # Quench the best configuration found at T > 0 to compare with the T = 0 global minimum
    quenched_coords = FIRE(
        best_c,
        lennard_jones_derivative,
        Nmax=10_000,
        Nmin=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha0=0.1,
        f_alpha=0.99,
        err_tol=1e-5,
        dt=0.002,
        max_step=0.05,
    )
    optimized_energy = cluster_potential(quenched_coords, lennard_jones_potential)

    known_global_min_energy = -16.505384

    assert optimized_energy < local_min_energy
    assert np.isclose(optimized_energy, known_global_min_energy, atol=1e-2)