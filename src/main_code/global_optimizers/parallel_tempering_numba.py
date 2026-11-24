import time
import numpy as np
from numba import njit
from src.main_code.potentials_calculations import (
    lennard_jones_potential,
    cluster_potential,
    lennard_jones_derivative,
    lennard_jones_potential_njit,
    single_atom_potential_njit,
)
from src.main_code.monte_carlo_sweep import mc_sweep, mc_sweep_njit
from src.main_code.local_optimizers.fire import FIRE
from src.main_code.local_optimizers.bfgs import LBFGS
from src.main_code.visualization import plot_cluster
from data.data_handling import highscores


@njit
def attempt_temp_swaps(M, replica_at, betas, energies, attempt_count):
    """
    Attempts to swap configurations between adjacent temperatures using the parallel tempering Metropolis criterion.
    Modifies replica_at in place.
    """
    start = attempt_count % 2
    for t in range(start, M - 1, 2):
        r_i, r_j = replica_at[t], replica_at[t + 1]
        delta = (betas[t] - betas[t + 1]) * (energies[r_i] - energies[r_j])
        if delta >= 0.0 or np.random.rand() < np.exp(delta):
            replica_at[t], replica_at[t + 1] = r_j, r_i

def temps_to_rho(temps: np.ndarray) -> np.ndarray:
    """
    Converts strictly increasing temperatures on [T_min, T_max] to log-spacings rho.
    """
    diffs = np.diff(temps)
    if np.any(diffs <= 0):
        raise ValueError("Initial temperatures must be strictly increasing to compute rho.")
    span = temps[-1] - temps[0]
    return np.log(diffs / span)


@njit
def update_temp_ladder(
    rho: np.ndarray,
    energies: np.ndarray,
    replica_at: np.ndarray,
    betas: np.ndarray,
    T_min: float,
    T_max: float,
    k_b: float,
    step_size: float,
    rho_bounds: tuple = (-6.0, 6.0),
):
    """
    Updates interior temperatures on a fixed [T_min, T_max] interval via
    Robbins-Monro equal-acceptance-rate adaptation (Atchadé et al. / Vousden et al.).
    """
    M = betas.shape[0]
    n_intervals = M - 1

    # Compute current adjacent swap acceptance probabilities
    accept_probs = np.empty(n_intervals, dtype=np.float64)
    mean_accept = 0.0
    for t in range(n_intervals):
        e_low = energies[replica_at[t]]
        e_high = energies[replica_at[t + 1]]
        log_ratio = (betas[t] - betas[t + 1]) * (e_low - e_high)
        p = 1.0 if log_ratio >= 0.0 else np.exp(log_ratio)
        accept_probs[t] = p
        mean_accept += p
    mean_accept /= n_intervals

    # Update rho toward equalizing adjacent acceptance rates around mean_accept
    rho_next = np.empty(n_intervals, dtype=np.float64)
    max_rho = -1e300
    for t in range(n_intervals):
        val = rho[t] + step_size * (accept_probs[t] - mean_accept)
        if val < rho_bounds[0]:
            val = rho_bounds[0]
        elif val > rho_bounds[1]:
            val = rho_bounds[1]
        rho_next[t] = val
        if val > max_rho:
            max_rho = val

    # Shift rho_next by max_rho (log-sum-exp trick) to prevent overflow and eliminate drift
    weights = np.empty(n_intervals, dtype=np.float64)
    sum_w = 0.0
    for t in range(n_intervals):
        rho_next[t] -= max_rho
        if rho_next[t] < rho_bounds[0]:
            rho_next[t] = rho_bounds[0]
        w = np.exp(rho_next[t])
        weights[t] = w
        sum_w += w

    # Reconstruct temperatures on [T_min, T_max]
    temps_next = np.empty(M, dtype=np.float64)
    betas_next = np.empty(M, dtype=np.float64)

    temps_next[0] = T_min
    betas_next[0] = 1.0 / (k_b * T_min)

    span = T_max - T_min
    cum_w = 0.0
    for t in range(n_intervals - 1):
        cum_w += weights[t] / sum_w
        t_val = T_min + span * cum_w
        temps_next[t + 1] = t_val
        betas_next[t + 1] = 1.0 / (k_b * t_val)

    # Pin the top endpoint identically to T_max (avoids floating-point drift)
    temps_next[M - 1] = T_max
    betas_next[M - 1] = 1.0 / (k_b * T_max)

    return rho_next, temps_next, betas_next


@njit
def _pt_loop_njit(
    coords_lst,
    energies_lst,
    M,
    temps_arr,
    deltas_arr,
    R_c,
    swap_every,
    quench_every,
    N_sweeps,
    potential_func_njit,
    local_optimisor,
    adapt_fraction=1.0,
):
    """
    Parallel Tempering inner loop using numba.
    """
    T_min = temps_arr[0]
    T_max = temps_arr[-1]
    rho = np.log(np.diff(temps_arr) / (T_max - T_min))
    replica_at = np.arange(M)
    k_b = 1.0
    betas = 1.0 / (k_b * temps_arr)

    n_atoms = coords_lst.shape[1]
    n_moves = n_atoms
    attempt_count = 0

    log_deltas = np.log(deltas_arr)
    log_d_min = np.log(1e-4)
    log_d_max = np.log(R_c)
    slot_accepted = np.zeros(M, dtype=np.float64)
    moves_per_window = float(swap_every * n_moves)
    adapt_cutoff = int(adapt_fraction * N_sweeps)

    best_idx = 0
    best_energy = energies_lst[0]
    for m in range(1, M):
        if energies_lst[m] < best_energy:
            best_energy = energies_lst[m]
            best_idx = m
    best_coords = coords_lst[best_idx].copy()

    for i in range(N_sweeps):
        for j in range(M):
            r = replica_at[j]
            energy, low_en, low_en_coords, num_accepted = mc_sweep_njit(
                coords_lst[r],
                energies_lst[r],
                n_atoms,
                betas[j],
                n_moves,
                deltas_arr[j],
                R_c,
                potential_func_njit,
            )
            energies_lst[r] = energy
            slot_accepted[j] += num_accepted
            if low_en < best_energy:
                best_energy = low_en
                best_coords = low_en_coords.copy()

        if (i + 1) % swap_every == 0:
            for m in range(M):
                cx = 0.0
                cy = 0.0
                cz = 0.0
                for a in range(n_atoms):
                    cx += coords_lst[m, a, 0]
                    cy += coords_lst[m, a, 1]
                    cz += coords_lst[m, a, 2]
                cx /= n_atoms #Average x pos of the atoms, used to recenter the cluster to (0,0,0)
                cy /= n_atoms
                cz /= n_atoms
                for a in range(n_atoms):
                    coords_lst[m, a, 0] -= cx
                    coords_lst[m, a, 1] -= cy
                    coords_lst[m, a, 2] -= cz

            attempt_temp_swaps(M, replica_at, betas, energies_lst, attempt_count)
            attempt_count += 1

            if i < adapt_cutoff:
                gamma_n = 0.5 / ((attempt_count + 50) ** 0.6)
                rho, temps_arr, betas = update_temp_ladder(
                rho, energies_lst, replica_at, betas, T_min, T_max, k_b, gamma_n
                )

                for j in range(M):
                    acc_rate = slot_accepted[j] / moves_per_window
                    ld = log_deltas[j] + gamma_n * (acc_rate - 0.234)
                    if ld < log_d_min:
                        ld = log_d_min
                    elif ld > log_d_max:
                        ld = log_d_max
                    log_deltas[j] = ld
                    deltas_arr[j] = np.exp(ld)

            for j in range(M):
                slot_accepted[j] = 0.0
        
        if quench_every is not None and (i + 1) % quench_every == 0:
            for j in range(min(3, M)):
                r = replica_at[j]
                q_coords = local_optimisor(coords_lst[r].copy())
                q_energy = float(cluster_potential(q_coords))
                if q_energy < best_energy:
                    best_energy = q_energy
                    best_coords = q_coords
    print("Temperatures at end", temps_arr)
    return best_energy, best_coords


def parallel_tempering_run(start_coords, start_energies, M, temps, deltas, R_c, swap_every, quench_every, N_sweeps, adapt_fraction, potential_func, local_optimisor):
    """
    Runs a parallel tempering Monte Carlo simulation across M temperature replicas
    to search for the global minimum potential energy.

    :param start_coords: array_like, shape (M, N, 3),
        Initial 3D coordinates of N particles for each of the M replicas.
    :param start_energies: array_like, shape (M),
        Initial potential energies corresponding to start_coords for each replica.
    :param M: int,
        Number of temperature replicas.
    :param temps: array_like, shape (M),
        Temperature ladder ordered from lowest to highest temperature.
    :param deltas: array_like, shape (M),
        Initial maximum single-atom displacement for each MC step for each temperature slot.
    :param R_c: float,
        Maximum radius of the cluster.
    :param swap_every: int,
        Frequency at which adjacent temperature swaps are attempted.
    :param quench_every: int or None,
        Frequency at which low temperatures replicas are quenched to check what basin they are in. 
        None if there should be no quenching attempts.
    :param N_sweeps: int,
        Total number of Monte Carlo sweeps to perform per replica.
    :param adapt_fraction: float,
        After N_sweeps*adapt_fraction the temperatures will no longer be updated dynamically. In some literature this is used 
        to freeze the temperatures so that certain data reliant on fixed temperatures can be read after an initial 
        adjustment phase of the system.
    :param potential_func: callable,
        Function to calculate the potential between two particles.
    :param local_optimisor: callable,
        Function to locally optimise the current state, for quenching.
    :returns: tuple of (float, np.ndarray)
        - best_energy -- Lowest potential energy found across all replicas during the simulation.
        - best_coords -- Coordinates corresponding to best_energy, shape (N, 3).
    """
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
    temps_arr = temps_arr.astype(float, copy=True)
    if temps_arr.ndim != 1 or temps_arr.shape[0] != M:
        raise ValueError("temps must be a 1D array with shape (M)")
    if not np.all(np.isfinite(temps_arr)) or np.any(temps_arr <= 0):
        raise ValueError("temps must contain only finite, positive values")
    if np.any(np.diff(temps_arr) <= 0):
        raise ValueError("temps must be ordered from lowest to highest temperature")

    # Validate deltas
    deltas_arr = np.asarray(deltas)
    if deltas_arr.dtype == bool or not np.issubdtype(deltas_arr.dtype, np.number):
        raise TypeError("deltas must contain numeric values")
    deltas_arr = deltas_arr.astype(float, copy=True)
    if deltas_arr.ndim != 1 or deltas_arr.shape[0] != M:
        raise ValueError("deltas must be a 1D array with shape (M)")
    if not np.all(np.isfinite(deltas_arr)) or np.any(deltas_arr <= 0):
        raise ValueError("deltas must contain only finite, positive values")

    # Validate start_coords and protect original input
    coords_lst = np.array(start_coords)
    if coords_lst.dtype == bool or not np.issubdtype(coords_lst.dtype, np.number):
        raise TypeError("start_coords must contain numeric values")
    coords_lst = coords_lst.astype(float, copy=True)
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
    energies_lst = energies_lst.astype(float, copy=True)
    if energies_lst.ndim != 1 or energies_lst.shape[0] != M:
        raise ValueError("start_energies must be a 1D array with shape (M,)")
    if not np.all(np.isfinite(energies_lst)):
        raise ValueError("start_energies must contain only finite values")


    best_energy, best_coords = _pt_loop_njit(
            coords_lst,
            energies_lst,
            M,
            temps_arr,
            deltas_arr,
            float(R_c),
            swap_every,
            quench_every,
            N_sweeps,
            potential_func,
            local_optimisor,
            adapt_fraction
        )
    return float(best_energy), best_coords


def make_deltas(temps, d_ref, T_ref):
    return d_ref * np.sqrt(np.asarray(temps) / T_ref)


def random_cluster(N, R_start, min_dist=0.8, max_tries=10000):
    pos = np.empty((N, 3))
    n = 0
    tries = 0
    while n < N:
        p = np.random.uniform(-R_start, R_start, 3)
        if np.linalg.norm(p) <= R_start and (n == 0 or np.all(np.sum((pos[:n] - p) ** 2, axis=1) > min_dist**2)):
            pos[n] = p
            n += 1
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Could not place atoms; increase R_start or lower min_dist")
    return pos


def make_start(N, M, R_c, cluster_pot_func):
    R_start = R_c - 0.5
    start_coords = np.array([random_cluster(N, R_start) for _ in range(M)])
    start_energies = np.array([cluster_pot_func(c) for c in start_coords])
    return start_energies, start_coords


if __name__ == "__main__":
    np.random.seed(42)

    M = 10
    N = 13
    R_c = 3.0
    swap_every = 3
    quench_every = None
    N_sweeps = 50000
    temps = np.geomspace(0.04, 0.48, M)
    deltas = make_deltas(temps, 0.08, temps[0])
    energies, coords = make_start(N, M, R_c, cluster_potential)
    adapt_fraction = 1.0

    t_start = time.perf_counter()
    best_e, best_c = parallel_tempering_run(
        coords, energies, M, temps, deltas, R_c, swap_every, quench_every, N_sweeps, adapt_fraction, lennard_jones_potential_njit, None
    )
    t_end = time.perf_counter()

    print(f"PT runtime: {t_end - t_start:.4f} s")

    best_c_q = FIRE(
        best_c,
        lennard_jones_derivative,
        Nmax=20_000,
        Nmin=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha0=0.1,
        f_alpha=0.99,
        err_tol=1e-5,
        dt=0.002,
        max_step=0.05,
    )
    best_e_q = cluster_potential(best_c_q)
    print("Quenched global energy minimum:", best_e_q, "for:", N, "atoms.")
    highscores(best_c_q, "parallel_tempering", "LJ", (1, 1), "First attempt of PT using numba")
    plot_cluster(best_c_q)