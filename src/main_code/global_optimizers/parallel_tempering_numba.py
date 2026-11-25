import time
import numpy as np
from numba import njit, objmode
from src.main_code.potentials_calculations import (
    lennard_jones_potential,
    cluster_potential,
    lennard_jones_derivative,
    lennard_jones_potential_njit,
    lennard_jones_derivative_njit,
    single_atom_potential_njit,
    cluster_potential_njit
)
from src.main_code.monte_carlo_sweep import mc_sweep, mc_sweep_njit
from src.main_code.local_optimizers.fire import FIRE
from src.main_code.local_optimizers.bfgs import LBFGS, lbfgs_njit
from src.main_code.visualization import plot_cluster
from data.data_handling import highscores, read_LJwales_data


@njit
def _check_interrupt(): #Lets you exit njit run with ctrl c
    with objmode():
        pass  # Entering objmode acquires the GIL and triggers any pending KeyboardInterrupt

@njit
def attempt_temp_swaps_njit(M, replica_at, betas, energies, attempt_count):
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
def update_temp_ladder_njit(
    rho: np.ndarray,
    energies: np.ndarray,
    replica_at: np.ndarray,
    betas: np.ndarray,
    T_min: float,
    T_max: float,
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
    betas_next[0] = 1.0 / (T_min)

    span = T_max - T_min
    cum_w = 0.0
    for t in range(n_intervals - 1):
        cum_w += weights[t] / sum_w
        t_val = T_min + span * cum_w
        temps_next[t + 1] = t_val
        betas_next[t + 1] = 1.0 / (t_val)

    # Pin the top endpoint identically to T_max (avoids floating-point drift)
    temps_next[M - 1] = T_max
    betas_next[M - 1] = 1.0 / (T_max)

    return rho_next, temps_next, betas_next


@njit
def _pt_loop_njit(
    coords_lst,
    energies_lst,
    best_coords_out,
    wales_energy,
    M,
    temps_arr,
    rho_bounds,
    deltas_arr,
    d_min,
    d_max,
    R_c,
    swap_every,
    target_accept_rate,
    gamma_scale,
    gamma_offset,
    gamma_decay,
    quench_every,
    quench_lowest_n,
    N_sweeps,
    print_progress,
    adapt_fraction,
    potential_func_njit,
    potential_derivative_njit,
    local_optimisor_njit,
):
    """
    Parallel Tempering inner loop using numba.
    """
    T_min = temps_arr[0]
    T_max = temps_arr[-1]
    rho = np.log(np.diff(temps_arr) / (T_max - T_min))
    replica_at = np.arange(M)
    betas = 1.0 /temps_arr

    n_atoms = coords_lst.shape[1]
    n_moves = n_atoms
    attempt_count = 0

    log_deltas = np.log(deltas_arr)
    log_d_min = np.log(d_min)
    log_d_max = np.log(d_max)
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
    best_coords_out[:] = best_coords

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
                best_coords_out[:] = best_coords

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

            attempt_temp_swaps_njit(M, replica_at, betas, energies_lst, attempt_count)
            attempt_count += 1

            if i < adapt_cutoff:
                gamma_n = gamma_scale / ((attempt_count + gamma_offset) ** gamma_decay)
                rho, temps_arr, betas = update_temp_ladder_njit(
                rho, energies_lst, replica_at, betas, T_min, T_max, gamma_n, rho_bounds
                )

                for j in range(M):
                    acc_rate = slot_accepted[j] / moves_per_window
                    ld = log_deltas[j] + gamma_n * (acc_rate - target_accept_rate)
                    if ld < log_d_min:
                        ld = log_d_min
                    elif ld > log_d_max:
                        ld = log_d_max
                    log_deltas[j] = ld
                    deltas_arr[j] = np.exp(ld)

            for j in range(M):
                slot_accepted[j] = 0.0
        
        if quench_every is not None and (i + 1) % quench_every == 0:
            for j in range(min(quench_lowest_n, M)):
                r = replica_at[j]
                q_coords, q_energy, _, _ = local_optimisor_njit(coords_lst[r].copy(), potential_derivative_njit,
                    potential_func_njit)
                if q_energy < best_energy:
                    best_energy = q_energy
                    best_coords = q_coords
                    best_coords_out[:] = best_coords
        if (i + 1) % 100 == 0:
            _check_interrupt()
            if print_progress:
                pct = int(round(100.0 * (i + 1) / N_sweeps))
                print("Progress:", pct, "% | Sweep:", i + 1, "/", N_sweeps, "| Best energy:", best_energy, "| Wales LJ energy:", wales_energy)
    print("Temperatures at end", temps_arr)
    #Quench:
    final_coords, final_energy, _, _ = local_optimisor_njit(best_coords, potential_derivative_njit, potential_func_njit)
    if final_energy < best_energy:
        best_energy = final_energy
        best_coords = final_coords
    return best_energy, best_coords


def parallel_tempering_run(
    start_coords,
    start_energies,
    M,
    temps,
    rho_bounds,
    deltas,
    d_min,
    d_max,
    R_c,
    swap_every,
    target_accept_rate,
    gamma_scale,
    gamma_offset,
    gamma_decay,
    quench_every,
    quench_lowest_n,
    N_sweeps,
    print_progress,
    adapt_fraction,
    potential_func_njit,
    potential_derivative_njit,
    local_optimisor_njit,
):
    """
    Runs a parallel tempering Monte Carlo simulation across M temperature replicas
    to search for the global minimum potential energy.

    :param start_coords: array_like, shape (M, N, 3),
        Initial 3D coordinates of N particles for each of the M replicas.
    :param start_energies: array_like, shape (M,),
        Initial potential energies corresponding to start_coords for each replica.
    :param M: int,
        Number of temperature replicas.
    :param temps: array_like, shape (M,),
        Initial temperature ladder ordered strictly from lowest to highest temperature.
    :param rho_bounds: tuple of (float, float),
        Lower and upper bounds (rho_min, rho_max) on the log temperature spacings rho_t.
    :param deltas: array_like, shape (M,),
        Initial maximum single-atom displacement for each MC step for each temperature slot.
    :param d_min: float,
        Minimum allowed single-atom displacement step size during adaptation.
    :param d_max: float,
        Maximum allowed single-atom displacement step size during adaptation.
    :param R_c: float,
        Maximum confining sphere radius of the cluster.
    :param swap_every: int,
        Frequency (in sweeps) at which adjacent temperature swaps and adaptations are attempted.
    :param target_accept_rate: float,
        Target Monte Carlo step acceptance probability in (0, 1) for Robbins-Monro step-size adaptation.
    :param gamma_scale: float,
        Initial Robbins-Monro gain 'a' in gamma_n = a / (n + offset)^decay.
    :param gamma_offset: float,
        Stabilization offset 'n_0' in the Robbins-Monro step-size schedule.
    :param gamma_decay: float,
        Decay exponent 'kappa' in (0.5, 1.0] in the Robbins-Monro step-size schedule.
    :param quench_every: int or None,
        Frequency (in sweeps) at which low-temperature replicas are quenched via local optimization.
        None disables periodic in-loop quenching.
    :param quench_lowest_n: int,
        Number of lowest-temperature slots to quench at each periodic quenching step.
    :param N_sweeps: int,
        Total number of Monte Carlo sweeps to perform per replica.
    :param adapt_fraction: float,
        Fraction in [0, 1] of N_sweeps during which temperatures and step sizes are adapted
        before freezing for the remainder of the run.
    :param potential_func_njit: callable,
        Numba-compiled scalar pair potential function V(r).
    :param potential_derivative_njit: callable,
        Numba-compiled scalar pair potential derivative dV/dr.
    :param local_optimisor_njit: callable,
        Numba-compiled local optimizer used for quenching replicas.
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

    # Validate d_min and d_max
    if isinstance(d_min, (bool, np.bool_)) or not isinstance(d_min, (int, float, np.integer, np.floating)):
        raise TypeError("d_min must be a numeric value")
    if not np.isfinite(d_min) or d_min <= 0:
        raise ValueError("d_min must be finite and positive")

    if isinstance(d_max, (bool, np.bool_)) or not isinstance(d_max, (int, float, np.integer, np.floating)):
        raise TypeError("d_max must be a numeric value")
    if not np.isfinite(d_max) or d_max < d_min:
        raise ValueError("d_max must be finite and greater than or equal to d_min")

    # Validate swap_every
    if isinstance(swap_every, (bool, np.bool_)) or not isinstance(swap_every, (int, np.integer)):
        raise TypeError("swap_every must be an integer")
    if swap_every <= 0:
        raise ValueError("swap_every must be a positive integer")

    # Validate target_accept_rate
    if isinstance(target_accept_rate, (bool, np.bool_)) or not isinstance(target_accept_rate, (int, float, np.integer, np.floating)):
        raise TypeError("target_accept_rate must be a numeric value")
    if not np.isfinite(target_accept_rate) or not (0.0 < target_accept_rate < 1.0):
        raise ValueError("target_accept_rate must be strictly between 0 and 1")

    # Validate Robbins-Monro parameters
    if isinstance(gamma_scale, (bool, np.bool_)) or not isinstance(gamma_scale, (int, float, np.integer, np.floating)):
        raise TypeError("gamma_scale must be a numeric value")
    if not np.isfinite(gamma_scale) or gamma_scale <= 0:
        raise ValueError("gamma_scale must be finite and positive")

    if isinstance(gamma_offset, (bool, np.bool_)) or not isinstance(gamma_offset, (int, float, np.integer, np.floating)):
        raise TypeError("gamma_offset must be a numeric value")
    if not np.isfinite(gamma_offset) or gamma_offset < 0:
        raise ValueError("gamma_offset must be finite and non-negative")

    if isinstance(gamma_decay, (bool, np.bool_)) or not isinstance(gamma_decay, (int, float, np.integer, np.floating)):
        raise TypeError("gamma_decay must be a numeric value")
    if not np.isfinite(gamma_decay) or not (0.5 < gamma_decay <= 1.0):
        raise ValueError("gamma_decay must be in the interval (0.5, 1.0]")

    # Validate quench_every and quench_lowest_n
    if quench_every is not None:
        if isinstance(quench_every, (bool, np.bool_)) or not isinstance(quench_every, (int, np.integer)):
            raise TypeError("quench_every must be an integer or None")
        if quench_every <= 0:
            raise ValueError("quench_every must be a positive integer")

    if isinstance(quench_lowest_n, (bool, np.bool_)) or not isinstance(quench_lowest_n, (int, np.integer)):
        raise TypeError("quench_lowest_n must be an integer")
    if quench_lowest_n < 0:
        raise ValueError("quench_lowest_n must be non-negative")

    # Validate N_sweeps
    if isinstance(N_sweeps, (bool, np.bool_)) or not isinstance(N_sweeps, (int, np.integer)):
        raise TypeError("N_sweeps must be an integer")
    if N_sweeps <= 0:
        raise ValueError("N_sweeps must be a positive integer")

    # Validate adapt_fraction
    if isinstance(adapt_fraction, (bool, np.bool_)) or not isinstance(adapt_fraction, (int, float, np.integer, np.floating)):
        raise TypeError("adapt_fraction must be a numeric value")
    if not np.isfinite(adapt_fraction) or not (0.0 <= adapt_fraction <= 1.0):
        raise ValueError("adapt_fraction must be between 0.0 and 1.0 inclusive")

    # Validate rho_bounds
    if not isinstance(rho_bounds, (tuple, list)) or len(rho_bounds) != 2:
        raise TypeError("rho_bounds must be a tuple or list of length 2")
    rho_min, rho_max = rho_bounds
    if isinstance(rho_min, (bool, np.bool_)) or not isinstance(rho_min, (int, float, np.integer, np.floating)):
        raise TypeError("rho_bounds elements must be numeric values")
    if isinstance(rho_max, (bool, np.bool_)) or not isinstance(rho_max, (int, float, np.integer, np.floating)):
        raise TypeError("rho_bounds elements must be numeric values")
    if not (np.isfinite(rho_min) and np.isfinite(rho_max)) or rho_min >= rho_max:
        raise ValueError("rho_bounds must satisfy -inf < rho_min < rho_max < inf")
    rho_bounds_tuple = (float(rho_min), float(rho_max))

    # Validate callables
    if not callable(potential_func_njit):
        raise TypeError("potential_func_njit must be callable")
    if not callable(potential_derivative_njit):
        raise TypeError("potential_derivative_njit must be callable")
    if not callable(local_optimisor_njit):
        raise TypeError("local_optimisor_njit must be callable")

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

    wales_energy = cluster_potential(read_LJwales_data(len(coords_lst[0])), lennard_jones_potential)
    best_coords_out = np.empty_like(coords_lst[0])
    try:
        best_energy, best_coords = _pt_loop_njit(
                coords_lst,
                energies_lst,
                best_coords_out,
                wales_energy,
                int(M),
                temps_arr,
                rho_bounds_tuple,
                deltas_arr,
                float(d_min),
                float(d_max),
                float(R_c),
                int(swap_every),
                float(target_accept_rate),
                float(gamma_scale),
                float(gamma_offset),
                float(gamma_decay),
                None if quench_every is None else int(quench_every),
                int(quench_lowest_n),
                int(N_sweeps),
                print_progress,
                float(adapt_fraction),
                potential_func_njit,
                potential_derivative_njit,
                local_optimisor_njit,
            )
    except KeyboardInterrupt:
        print("\nInterrupted! Returning best coordinates found so far.")
        best_coords, best_energy, _, _ = local_optimisor_njit(
            best_coords_out, potential_derivative_njit, potential_func_njit
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

    N = 78                                                  # Number of atoms in the cluster 
    M = max(6, int(round(3.0 * np.sqrt(N))))                # Number of replicas; scales as ~3*sqrt(N) to keep adjacent energy overlap constant
    R_c = round(1.3 * (N ** (1.0 / 3.0)), 2)                # Confining sphere radius (~1.3 * N^(1/3)): prevents evaporation without pinning surface atoms
    swap_every = 3                                          # Sweeps between swap attempts, should be usually between 1 and 20
    quench_every = 40 * N                                   # Sweeps between periodic L-BFGS quenches (~30*N to 50*N so quenching stays cheap). 
    quench_lowest_n = max(2, int(round(0.35 * M)))          # Lowest temperature slots to quench (~30-40% of M, targeting replicas near/below melting)
    N_sweeps = 200*(N ** 2)                               # Total MC sweeps per replica (~N^2 for easy icosahedral sizes; increase 5-10x for multi-funnel N=38, 75)
    #T_min and T_max in the line below are VERY IMPORTANT, because they are fixed, even with iterative temp updates
    temps = np.geomspace(0.04, 0.50, M)                     # Initial geometric ladder: T_min in deep solid (0.02-0.05), T_max above melting (~0.45-0.60)
    deltas = make_deltas(temps, 0.08, temps[0])             # Initial max displacements scaled via equipartition d_ref * sqrt(T / T_min), with d_ref ~ 0.05-0.10
    d_min = 1e-4                                            # Lower bound on MC step size to prevent collapse if trapped at T_min
    d_max = R_c                                             # Upper bound on MC step size (moves larger than R_c always hit the confining wall)
    energies, coords = make_start(N, M, R_c, cluster_potential)  # Initial non-overlapping random cluster configurations and energies inside R_c - 0.5
    adapt_fraction = 1.0                                    # Fraction of sweeps with active temperature adaptation (1.0 for global optimization; 0.2-0.5 if sampling equilibrium)

    rho_bounds = (-10.0, 10.0)                                # Min/max bounds on log temperature spacings rho_t to prevent ladder collapse, (-10, 10) is pretty much always fine
    target_accept_rate = 0.234                              # Optimal multi-dimensional random-walk Metropolis acceptance rate (Roberts-Gelman-Gilks). 0.234 is found from literature.
    gamma_scale = 0.5                                       # Initial Robbins-Monro gain 'a' in gamma_n = a / (n + offset)^decay, gamma_n is the step size for the temperature updates.
    gamma_offset = 50.0                                     # Robbins-Monro offset to damp noisy early ladder/step-size jumps
    gamma_decay = 0.6                                       # Robbins-Monro decay exponent in (0.5, 1.0] guaranteeing asymptotic convergence
    print_progress = True

    t_start = time.perf_counter()
    best_e, best_c = parallel_tempering_run(
        coords, energies, M, temps, rho_bounds, deltas, d_min, d_max, R_c, 
        swap_every, target_accept_rate, gamma_scale, gamma_offset, gamma_decay, 
        quench_every, quench_lowest_n, N_sweeps, print_progress, adapt_fraction, 
        lennard_jones_potential_njit, lennard_jones_derivative_njit, lbfgs_njit
    )
    t_end = time.perf_counter()

    print(f"\nPT runtime: {t_end - t_start:.4f} s")
    print("Quenched global energy minimum:", best_e, "for:", N, "atoms.")
    print("Wales databas energy minimum:", cluster_potential(read_LJwales_data(N), lennard_jones_potential))
    highscores(best_c, "parallel_tempering", "LJ", (1, 1), "First attempt of PT using numba")
    plot_cluster(best_c)