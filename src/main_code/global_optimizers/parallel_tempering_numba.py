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
def _check_interrupt():  # Lets you exit njit run with ctrl c
    with objmode():
        pass  # Entering objmode acquires the GIL and triggers any pending KeyboardInterrupt


@njit
def attempt_temp_swaps_njit(
    M,
    replica_at,
    betas,
    energies,
    attempt_count,
    track_diagnostics,
    swap_attempts,
    swap_accepts,
):
    """
    Attempts to swap configurations between adjacent temperatures using the parallel tempering Metropolis criterion.
    Modifies replica_at in place and optionally records per-interval swap statistics.
    """
    start = attempt_count % 2
    for t in range(start, M - 1, 2):
        r_i, r_j = replica_at[t], replica_at[t + 1]
        if track_diagnostics:
            swap_attempts[t] += 1
        delta = (betas[t] - betas[t + 1]) * (energies[r_i] - energies[r_j])
        if delta >= 0.0 or np.random.rand() < np.exp(delta):
            replica_at[t], replica_at[t + 1] = r_j, r_i
            if track_diagnostics:
                swap_accepts[t] += 1


@njit
def update_temp_ladder_njit(
    rho: np.ndarray,
    energies: np.ndarray,
    replica_at: np.ndarray,
    temps: np.ndarray,
    betas: np.ndarray,
    T_min: float,
    T_max: float,
    step_size: float,
    rho_bounds: tuple = (-6.0, 6.0),
):
    """
    Updates interior temperatures on a fixed [T_min, T_max] interval via
    Robbins-Monro equal-acceptance-rate adaptation (Atchadé et al. / Vousden et al.).
    Updates temps and betas in place.
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
    max_rho = rho_bounds[0]
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

    temps[0] = T_min
    betas[0] = 1.0 / T_min

    span = T_max - T_min
    cum_w = 0.0
    for t in range(n_intervals - 1):
        cum_w += weights[t] / sum_w
        t_val = T_min + span * cum_w
        temps[t + 1] = t_val
        betas[t + 1] = 1.0 / t_val

    temps[M - 1] = T_max
    betas[M - 1] = 1.0 / T_max

    return rho_next


@njit
def update_temp_ladder_feedback_njit(
    n_up: np.ndarray,
    n_down: np.ndarray,
    temps: np.ndarray,
    betas: np.ndarray,
    T_min: float,
    T_max: float,
    damping: float = 0.5,
):
    """
    Updates interior temperatures on a fixed [T_min, T_max] interval via
    Katzgraber-Trebst-Huse-Troyer (2006) feedback optimization to maximize
    round-trip replica flux across phase transitions.
    Updates temps and betas in place. Returns True if update succeeded, False if data was insufficient.
    (https://doi.org/10.48550/arXiv.cond-mat/0602085)
    """
    M = temps.shape[0]
    n_intervals = M - 1

    # Calculate f as in equation 3 of the paper
    f = np.empty(M, dtype=np.float64)
    f[0] = 1.0
    f[M - 1] = 0.0
    total_labeled = 0.0
    for j in range(1, M - 1):
        tot = n_up[j] + n_down[j]
        total_labeled += tot
        f[j] = n_up[j] / tot

    if total_labeled < float(M):
        return False

    # Added enforcing monotonic non-increasing f[0] >= f[1] >= ... >= f[M-1], since the noise in the approxmation might break this. This is not needed in the papers theoretical approach.
    blocks_val = np.empty(M, dtype=np.float64)
    blocks_len = np.empty(M, dtype=np.int64)
    n_blocks = 0
    for j in range(M):
        blocks_val[n_blocks] = f[j]
        blocks_len[n_blocks] = 1
        n_blocks += 1
        while n_blocks > 1 and blocks_val[n_blocks - 2] < blocks_val[n_blocks - 1]:
            w1 = float(blocks_len[n_blocks - 2])
            w2 = float(blocks_len[n_blocks - 1])
            blocks_val[n_blocks - 2] = (w1 * blocks_val[n_blocks - 2] + w2 * blocks_val[n_blocks - 1]) / (w1 + w2)
            blocks_len[n_blocks - 2] += blocks_len[n_blocks - 1]
            n_blocks -= 1

    idx = 0
    for b in range(n_blocks):
        for _ in range(blocks_len[b]):
            f[idx] = blocks_val[b]
            idx += 1
    f[0] = 1.0
    f[M - 1] = 0.0

    # Eta as in equation 8 and 10. We use the log of the temperatures in stead of temps directly as the spacing is a bit different
    log_temps = np.log(temps)
    eta = np.empty(n_intervals, dtype=np.float64)
    cum_eta = np.empty(M, dtype=np.float64)
    cum_eta[0] = 0.0
    eps_floor = 0.05 / float(n_intervals)
    for t in range(n_intervals):
        d_log_t = log_temps[t + 1] - log_temps[t]
        df = f[t] - f[t + 1]
        if df < eps_floor:
            df = eps_floor
        eta[t] = np.sqrt(df) / d_log_t
        cum_eta[t + 1] = cum_eta[t] + eta[t] * d_log_t

    total_integral = cum_eta[M - 1]
    if total_integral <= 0.0:
        return False

    # Invert cumulative integral to place interior temperatures at equal flux-weight quantiles
    new_temps = np.empty(M, dtype=np.float64)
    new_temps[0] = T_min
    new_temps[M - 1] = T_max

    seg = 0
    for j in range(1, M - 1):
        target_c = total_integral * (float(j) / float(n_intervals))
        while seg < n_intervals - 1 and cum_eta[seg + 1] < target_c:
            seg += 1
        denom = cum_eta[seg + 1] - cum_eta[seg]
        frac = (target_c - cum_eta[seg]) / denom if denom > 0.0 else 0.0
        target_log_t = log_temps[seg] + frac * (log_temps[seg + 1] - log_temps[seg])
        # Damp update in log-T space for stability across epochs
        blended_log_t = (1.0 - damping) * log_temps[j] + damping * target_log_t
        new_temps[j] = np.exp(blended_log_t)

    # Guarantee strict monotonicity and update temps / betas in place
    temps[0] = T_min
    betas[0] = 1.0 / T_min
    for j in range(1, M - 1):
        if new_temps[j] <= temps[j - 1]:
            new_temps[j] = temps[j - 1] * 1.0001
        temps[j] = new_temps[j]
        betas[j] = 1.0 / new_temps[j]
    temps[M - 1] = T_max
    betas[M - 1] = 1.0 / T_max

    return True


@njit
def update_temp_ladder_feedback_exact_njit(
    n_up: np.ndarray,
    n_down: np.ndarray,
    temps: np.ndarray,
    betas: np.ndarray,
    T_min: float,
    T_max: float,
):
    """
    Exact linear-T implementation of Katzgraber, Trebst, Huse, & Troyer (2006),
    J. Stat. Mech. P03018:
      1. f(T_i) = n_up(T_i) / (n_up(T_i) + n_down(T_i)),  f(T_0) = 1, f(T_{M-1}) = 0
      2. eta'(T) = C' * sqrt( (1 / dT) * |df / dT| )
      3. Integral_{T_min}^{T'_k} eta'(T) dT = k / (M - 1)
    Updates temps and betas in place. Returns True if update succeeded.
    """
    M = temps.shape[0]
    n_intervals = M - 1

    # 1. Up-moving replica fraction f(T_j) with boundary conditions f(T_min)=1, f(T_max)=0
    f = np.empty(M, dtype=np.float64)
    f[0] = 1.0
    f[M - 1] = 0.0
    total_labeled = 0.0
    for j in range(1, M - 1):
        tot = n_up[j] + n_down[j]
        total_labeled += tot
        f[j] = n_up[j] / tot

    if total_labeled < float(M):
        return False

    # 2. Estimate eta'(T) = C' * sqrt( (1 / dT) * |df / dT| ) in linear T
    #    With df_dT = |f[t+1] - f[t]| / dT, eta_unnorm[t] = sqrt(df_dT / dT) = sqrt(|df|) / dT
    cum_eta = np.empty(M, dtype=np.float64)
    cum_eta[0] = 0.0
    for t in range(n_intervals):
        d_t = temps[t + 1] - temps[t]
        abs_df = abs(f[t + 1] - f[t])
        if abs_df < 1e-12:
            abs_df = 1e-12
        df_dT = abs_df / d_t
        eta_t = np.sqrt(df_dT / d_t)
        cum_eta[t + 1] = cum_eta[t] + eta_t * d_t

    total_integral = cum_eta[M - 1]
    if total_integral <= 0.0:
        return False

    # 3. Solve Integral_{T_min}^{T'_j} eta'(T) dT = j / (M - 1) directly (no damping)
    new_temps = np.empty(M, dtype=np.float64)
    new_temps[0] = T_min
    new_temps[M - 1] = T_max

    seg = 0
    for j in range(1, M - 1):
        target_c = total_integral * (float(j) / float(n_intervals))
        while seg < n_intervals - 1 and cum_eta[seg + 1] < target_c:
            seg += 1
        denom = cum_eta[seg + 1] - cum_eta[seg]
        frac = (target_c - cum_eta[seg]) / denom if denom > 0.0 else 0.0
        new_temps[j] = temps[seg] + frac * (temps[seg + 1] - temps[seg])

    # Write updated temperatures and inverse temperatures in place
    temps[0] = T_min
    betas[0] = 1.0 / T_min
    for j in range(1, M - 1):
        if new_temps[j] <= temps[j - 1]:
            new_temps[j] = temps[j - 1] + 1e-12
        temps[j] = new_temps[j]
        betas[j] = 1.0 / new_temps[j]
    temps[M - 1] = T_max
    betas[M - 1] = 1.0 / T_max

    return True


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
    print_diagnostics,
    diag_swap_attempts,
    diag_swap_accepts,
    diag_half_trips,
    diag_mc_accepted,
    diag_mc_moves,
    diag_max_r_stats,
    diag_best_info,
    adapt_fraction,
    use_feedback_update,
    feedback_damping,
    use_feedback_update_exact,
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
    betas = 1.0 / temps_arr

    n_atoms = coords_lst.shape[1]
    n_moves = n_atoms
    attempt_count = 0

    log_deltas = np.log(deltas_arr)
    log_d_min = np.log(d_min)
    log_d_max = np.log(d_max)
    slot_accepted = np.zeros(M, dtype=np.float64)
    moves_per_window = float(swap_every * n_moves)
    adapt_cutoff = int(adapt_fraction * N_sweeps)
    diag_start = 0

    # Track extreme visits whenever diagnostics OR feedback temperature adaptation is enabled
    any_feedback = use_feedback_update or use_feedback_update_exact
    track_extremes = print_diagnostics or any_feedback
    extreme_state = np.zeros(M, dtype=np.int64)  # -1 = last visited T_min, +1 = last visited T_max
    fb_n_up = np.zeros(M, dtype=np.float64)
    fb_n_down = np.zeros(M, dtype=np.float64)

    # Exponentially growing epoch schedule for feedback optimization (e.g. 10%, 25%, 50%, 80%, 100% of adapt_cutoff)
    fb_epoch_len = max(swap_every * 50, adapt_cutoff // 10)
    fb_next_update = fb_epoch_len

    best_idx = 0
    best_energy = energies_lst[0]
    for m in range(1, M):
        if energies_lst[m] < best_energy:
            best_energy = energies_lst[m]
            best_idx = m
    best_coords = coords_lst[best_idx].copy()
    best_coords_out[:] = best_coords

    if track_extremes:
        for m in range(M):
            extreme_state[replica_at[m]] = -1 if m < (M // 2) else 1

    if print_diagnostics:
        diag_best_info[0] = 0.0
        diag_best_info[1] = float(best_idx)
        diag_best_info[2] = temps_arr[best_idx]
        diag_best_info[3] = 0.0  # 0 = initial, 1 = MC sweep, 2 = periodic quench, 3 = final quench

    for i in range(N_sweeps):
        record_diagnostics = print_diagnostics and (i >= diag_start)
        if i == diag_start and print_diagnostics:
            diag_half_trips[:] = 0
            if not any_feedback:
                extreme_state[:] = 0
                extreme_state[replica_at[0]] = -1
                extreme_state[replica_at[M - 1]] = 1

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
            if record_diagnostics:
                diag_mc_accepted[j] += num_accepted
                diag_mc_moves[j] += n_moves
            if low_en < best_energy:
                is_meaningful_improvement = (best_energy - low_en) > 1e-8
                best_energy = low_en
                best_coords = low_en_coords.copy()
                best_coords_out[:] = best_coords
                if print_diagnostics and is_meaningful_improvement:
                    diag_best_info[0] = float(i + 1)
                    diag_best_info[1] = float(j)
                    diag_best_info[2] = temps_arr[j]
                    diag_best_info[3] = 1.0

        if (i + 1) % swap_every == 0:
            for m in range(M):
                cx = 0.0
                cy = 0.0
                cz = 0.0
                for a in range(n_atoms):
                    cx += coords_lst[m, a, 0]
                    cy += coords_lst[m, a, 1]
                    cz += coords_lst[m, a, 2]
                cx /= n_atoms  # Average x pos of the atoms, used to recenter the cluster to (0,0,0)
                cy /= n_atoms
                cz /= n_atoms
                for a in range(n_atoms):
                    coords_lst[m, a, 0] -= cx
                    coords_lst[m, a, 1] -= cy
                    coords_lst[m, a, 2] -= cz

            if record_diagnostics:
                # 1. Track outermost atom radius at T_max (evaporation / R_c check)
                r_hot = replica_at[M - 1]
                max_rad = 0.0
                for a in range(n_atoms):
                    dx = coords_lst[r_hot, a, 0]
                    dy = coords_lst[r_hot, a, 1]
                    dz = coords_lst[r_hot, a, 2]
                    rad = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if rad > max_rad:
                        max_rad = rad
                diag_max_r_stats[0] += max_rad
                diag_max_r_stats[1] += 1.0
                if max_rad > diag_max_r_stats[2]:
                    diag_max_r_stats[2] = max_rad

            attempt_temp_swaps_njit(
                M,
                replica_at,
                betas,
                energies_lst,
                attempt_count,
                record_diagnostics,
                diag_swap_attempts,
                diag_swap_accepts,
            )
            attempt_count += 1

            if track_extremes:
                r_low = replica_at[0]
                if extreme_state[r_low] == 1:
                    if record_diagnostics:
                        diag_half_trips[r_low] += 1
                    extreme_state[r_low] = -1
                elif extreme_state[r_low] == 0:
                    extreme_state[r_low] = -1

                r_high = replica_at[M - 1]
                if extreme_state[r_high] == -1:
                    if record_diagnostics:
                        diag_half_trips[r_high] += 1
                    extreme_state[r_high] = 1
                elif extreme_state[r_high] == 0:
                    extreme_state[r_high] = 1

            if i < adapt_cutoff:
                gamma_n = gamma_scale / ((attempt_count + gamma_offset) ** gamma_decay)
                if use_feedback_update_exact:
                    for j in range(M):
                        st = extreme_state[replica_at[j]]
                        if st == -1:
                            fb_n_up[j] += 1.0
                        elif st == 1:
                            fb_n_down[j] += 1.0

                    if (i + 1) >= fb_next_update:
                        updated = update_temp_ladder_feedback_exact_njit(
                            fb_n_up, fb_n_down, temps_arr, betas, T_min, T_max
                        )
                        if updated:
                            fb_n_up[:] = 0.0
                            fb_n_down[:] = 0.0
                            fb_epoch_len *= 2
                        fb_next_update = (i + 1) + fb_epoch_len
                elif use_feedback_update:
                    for j in range(M):
                        st = extreme_state[replica_at[j]]
                        if st == -1:
                            fb_n_up[j] += 1.0
                        elif st == 1:
                            fb_n_down[j] += 1.0

                    if (i + 1) >= fb_next_update:
                        updated = update_temp_ladder_feedback_njit(
                            fb_n_up, fb_n_down, temps_arr, betas, T_min, T_max, feedback_damping
                        )
                        if updated:
                            fb_n_up[:] = 0.0
                            fb_n_down[:] = 0.0
                            fb_epoch_len = int(fb_epoch_len * 1.5)
                        fb_next_update = (i + 1) + fb_epoch_len
                else:
                    rho = update_temp_ladder_njit(
                        rho, energies_lst, replica_at, temps_arr, betas, T_min, T_max, gamma_n, rho_bounds
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
                q_coords, q_energy, _, _ = local_optimisor_njit(
                    coords_lst[r].copy(), potential_derivative_njit, potential_func_njit
                )
                if q_energy < best_energy:
                    is_meaningful_improvement = (best_energy - q_energy) > 1e-8
                    best_energy = q_energy
                    best_coords = q_coords
                    best_coords_out[:] = best_coords
                    if print_diagnostics and is_meaningful_improvement:
                        diag_best_info[0] = float(i + 1)
                        diag_best_info[1] = float(j)
                        diag_best_info[2] = temps_arr[j]
                        diag_best_info[3] = 2.0

        if (i + 1) % 100 == 0:
            _check_interrupt()
            if print_progress:
                pct = int(round(100.0 * (i + 1) / N_sweeps))
                print(
                    "Progress:",
                    pct,
                    "% | Sweep:",
                    i + 1,
                    "/",
                    N_sweeps,
                    "| Best energy:",
                    best_energy,
                    "| Wales LJ energy:",
                    wales_energy,
                )
    # Final Quench:
    final_coords, final_energy, _, _ = local_optimisor_njit(
        best_coords, potential_derivative_njit, potential_func_njit
    )
    if final_energy < best_energy:
        is_meaningful_improvement = (best_energy - final_energy) > 1e-8
        best_energy = final_energy
        best_coords = final_coords
        best_coords_out[:] = best_coords
        if print_diagnostics and is_meaningful_improvement:
            diag_best_info[0] = float(N_sweeps)
            diag_best_info[3] = 3.0
    return best_energy, best_coords


def _print_pt_diagnostics(
    temps_arr,
    deltas_arr,
    d_min,
    d_max,
    R_c,
    target_accept_rate,
    diag_swap_attempts,
    diag_swap_accepts,
    diag_half_trips,
    diag_mc_accepted,
    diag_mc_moves,
    diag_max_r_stats,
    diag_best_info,
):
    M = len(temps_arr)
    print("\n================ Parallel Tempering Diagnostics ================")
    print("Final temperatures:", np.round(temps_arr, 5))

    # 1. Adjacent swap acceptance rates (tests M & ladder adaptation)
    if M > 1 and np.any(diag_swap_attempts > 0):
        valid = diag_swap_attempts > 0
        swap_rates = np.zeros(M - 1, dtype=float)
        swap_rates[valid] = diag_swap_accepts[valid] / diag_swap_attempts[valid]
        min_idx = int(np.argmin(swap_rates))
        max_idx = int(np.argmax(swap_rates))
        print(
            f"Adjacent swap acceptance: min={swap_rates[min_idx]:.3f} (at T={temps_arr[min_idx]:.4f}->{temps_arr[min_idx+1]:.4f}), "
            f"mean={np.mean(swap_rates):.3f}, max={swap_rates[max_idx]:.3f} (target ~0.20-0.40)"
        )
        print("Swap rates per interval: ", np.round(swap_rates, 3))

    # 2. Round trips (tests ergodic mixing across T_min <-> T_max)
    round_trips = diag_half_trips // 2
    print(
        f"Replica round trips (T_max <-> T_min): total={int(np.sum(round_trips))}, "
        f"min/replica={int(np.min(round_trips))}, max/replica={int(np.max(round_trips))}"
    )

    # 3. MC step acceptance & step-size bounds (tests deltas, d_min, d_max, Robbins-Monro)
    valid_mc = diag_mc_moves > 0
    mc_rates = np.zeros(M, dtype=float)
    mc_rates[valid_mc] = diag_mc_accepted[valid_mc] / diag_mc_moves[valid_mc]
    n_pinned_min = int(np.sum(np.isclose(deltas_arr, d_min, rtol=1e-3)))
    n_pinned_max = int(np.sum(np.isclose(deltas_arr, d_max, rtol=1e-3)))
    print(
        f"MC acceptance rates: min={np.min(mc_rates):.3f}, mean={np.mean(mc_rates):.3f}, "
        f"max={np.max(mc_rates):.3f} (target={target_accept_rate:.3f})"
    )
    print(
        f"Final step sizes delta: min={np.min(deltas_arr):.5f}, max={np.max(deltas_arr):.5f} "
        f"| pinned at d_min: {n_pinned_min}, pinned at d_max: {n_pinned_max}"
    )

    # 4. Confinement & Melting checks (tests T_min, T_max, R_c)
    if diag_max_r_stats[1] > 0:
        mean_max_r = diag_max_r_stats[0] / diag_max_r_stats[1]
        peak_max_r = diag_max_r_stats[2]
        print(
            f"Outermost atom radius at T_max: mean={mean_max_r:.3f}, peak={peak_max_r:.3f} vs R_c={R_c:.3f} "
            f"({100.0 * mean_max_r / R_c:.1f}% of R_c)"
        )

    # 5. Best energy discovery attribution (tests N_sweeps & quench_lowest_n)
    source_map = {
        0: "initial configuration",
        1: "MC sweep",
        2: "periodic quench",
        3: "final quench",
    }
    src_label = source_map.get(int(diag_best_info[3]), "unknown")
    print(
        f"Best energy found at sweep {int(diag_best_info[0])} | slot {int(diag_best_info[1])} "
        f"(T={diag_best_info[2]:.5f}) | via {src_label}"
    )
    print("================================================================\n")


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
    print_diagnostics,
    adapt_fraction,
    potential_func_njit,
    potential_derivative_njit,
    local_optimisor_njit,
    temp_update_method="equal_accept",
    feedback_damping=0.5,
):
    """
    Runs a parallel tempering Monte Carlo simulation across M temperature replicas
    to search for the global minimum potential energy.

    :param temp_update_method: str,
        Temperature adaptation strategy: "equal_accept" (Vousden/Atchadé Robbins-Monro)
        or "feedback" (Katzgraber-Trebst-Huse-Troyer maximum round-trip flux).
    :param feedback_damping: float,
        Blending factor in (0, 1] used per epoch when temp_update_method="feedback".
    """
    # Validate M
    if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)):
        raise TypeError("M must be an integer")
    if M < 2:
        raise ValueError("M must be a positive integer and greater than 1")

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

    # Validate boolean flags
    if not isinstance(print_progress, (bool, np.bool_)):
        raise TypeError("print_progress must be a boolean")
    if not isinstance(print_diagnostics, (bool, np.bool_)):
        raise TypeError("print_diagnostics must be a boolean")

    # Validate adapt_fraction
    if isinstance(adapt_fraction, (bool, np.bool_)) or not isinstance(adapt_fraction, (int, float, np.integer, np.floating)):
        raise TypeError("adapt_fraction must be a numeric value")
    if not np.isfinite(adapt_fraction) or not (0.0 <= adapt_fraction <= 1.0):
        raise ValueError("adapt_fraction must be between 0.0 and 1.0 inclusive")

    # Validate temp_update_method and feedback_damping
    if temp_update_method not in ("equal_accept", "feedback", "feedback_exact"):
        raise ValueError("temp_update_method must be 'equal_accept', 'feedback', or 'feedback_exact'")
    use_feedback_update = (temp_update_method == "feedback")
    use_feedback_update_exact = (temp_update_method == "feedback_exact")

    if isinstance(feedback_damping, (bool, np.bool_)) or not isinstance(feedback_damping, (int, float, np.integer, np.floating)):
        raise TypeError("feedback_damping must be a numeric value")
    if not np.isfinite(feedback_damping) or not (0.0 < feedback_damping <= 1.0):
        raise ValueError("feedback_damping must be in the interval (0.0, 1.0]")

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

    # Pre-allocate diagnostic buffers (tiny 1D arrays; untouched when print_diagnostics=False)
    diag_swap_attempts = np.zeros(max(1, int(M) - 1), dtype=np.int64)
    diag_swap_accepts = np.zeros(max(1, int(M) - 1), dtype=np.int64)
    diag_half_trips = np.zeros(int(M), dtype=np.int64)
    diag_mc_accepted = np.zeros(int(M), dtype=np.float64)
    diag_mc_moves = np.zeros(int(M), dtype=np.float64)
    diag_max_r_stats = np.zeros(3, dtype=np.float64)
    diag_best_info = np.zeros(4, dtype=np.float64)

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
            bool(print_progress),
            bool(print_diagnostics),
            diag_swap_attempts,
            diag_swap_accepts,
            diag_half_trips,
            diag_mc_accepted,
            diag_mc_moves,
            diag_max_r_stats,
            diag_best_info,
            float(adapt_fraction),
            bool(use_feedback_update),
            float(feedback_damping),
            bool(use_feedback_update_exact),
            potential_func_njit,
            potential_derivative_njit,
            local_optimisor_njit,
        )
    except KeyboardInterrupt:
        print("\nInterrupted! Returning best coordinates found so far.")
        best_coords, best_energy, _, _ = local_optimisor_njit(
            best_coords_out, potential_derivative_njit, potential_func_njit
        )

    if print_diagnostics:
        _print_pt_diagnostics(
            temps_arr,
            deltas_arr,
            float(d_min),
            float(d_max),
            float(R_c),
            float(target_accept_rate),
            diag_swap_attempts,
            diag_swap_accepts,
            diag_half_trips,
            diag_mc_accepted,
            diag_mc_moves,
            diag_max_r_stats,
            diag_best_info,
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

    N = 40                                                  # Number of atoms in the cluster
    M = 16                                                  # Number of replicas
    R_c = round(1.3 * (N ** (1.0 / 3.0)), 2)                # Confining sphere radius (~1.3 * N^(1/3))
    swap_every = 4                                          # Sweeps between swap attempts
    quench_every = 40 * N                                   # Sweeps between periodic L-BFGS quenches
    quench_lowest_n = max(2, int(round(0.35 * M)))          # Lowest temperature slots to quench
    N_sweeps = 100 * (N ** 2)                               # Total MC sweeps per replica
    temps = np.geomspace(0.08, 0.40, M)                    # Initial geometric ladder
    deltas = make_deltas(temps, 0.08, temps[0])             # Initial max displacements
    d_min = 1e-4                                            # Lower bound on MC step size
    d_max = R_c                                             # Upper bound on MC step size
    energies, coords = make_start(N, M, R_c, cluster_potential)
    adapt_fraction = 1.0                                    # Adapt for first 50% of sweeps, then measure fixed-ladder diagnostics

    temp_update_method = "feedback_exact"                         # "equal_accept" (Vousden/Atchadé) or "feedback" or "feedback_exact" (Katzgraber et al.)
    feedback_damping = 0.5                                  # Damping factor per feedback epoch in (0, 1]

    rho_bounds = (-10.0, 10.0)
    target_accept_rate = 0.234
    gamma_scale = 0.5
    gamma_offset = 50.0
    gamma_decay = 0.6
    print_progress = True
    print_diagnostics = True

    t_start = time.perf_counter()
    best_e, best_c = parallel_tempering_run(
        coords, energies, M, temps, rho_bounds, deltas, d_min, d_max, R_c,
        swap_every, target_accept_rate, gamma_scale, gamma_offset, gamma_decay,
        quench_every, quench_lowest_n, N_sweeps, print_progress, print_diagnostics, adapt_fraction,
        lennard_jones_potential_njit, lennard_jones_derivative_njit, lbfgs_njit,
        temp_update_method=temp_update_method,
        feedback_damping=feedback_damping,
    )
    t_end = time.perf_counter()

    print(f"\nPT runtime: {t_end - t_start:.4f} s")
    print("Quenched global energy minimum:", best_e, "for:", N, "atoms.")
    print("Wales databas energy minimum:", cluster_potential(read_LJwales_data(N), lennard_jones_potential))
    highscores(best_c, "parallel_tempering", "LJ", (1, 1), "First attempt of PT using numba")
    plot_cluster(best_c)