import numpy as np
import pytest
from src.main_code.potentials_calculations import (
    lennard_jones_potential,
    cluster_potential,
    lennard_jones_derivative,
    lennard_jones_potential_njit,
    lennard_jones_derivative_njit,
)
from src.main_code.global_optimizers.parallel_tempering import (
    attempt_temp_swaps_njit,
    temps_to_rho,
    update_temp_ladder_njit,
    parallel_tempering_run,
    make_deltas,
    make_start,
)
from src.main_code.local_optimizers.bfgs import lbfgs_njit
from src.main_code.local_optimizers.fire import FIRE


def _default_pt_kwargs(M, temps, deltas, R_c=2.5, N_sweeps=10):
    """Helper returning valid default keyword arguments for parallel_tempering_run."""
    return dict(
        M=M,
        temps=temps,
        rho_bounds=(-10.0, 10.0),
        deltas=deltas,
        d_min=1e-4,
        d_max=R_c,
        R_c=R_c,
        swap_every=5,
        target_accept_rate=0.234,
        gamma_scale=0.5,
        gamma_offset=50.0,
        gamma_decay=0.6,
        quench_every=None,
        quench_lowest_n=2,
        N_sweeps=N_sweeps,
        adapt_fraction=1.0,
        potential_func_njit=lennard_jones_potential_njit,
        potential_derivative_njit=lennard_jones_derivative_njit,
        local_optimisor_njit=lbfgs_njit,
    )


def test_pt_invalid_scalar_parameters():
    """Verify TypeError and ValueError for invalid scalar hyperparameters."""
    M = 3
    temps = np.array([0.1, 0.2, 0.3])
    deltas = np.array([0.05, 0.1, 0.15])
    coords = np.zeros((M, 4, 3))
    energies = np.zeros(M)

    # M validation (reject bools, floats, non-positive ints)
    for bad_M, exc in [(True, TypeError), (3.0, TypeError), (0, ValueError), (-2, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["M"] = bad_M
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # R_c validation
    for bad_Rc, exc in [(False, TypeError), (-1.0, ValueError), (0.0, ValueError), (np.inf, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["R_c"] = bad_Rc
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # d_min and d_max validation
    for bad_dmin, exc in [(True, TypeError), (0.0, ValueError), (-1e-3, ValueError), (np.nan, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["d_min"] = bad_dmin
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    for bad_dmax, exc in [("2.5", TypeError), (1e-5, ValueError), (np.inf, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["d_max"] = bad_dmax
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # swap_every validation
    for bad_swap, exc in [(2.5, TypeError), (True, TypeError), (0, ValueError), (-3, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["swap_every"] = bad_swap
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # target_accept_rate validation
    for bad_tar, exc in [(False, TypeError), (0.0, ValueError), (1.0, ValueError), (-0.1, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["target_accept_rate"] = bad_tar
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # Robbins-Monro parameters validation
    for bad_gs, exc in [(True, TypeError), (0.0, ValueError), (-0.5, ValueError), (np.inf, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["gamma_scale"] = bad_gs
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    for bad_go, exc in [(False, TypeError), (-1.0, ValueError), (np.nan, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["gamma_offset"] = bad_go
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    for bad_gd, exc in [(True, TypeError), (0.5, ValueError), (1.1, ValueError), (0.2, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["gamma_decay"] = bad_gd
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # quench_every and quench_lowest_n validation
    for bad_qe, exc in [(10.5, TypeError), (True, TypeError), (0, ValueError), (-10, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["quench_every"] = bad_qe
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    for bad_qln, exc in [(1.5, TypeError), (False, TypeError), (-1, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["quench_lowest_n"] = bad_qln
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # N_sweeps and adapt_fraction validation
    for bad_ns, exc in [(True, TypeError), (10.0, TypeError), (0, ValueError), (-5, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["N_sweeps"] = bad_ns
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    for bad_af, exc in [(True, TypeError), (-0.1, ValueError), (1.5, ValueError), (np.nan, ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["adapt_fraction"] = bad_af
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    # rho_bounds and callable validation
    for bad_rb, exc in [((-6.0,), TypeError), ((True, 6.0), TypeError), ((6.0, -6.0), ValueError), ((-np.inf, 6.0), ValueError)]:
        kw = _default_pt_kwargs(M, temps, deltas)
        kw["rho_bounds"] = bad_rb
        with pytest.raises(exc):
            parallel_tempering_run(coords, energies, **kw)

    for fn_key in ("potential_func_njit", "potential_derivative_njit", "local_optimisor_njit"):
        kw = _default_pt_kwargs(M, temps, deltas)
        kw[fn_key] = "not_callable"
        with pytest.raises(TypeError):
            parallel_tempering_run(coords, energies, **kw)


def test_pt_invalid_array_parameters():
    """Verify shape, ordering, finiteness, and positivity checks for temps, deltas, coords, and energies."""
    M = 3
    valid_temps = np.array([0.1, 0.2, 0.3])
    valid_deltas = np.array([0.05, 0.1, 0.15])
    valid_coords = np.zeros((M, 4, 3))
    valid_energies = np.zeros(M)

    # Unordered, duplicate, non-positive, or wrong-length temperatures
    for bad_temps in [[0.3, 0.2, 0.1], [0.1, 0.2, 0.2], [0.0, 0.2, 0.3], [0.1, 0.2]]:
        kw = _default_pt_kwargs(M, bad_temps, valid_deltas)
        with pytest.raises(ValueError):
            parallel_tempering_run(valid_coords, valid_energies, **kw)

    # Non-positive or mismatched deltas
    for bad_deltas in [[0.05, 0.0, 0.15], [0.05, 0.1]]:
        kw = _default_pt_kwargs(M, valid_temps, bad_deltas)
        with pytest.raises(ValueError):
            parallel_tempering_run(valid_coords, valid_energies, **kw)

    # Invalid start_coords shapes, 0 atoms, or NaNs
    kw = _default_pt_kwargs(M, valid_temps, valid_deltas)
    with pytest.raises(ValueError):
        parallel_tempering_run(np.zeros((4, 3)), valid_energies, **kw)
    with pytest.raises(ValueError):
        parallel_tempering_run(np.zeros((M, 0, 3)), valid_energies, **kw)
    bad_coords = valid_coords.copy()
    bad_coords[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        parallel_tempering_run(bad_coords, valid_energies, **kw)

    # Invalid start_energies shape or NaN
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, np.zeros(M + 1), **kw)
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, [0.0, np.nan, 0.0], **kw)


def test_attempt_temp_swaps_njit_deterministic_and_parity():
    """
    Verify attempt_temp_swaps_njit respects even/odd alternation (attempt_count % 2),
    always accepts favorable swaps (delta >= 0), and keeps replica_at a valid permutation.
    """
    M = 4
    betas = np.array([10.0, 5.0, 2.0, 1.0])
    energies = np.array([0.0, -5.0, -10.0, -15.0])
    replica_at = np.array([0, 1, 2, 3])

    # attempt_count = 0 -> checks even pairs (0, 1) and (2, 3)
    attempt_temp_swaps_njit(M, replica_at, betas, energies, attempt_count=0)
    assert np.array_equal(replica_at, np.array([1, 0, 3, 2])), \
        "Even attempt_count should swap favorable pairs (0,1) and (2,3)."

    # Reset and test attempt_count = 1 -> checks odd pair (1, 2) only
    replica_at = np.array([0, 1, 2, 3])
    attempt_temp_swaps_njit(M, replica_at, betas, energies, attempt_count=1)
    assert np.array_equal(replica_at, np.array([0, 2, 1, 3])), \
        "Odd attempt_count should only check pair (1,2)."
    assert set(replica_at) == {0, 1, 2, 3}, "replica_at must remain an exact permutation of 0..M-1."


def test_attempt_temp_swaps_njit_rejects_huge_uphill():
    """Verify unfavorable swaps with huge negative delta are rejected."""
    M = 2
    betas = np.array([1000.0, 1.0])
    energies = np.array([-50.0, 0.0])
    replica_at = np.array([0, 1])

    attempt_temp_swaps_njit(M, replica_at, betas, energies, attempt_count=0)
    assert np.array_equal(replica_at, np.array([0, 1])), \
        "Extremely unfavorable swap should be rejected."


def test_temps_to_rho_and_update_temp_ladder_njit_invariants():
    """Verify temps_to_rho round-trips and update_temp_ladder_njit preserves [T_min, T_max], ordering, and bounds."""
    temps = np.array([0.05, 0.10, 0.25, 0.50])
    T_min = float(temps[0])
    T_max = float(temps[-1])
    span = T_max - T_min
    betas = 1.0 / temps
    rho = temps_to_rho(temps)

    # Reconstruct temps from initial fixed-endpoint rho
    weights = np.exp(rho)
    reconstructed = T_min + span * np.concatenate(([0.0], np.cumsum(weights) / np.sum(weights)))
    assert np.allclose(reconstructed, temps)

    # Perform an update with permuted replica_at and verify invariants
    energies = np.array([-10.0, -5.0, -2.0, 0.0])
    replica_at = np.array([2, 0, 3, 1])
    rho_next, temps_next, betas_next = update_temp_ladder_njit(
        rho=rho,
        energies=energies,
        replica_at=replica_at,
        betas=betas,
        T_min=T_min,
        T_max=T_max,
        step_size=0.2,
        rho_bounds=(-6.0, 3.0),
    )

    assert temps_next[0] == pytest.approx(T_min), "Lower endpoint T_min must stay fixed."
    assert temps_next[-1] == pytest.approx(T_max), "Upper endpoint T_max must stay fixed."
    assert np.all(np.diff(temps_next) > 0), "Updated temperatures must remain strictly increasing."
    assert np.allclose(betas_next, 1.0 / temps_next), "betas_next must equal 1 / temps_next."
    assert np.all((rho_next >= -6.0) & (rho_next <= 3.0)), "rho_next must stay within rho_bounds."


def test_pt_does_not_mutate_inputs_and_returns_valid_state():
    """Verify input arrays are not modified in-place and returned best_energy matches best_coords."""
    np.random.seed(7)
    M = 4
    N = 5
    R_c = 2.5
    temps = np.geomspace(0.05, 0.35, M)
    deltas = make_deltas(temps, 0.08, temps[0])
    energies, coords = make_start(N, M, R_c, cluster_potential)

    coords_before = coords.copy()
    energies_before = energies.copy()
    temps_before = temps.copy()
    deltas_before = deltas.copy()

    kw = _default_pt_kwargs(M, temps, deltas, R_c=R_c, N_sweeps=50)
    kw["quench_every"] = 25
    best_e, best_c = parallel_tempering_run(coords, energies, **kw)

    # Inputs untouched
    assert np.array_equal(coords, coords_before), "start_coords must not be mutated in-place."
    assert np.array_equal(energies, energies_before), "start_energies must not be mutated in-place."
    assert np.array_equal(temps, temps_before), "temps must not be mutated in-place."
    assert np.array_equal(deltas, deltas_before), "deltas must not be mutated in-place."

    # Output consistency
    assert best_c.shape == (N, 3)
    assert best_e == pytest.approx(cluster_potential(best_c), abs=1e-6), \
        "Returned best_energy must match the potential energy of best_coords."
    assert best_e <= np.min(energies_before), "best_energy must be <= lowest initial energy."


def test_pt_finds_lj6_global_minimum():
    """
    Verify parallel tempering with in-loop quenching locates the global minimum for a 6-atom LJ cluster
    (known Cambridge Cluster Database minimum: -12.712062).
    """
    np.random.seed(42)
    M = 8
    N = 6
    R_c = 2.5
    temps = np.geomspace(0.04, 0.35, M)
    deltas = make_deltas(temps, 0.08, temps[0])
    energies, coords = make_start(N, M, R_c, cluster_potential)

    kw = _default_pt_kwargs(M, temps, deltas, R_c=R_c, N_sweeps=1500)
    kw["swap_every"] = 10
    kw["quench_every"] = 250
    kw["quench_lowest_n"] = 3

    best_e, best_c = parallel_tempering_run(coords, energies, **kw)

    assert best_e == pytest.approx(-12.712062, abs=1e-4), \
        f"Expected PT to reach the LJ_6 global minimum (-12.712062), got {best_e}."
    assert best_e == pytest.approx(cluster_potential(best_c), abs=1e-6)


def test_pt_escape_local_minima_lj7_cluster():
    """
    Verify parallel tempering escapes the capped-octahedron local minimum (-15.935043)
    of a 7-atom LJ cluster and locates the global minimum (-16.505384).
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
    local_min_energy = cluster_potential(local_min_coords)
    assert np.isclose(local_min_energy, -15.935043, atol=1e-2)

    # Initialize all M replicas trapped in that exact local minimum
    M = 8
    R_c = 3.0
    temps = np.geomspace(0.05, 0.45, M)
    deltas = make_deltas(temps, 0.08, temps[0])
    start_coords = np.repeat(local_min_coords[np.newaxis, :, :], M, axis=0)
    start_energies = np.full(M, local_min_energy)

    kw = _default_pt_kwargs(M, temps, deltas, R_c=R_c, N_sweeps=1500)
    kw["swap_every"] = 10
    kw["quench_every"] = 250
    kw["quench_lowest_n"] = 3

    best_e, best_c = parallel_tempering_run(start_coords, start_energies, **kw)

    known_global_min_energy = -16.505384

    assert best_e < local_min_energy
    assert np.isclose(best_e, known_global_min_energy, atol=1e-3)
    assert np.isclose(cluster_potential(best_c), known_global_min_energy, atol=1e-3)