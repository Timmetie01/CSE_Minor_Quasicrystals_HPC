import numpy as np
import pytest
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential, lennard_jones_derivative
from src.main_code.global_optimizers.parallel_tempering import (
    attempt_temp_swaps,
    parallel_tempering_run,
    make_deltas,
    make_start,
)
from src.main_code.local_optimizers.fire import FIRE
def test_pt_invalid_scalar_parameters():
    """Verify TypeError and ValueError for invalid M, R_c, swap_every, and N_sweeps."""
    M = 3
    temps = np.array([0.1, 0.2, 0.3])
    deltas = np.array([0.05, 0.1, 0.15])
    coords = np.zeros((M, 4, 3))
    energies = np.zeros(M)

    # M validation (reject bools, floats, non-positive ints)
    with pytest.raises(TypeError):
        parallel_tempering_run(coords, energies, True, temps, deltas, 2.5, 5, 10)
    with pytest.raises(TypeError):
        parallel_tempering_run(coords, energies, 3.0, temps, deltas, 2.5, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(coords, energies, 0, temps, deltas, 2.5, 5, 10)

    # R_c validation
    with pytest.raises(TypeError):
        parallel_tempering_run(coords, energies, M, temps, deltas, False, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(coords, energies, M, temps, deltas, -1.0, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(coords, energies, M, temps, deltas, np.inf, 5, 10)

    # swap_every validation
    with pytest.raises(TypeError):
        parallel_tempering_run(coords, energies, M, temps, deltas, 2.5, 2.5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(coords, energies, M, temps, deltas, 2.5, 0, 10)

    # N_sweeps validation
    with pytest.raises(TypeError):
        parallel_tempering_run(coords, energies, M, temps, deltas, 2.5, 5, True)
    with pytest.raises(ValueError):
        parallel_tempering_run(coords, energies, M, temps, deltas, 2.5, 5, -5)


def test_pt_invalid_array_parameters():
    """Verify shape, ordering, finiteness, and positivity checks for temps, deltas, coords, and energies."""
    M = 3
    valid_temps = np.array([0.1, 0.2, 0.3])
    valid_deltas = np.array([0.05, 0.1, 0.15])
    valid_coords = np.zeros((M, 4, 3))
    valid_energies = np.zeros(M)

    # Unordered or non-positive temperatures
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, valid_energies, M, [0.3, 0.2, 0.1], valid_deltas, 2.5, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, valid_energies, M, [0.0, 0.2, 0.3], valid_deltas, 2.5, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, valid_energies, M, [0.1, 0.2], valid_deltas, 2.5, 5, 10)

    # Non-positive or mismatched deltas
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, valid_energies, M, valid_temps, [0.05, 0.0, 0.15], 2.5, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, valid_energies, M, valid_temps, [0.05, 0.1], 2.5, 5, 10)

    # Invalid start_coords shapes, 0 atoms, or NaNs
    with pytest.raises(ValueError):
        parallel_tempering_run(np.zeros((4, 3)), valid_energies, M, valid_temps, valid_deltas, 2.5, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(np.zeros((M, 0, 3)), valid_energies, M, valid_temps, valid_deltas, 2.5, 5, 10)
    bad_coords = valid_coords.copy()
    bad_coords[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        parallel_tempering_run(bad_coords, valid_energies, M, valid_temps, valid_deltas, 2.5, 5, 10)

    # Invalid start_energies shape or NaN
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, np.zeros(M + 1), M, valid_temps, valid_deltas, 2.5, 5, 10)
    with pytest.raises(ValueError):
        parallel_tempering_run(valid_coords, [0.0, np.nan, 0.0], M, valid_temps, valid_deltas, 2.5, 5, 10)


def test_attempt_temp_swaps_deterministic_and_parity():
    """
    Verify attempt_temp_swaps respects even/odd alternation (attempt_count % 2),
    always accepts favorable swaps (delta >= 0), and keeps replica_at a valid permutation.
    """
    M = 4
    # T increases with slot index -> betas strictly decrease with slot index (betas[0] > betas[1] > ...)
    betas = np.array([10.0, 5.0, 2.0, 1.0])

    # Set energies so hotter slots hold LOWER energy -> (betas[t] - betas[t+1]) * (E_i - E_j) > 0
    # Wait: betas[t] > betas[t+1], so delta > 0 when E[r_i] > E[r_j] (cold slot has higher energy than hot slot)
    energies = np.array([0.0, -5.0, -10.0, -15.0])
    replica_at = np.array([0, 1, 2, 3])

    # attempt_count = 0 -> checks even pairs (0, 1) and (2, 3)
    attempt_temp_swaps(M, replica_at, betas, energies, attempt_count=0)
    assert np.array_equal(replica_at, np.array([1, 0, 3, 2])), \
        "Even attempt_count should swap favorable pairs (0,1) and (2,3)."

    # Reset and test attempt_count = 1 -> checks odd pair (1, 2) only
    replica_at = np.array([0, 1, 2, 3])
    attempt_temp_swaps(M, replica_at, betas, energies, attempt_count=1)
    assert np.array_equal(replica_at, np.array([0, 2, 1, 3])), \
        "Odd attempt_count should only check pair (1,2)."
    assert set(replica_at) == {0, 1, 2, 3}, "replica_at must remain an exact permutation of 0..M-1."


def test_attempt_temp_swaps_rejects_huge_uphill():
    """Verify unfavorable swaps with huge negative delta are rejected."""
    np.random.seed(0)
    M = 2
    betas = np.array([1000.0, 1.0])
    # Cold replica (0) already has much lower energy (-50) than hot replica (1) (0.0)
    # delta = (1000 - 1) * (-50 - 0) = -49950 -> exp(-49950) == 0.0
    energies = np.array([-50.0, 0.0])
    replica_at = np.array([0, 1])

    attempt_temp_swaps(M, replica_at, betas, energies, attempt_count=0)
    assert np.array_equal(replica_at, np.array([0, 1])), \
        "Extremely unfavorable swap should be rejected."


def test_pt_does_not_mutate_inputs_and_returns_valid_state():
    """Verify input arrays are not modified in-place and returned best_energy matches best_coords."""
    np.random.seed(7)
    M = 4
    N = 5
    R_c = 2.5
    temps = np.geomspace(0.05, 0.35, M)
    deltas = make_deltas(temps)
    energies, coords = make_start(N, M, R_c)

    coords_before = coords.copy()
    energies_before = energies.copy()

    best_e, best_c = parallel_tempering_run(
        coords, energies, M, temps, deltas, R_c, swap_every=5, N_sweeps=50
    )

    # Inputs untouched
    assert np.array_equal(coords, coords_before), "start_coords must not be mutated in-place."
    assert np.array_equal(energies, energies_before), "start_energies must not be mutated in-place."

    # Output consistency
    assert best_c.shape == (N, 3)
    assert np.all(np.linalg.norm(best_c, axis=1) <= R_c + 1e-12), "All atoms in best_c must stay within R_c."
    assert best_e == pytest.approx(cluster_potential(best_c, lennard_jones_potential)), \
        "Returned best_energy must match the potential energy of best_coords."
    assert best_e <= np.min(energies_before), "best_energy must be <= lowest initial energy."


def test_pt_finds_lj6_global_minimum():
    """
    Verify parallel tempering locates the global minimum basin for a 6-atom LJ cluster
    (known Cambridge Cluster Database minimum: -12.712062).
    """
    np.random.seed(42)
    M = 8
    N = 6
    R_c = 2.5
    temps = np.geomspace(0.04, 0.35, M)
    deltas = make_deltas(temps)
    energies, coords = make_start(N, M, R_c)

    best_e, best_c = parallel_tempering_run(
        coords, energies, M, temps, deltas, R_c, swap_every=10, N_sweeps=800
    )

    # Unquenched T > 0 energy should already be very close to the -12.712062 global minimum basin
    assert best_e == pytest.approx(-12.712062, abs=0.25), \
        f"Expected PT to reach the LJ_6 global minimum basin (~ -12.712), got {best_e}."
    assert best_e == pytest.approx(cluster_potential(best_c, lennard_jones_potential))

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