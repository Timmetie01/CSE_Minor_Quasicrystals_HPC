import numpy as np
import pytest
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential
from src.main_code.monte_carlo_sweep import mc_step, mc_sweep


def test_mc_step_downhill_always_accepted():
    """A move that strictly lowers energy must always be accepted, even at near-zero T (huge beta)."""
    np.random.seed(42)
    coords = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    initial_energy = cluster_potential(coords, lennard_jones_potential)
    R_c = 5.0

    # Run attempts until we get a move that lowers the energy; with large beta, only downhill moves can pass
    accepted_any = False
    for _ in range(100):
        old_coords = coords.copy()
        new_energy, accepted = mc_step(coords, initial_energy, beta=1e6, delta=0.1, R_c=R_c)
        if accepted:
            accepted_any = True
            assert new_energy <= initial_energy, "At huge beta, accepted move must not increase energy."
            assert new_energy == pytest.approx(cluster_potential(coords, lennard_jones_potential))
            assert not np.array_equal(coords, old_coords), "Coordinates should update in-place on accept."
            break
        else:
            assert new_energy == pytest.approx(initial_energy), "Rejected move must return old energy."
            assert np.array_equal(coords, old_coords), "Rejected move must revert coordinates."

    assert accepted_any, "Expected at least one downhill move to be found and accepted."


def test_mc_step_rejects_outside_cutoff():
    """Verify that any move pushing an atom beyond R_c is rejected and reverted, even if downhill."""
    np.random.seed(10)
    R_c = 2.0
    # Place 2 close atoms near the boundary (r = 1.95 < R_c = 2.0) where moving outward lowers repulsion
    coords = np.array([[1.95, 0.0, 0.0], [1.10, 0.0, 0.0]])
    initial_energy = cluster_potential(coords, lennard_jones_potential)

    for _ in range(50):
        old_coords = coords.copy()
        energy, accepted = mc_step(coords, initial_energy, beta=0.01, delta=0.2, R_c=R_c)
        # Every atom must strictly stay within R_c at all times
        radii = np.linalg.norm(coords, axis=1)
        assert np.all(radii <= R_c + 1e-12), "Atoms must never exceed R_c."
        if accepted:
            initial_energy = energy
        else:
            assert energy == pytest.approx(initial_energy)
            assert np.array_equal(coords, old_coords)


def test_mc_step_single_atom_and_bounds():
    """Verify at most one atom moves per step, its displacement never exceeds delta, and all stay within R_c."""
    np.random.seed(0)
    R_c = 5.0
    coords = np.random.rand(10, 3) * 2.5  # max radius is sqrt(3)*2.5 ≈ 4.33 < R_c
    delta = 0.05
    energy = cluster_potential(coords, lennard_jones_potential)

    for _ in range(50):
        before = coords.copy()
        energy, accepted = mc_step(coords, energy, beta=0.1, delta=delta, R_c=R_c)
        diff = np.abs(coords - before)
        moved_atoms = np.any(diff > 0.0, axis=1)

        assert np.all(np.linalg.norm(coords, axis=1) <= R_c + 1e-12), "No atom should exceed R_c."

        if accepted:
            assert np.sum(moved_atoms) <= 1, "At most one atom should move in a single mc_step."
            assert np.max(diff) <= delta + 1e-12, "Atom displacement must not exceed [-delta, delta]."
        else:
            assert np.sum(moved_atoms) == 0, "No atoms should move when step is rejected."


def test_mc_sweep_energy_consistency_and_best_tracking():
    """Verify returned final energy matches final coords, and best_energy matches best_coords."""
    np.random.seed(123)
    R_c = 5.0
    coords = np.random.rand(6, 3) * 2.0
    initial_energy = cluster_potential(coords, lennard_jones_potential)
    beta = 1.0 / 0.5
    n_moves = 200

    final_energy, best_energy, best_coords, n_accepted = mc_sweep(
        coords, energy=initial_energy, beta=beta, n_moves=n_moves, delta=0.1, R_c=R_c
    )

    assert 0 <= n_accepted <= n_moves
    assert best_energy <= final_energy, "best_energy must be <= final_energy."
    assert final_energy == pytest.approx(cluster_potential(coords, lennard_jones_potential)), \
        "Returned final energy must match the potential of the final coords."
    assert best_energy == pytest.approx(cluster_potential(best_coords, lennard_jones_potential)), \
        "Returned best_energy must match the potential of best_coords."


def test_mc_sweep_finds_lj3_global_minimum():
    """Verify mc_sweep relaxes a distorted 3-atom LJ cluster close to the known global minimum (-3.0)."""
    np.random.seed(99)
    R_c = 5.0
    # Start near an equilateral triangle with slight perturbation
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.25, 0.05, 0.0],
        [0.55, 1.05, -0.05]
    ])
    initial_energy = cluster_potential(coords, lennard_jones_potential)
    beta = 1.0 / 0.05

    _, best_energy, best_coords, _ = mc_sweep(
        coords, energy=initial_energy, beta=beta, n_moves=3000, delta=0.05, R_c=R_c
    )

    assert best_energy == pytest.approx(-3.0, abs=0.02), \
        "3-atom LJ cluster should relax close to the global minimum of -3.0."
    assert best_coords.shape == (3, 3)