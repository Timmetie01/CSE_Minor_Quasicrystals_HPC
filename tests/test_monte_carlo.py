import numpy as np
import pytest
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential
from src.main_code.monte_carlo_sweep  import mc_step, mc_sweep


def test_mc_step_downhill_always_accepted():
    """A move that strictly lowers energy must always be accepted, even at near-zero T (huge beta)."""
    np.random.seed(42)
    coords = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    initial_energy = cluster_potential(coords, lennard_jones_potential)

    # Run attempts until we get a move that lowers the energy; with large beta, only downhill moves can pass
    accepted_any = False
    for _ in range(100):
        old_coords = coords.copy()
        new_energy, accepted = mc_step(coords, initial_energy, beta=1e6, delta=0.1)
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


def test_mc_step_single_atom_and_bounds():
    """Verify at most one atom moves per step, and its displacement never exceeds delta."""
    np.random.seed(0)
    coords = np.random.rand(10, 3) * 3.0
    delta = 0.05
    energy = cluster_potential(coords, lennard_jones_potential)

    for _ in range(50):
        before = coords.copy()
        energy, accepted = mc_step(coords, energy, beta=0.1, delta=delta)
        diff = np.abs(coords - before)
        moved_atoms = np.any(diff > 0.0, axis=1)

        if accepted:
            assert np.sum(moved_atoms) <= 1, "At most one atom should move in a single mc_step."
            assert np.max(diff) <= delta + 1e-12, "Atom displacement must not exceed [-delta, delta]."
        else:
            assert np.sum(moved_atoms) == 0, "No atoms should move when step is rejected."


def test_mc_sweep_energy_consistency_and_best_tracking():
    """Verify returned final energy matches final coords, and best_energy matches best_coords."""
    np.random.seed(123)
    coords = np.random.rand(6, 3) * 2.5
    n_moves = 200

    final_energy, best_energy, best_coords, n_accepted = mc_sweep(
        coords, T=0.5, n_moves=n_moves, delta=0.1
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
    # Start near an equilateral triangle with slight perturbation
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.25, 0.05, 0.0],
        [0.55, 1.05, -0.05]
    ])

    _, best_energy, best_coords, _ = mc_sweep(coords, T=0.05, n_moves=3000, delta=0.05)

    assert best_energy == pytest.approx(-3.0, abs=0.02), \
        "3-atom LJ cluster should relax close to the global minimum of -3.0."
    assert best_coords.shape == (3, 3)