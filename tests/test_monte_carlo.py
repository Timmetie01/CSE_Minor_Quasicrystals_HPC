import numpy as np
import pytest
from src.main_code.potentials_calculations import (
    lennard_jones_potential,
    lennard_jones_potential_njit,
    cluster_potential,
)
from src.main_code.monte_carlo_sweep import mc_step, mc_sweep, mc_sweep_njit


def test_mc_step_downhill_always_accepted():
    """A move that strictly lowers energy must always be accepted, even at near-zero T (huge beta)."""
    np.random.seed(42)
    coords = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    initial_energy = cluster_potential(coords, lennard_jones_potential)
    R_c = 5.0

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
    coords = np.array([[1.95, 0.0, 0.0], [1.10, 0.0, 0.0]])
    initial_energy = cluster_potential(coords, lennard_jones_potential)

    for _ in range(50):
        old_coords = coords.copy()
        energy, accepted = mc_step(coords, initial_energy, beta=0.01, delta=0.2, R_c=R_c)
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
    coords = np.random.rand(10, 3) * 2.5
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
    n_atoms = 6
    coords = np.random.rand(n_atoms, 3) * 2.0
    initial_energy = cluster_potential(coords, lennard_jones_potential)
    beta = 1.0 / 0.5
    n_moves = 200

    final_energy, best_energy, best_coords, n_accepted = mc_sweep(
        coords, energy=initial_energy, n_atoms=n_atoms, beta=beta, n_moves=n_moves, delta=0.1, R_c=R_c
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
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.25, 0.05, 0.0],
        [0.55, 1.05, -0.05]
    ])
    n_atoms = coords.shape[0]
    initial_energy = cluster_potential(coords, lennard_jones_potential)
    beta = 1.0 / 0.05

    _, best_energy, best_coords, _ = mc_sweep(
        coords, energy=initial_energy, n_atoms=n_atoms, beta=beta, n_moves=3000, delta=0.05, R_c=R_c
    )

    assert best_energy == pytest.approx(-3.0, abs=0.02), \
        "3-atom LJ cluster should relax close to the global minimum of -3.0."
    assert best_coords.shape == (3, 3)


def test_mc_sweep_njit_congruent_with_python_sweep():
    """
    Verify mc_sweep_njit matches mc_sweep in energy accounting, boundary enforcement,
    and equilibrium ensemble statistics (mean energy and acceptance rate).
    """
    np.random.seed(42)
    R_c = 3.5
    beta = 1.0 / 0.25
    delta = 0.08
    n_moves = 500
    n_trials = 40

    # Start near a relaxed 4-atom tetrahedron
    a = 2.0 ** (1.0 / 6.0)
    base_coords = np.array([
        [0.0, 0.0, 0.0],
        [a, 0.0, 0.0],
        [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
        [0.5 * a, a / (2.0 * np.sqrt(3.0)), np.sqrt(2.0 / 3.0) * a],
    ])
    n_atoms = base_coords.shape[0]
    base_energy = cluster_potential(base_coords, lennard_jones_potential)

    py_energies, njit_energies = [], []
    py_acc_rates, njit_acc_rates = [], []

    for _ in range(n_trials):
        c_py = base_coords.copy()
        e_py, best_e_py, best_c_py, acc_py = mc_sweep(
            c_py, base_energy, n_atoms, beta, n_moves, delta, R_c, lennard_jones_potential
        )

        c_njit = base_coords.copy()
        e_njit, best_e_njit, best_c_njit, acc_njit = mc_sweep_njit(
            c_njit, base_energy, n_atoms, beta, n_moves, delta, R_c, lennard_jones_potential_njit
        )

        # Exact energy and boundary invariants for mc_sweep_njit
        assert 0 <= acc_njit <= n_moves
        assert best_e_njit <= e_njit
        assert np.all(np.linalg.norm(c_njit, axis=1) <= R_c + 1e-12)
        assert e_njit == pytest.approx(cluster_potential(c_njit, lennard_jones_potential), abs=1e-10)
        assert best_e_njit == pytest.approx(cluster_potential(best_c_njit, lennard_jones_potential), abs=1e-10)

        py_energies.append(e_py)
        njit_energies.append(e_njit)
        py_acc_rates.append(acc_py / n_moves)
        njit_acc_rates.append(acc_njit / n_moves)

    # Thermodynamic / statistical congruence across ensembles
    assert np.mean(njit_energies) == pytest.approx(np.mean(py_energies), abs=0.15), \
        f"Mean energy mismatch: njit={np.mean(njit_energies):.4f} vs python={np.mean(py_energies):.4f}"
    assert np.mean(njit_acc_rates) == pytest.approx(np.mean(py_acc_rates), abs=0.05), \
        f"Acceptance rate mismatch: njit={np.mean(njit_acc_rates):.4f} vs python={np.mean(py_acc_rates):.4f}"