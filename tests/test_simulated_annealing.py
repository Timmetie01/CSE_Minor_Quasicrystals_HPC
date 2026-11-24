import numpy as np
import pytest

import src.main_code.potentials_calculations as pt
from src.main_code.global_optimizers.simulated_annealing import (
    atom_energy,
    random_in_sphere,
    simulated_annealing,
)

#Potentials used by several tests
v = pt.make_lennard_jones_potential()
dv = pt.make_lennard_jones_derivative()


#random_in_sphere 

def test_random_in_sphere_count():
    """It always returns exactly n points."""
    rng = np.random.default_rng(0)
    points = random_in_sphere(50, 2.0, rng)
    assert points.shape == (50, 3)


def test_random_in_sphere_inside():
    """Every point lies inside the sphere."""
    rng = np.random.default_rng(1)
    points = random_in_sphere(1000, 2.0, rng)
    assert np.all(np.linalg.norm(points, axis=1) <= 2.0)


#atom_energy
def test_atom_energy_two_atoms():
    """With two atoms, the energy of one atom is just V(r) between them."""
    x = np.array([[0.0, 0.0, 0.0],
                  [1.5, 0.0, 0.0]])
    assert atom_energy(x, 0, v) == pytest.approx(v(1.5))


def test_atom_energy_matches_cluster_energy():
    """Removing atom i lowers the total energy by exactly atom_energy(i)."""
    rng = np.random.default_rng(2)
    x = random_in_sphere(10, 2.0, rng)
    i = 3
    E_total = pt.cluster_potential(x, v)
    E_without_i = pt.cluster_potential(np.delete(x, i, axis=0), v)
    assert atom_energy(x, i, v) == pytest.approx(E_total - E_without_i)


#simulated_annealing

def test_annealing_output_shape():
    """The result has one position per atom."""
    x, E = simulated_annealing(5, v, dv, steps_per_atom=200, seed=0)
    assert x.shape == (5, 3)


def test_annealing_energy_matches_positions():
    """The returned energy is the energy of the returned positions."""
    x, E = simulated_annealing(5, v, dv, steps_per_atom=200, seed=0)
    assert E == pytest.approx(pt.cluster_potential(x, v))


def test_annealing_reproducible():
    """The same seed gives the same result."""
    x1, E1 = simulated_annealing(5, v, dv, steps_per_atom=200, seed=4)
    x2, E2 = simulated_annealing(5, v, dv, steps_per_atom=200, seed=4)
    np.testing.assert_allclose(x1, x2)
    assert E1 == E2


def test_annealing_ends_at_a_minimum():
    """After FIRE, the forces on all atoms are (almost) zero."""
    x, E = simulated_annealing(6, v, dv, steps_per_atom=200, seed=0)
    forces = -pt.cluster_potential_gradient(x, dv)
    assert np.max(np.abs(forces)) < 1e-4


def test_annealing_finds_lj4():
    """Four LJ atoms form a tetrahedron with energy -6 (known exactly)."""
    x, E = simulated_annealing(4, v, dv, steps_per_atom=500, seed=0)
    assert E == pytest.approx(-6.0, abs=1e-5)


@pytest.mark.slow
def test_annealing_finds_lj13():
    """At least one of 5 runs finds the LJ13 icosahedron (-44.326801)."""
    energies = [simulated_annealing(13, v, dv, seed=s)[1] for s in range(5)]
    assert min(energies) == pytest.approx(-44.326801, abs=1e-5)