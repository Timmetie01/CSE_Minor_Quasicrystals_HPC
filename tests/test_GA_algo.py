import numpy as np
import pytest
from src.main_code.global_optimizers import genetic_algorithm as ga
from src.main_code.potentials_calculations import (
    make_lennard_jones_potential,
    make_lennard_jones_derivative,
    cluster_potential,
)
from src.main_code.local_optimizers.fire import FIRE


def build_ga_operators(rng, cutoff_perc=0.5, mutation_prob=0.25, max_frac=0.15):
    lj_pot = make_lennard_jones_potential(1, 1)
    lj_der = make_lennard_jones_derivative(1, 1)

    selection = ga.make_operators(
        ga.truncation_selection,
        potential_func=lj_pot,
        cutoff_perc=cutoff_perc,
        rng=rng,
    )
    crossover = ga.make_operators(
        ga.cut_splice,
        max_attempts=1000,
        rng=rng,
    )
    mutation = ga.make_operators(
        ga.random_displacement,
        rng=rng,
        prob=mutation_prob,
        max_frac=max_frac,
    )
    local_optimizer = ga.make_population_local_optimizer(
        FIRE,
        potential_derivative_func=lj_der,
    )
    return selection, crossover, mutation, local_optimizer, lj_pot


@pytest.mark.parametrize(
    "n_atoms, known_global_min",
    [
        (4, -6.0),
        (5, -9.103852),
    ],
)
def test_ga_finds_lj4_and_lj5_global_minimum(n_atoms, known_global_min):
    np.random.seed(42)
    rng = np.random.default_rng(42)

    Np = 6
    initial_population = ga.rcube_initial_population(Np=Np, K=n_atoms, r=3.0)
    selection, crossover, mutation, local_optimizer, lj_pot = build_ga_operators(rng)

    final_population = ga.GA(
        initial_population,
        selection,
        crossover,
        mutation,
        local_optimizer,
        number_of_iterations=10,
        elitism=2,
    )

    assert len(final_population) == Np

    energies = [cluster_potential(c, lj_pot) for c in final_population]
    best_energy = min(energies)

    assert np.isclose(best_energy, known_global_min, atol=1e-4)


def test_ga_escapes_local_minima_lj7():
    rng = np.random.default_rng(48)
    a = 2 ** (1 / 6)
    trapped_coords = np.array([
        [a, 0.0, 0.0],
        [-a, 0.0, 0.0],
        [0.0, a, 0.0],
        [0.0, -a, 0.0],
        [0.0, 0.0, a],
        [0.0, 0.0, -a],
        [a, a, a],
    ])

    Np = 8
    initial_population = [trapped_coords.copy() for _ in range(Np)]
    selection, crossover, mutation, local_optimizer, lj_pot = build_ga_operators(
        rng, cutoff_perc=0.5, mutation_prob=0.4, max_frac=0.2
    )

    final_population = ga.GA(
        initial_population,
        selection,
        crossover,
        mutation,
        local_optimizer,
        number_of_iterations=25,
        elitism=2,
    )

    energies = [cluster_potential(c, lj_pot) for c in final_population]
    best_energy = min(energies)
    known_global_min_energy = -16.505384

    assert best_energy < -15.935043
    assert np.isclose(best_energy, known_global_min_energy, atol=1e-3)


def test_ga_invalid_parameters():
    rng = np.random.default_rng(1)
    Np = 4
    initial_population = ga.rcube_initial_population(Np=Np, K=4, r=3.0)
    selection, crossover, mutation, local_optimizer, _ = build_ga_operators(rng)

    # elitism >= Np
    with pytest.raises(ValueError):
        ga.GA(initial_population, selection, crossover, mutation, local_optimizer,
              number_of_iterations=5, elitism=Np)

    # negative elitism
    with pytest.raises(ValueError):
        ga.GA(initial_population, selection, crossover, mutation, local_optimizer,
              number_of_iterations=5, elitism=-1)

    # negative iterations
    with pytest.raises(ValueError):
        ga.GA(initial_population, selection, crossover, mutation, local_optimizer,
              number_of_iterations=-1, elitism=0)

    # population too small
    with pytest.raises(ValueError):
        ga.GA([initial_population[0]], selection, crossover, mutation, local_optimizer,
              number_of_iterations=5, elitism=0)