import numpy as np
import pytest
from src.main_code.global_optimizers import basin_hopping as bh
from src.main_code import potentials_calculations as pot
from src.main_code.local_optimizers.fire import FIRE

def local_minimization_wrapper(coords: np.ndarray) -> np.ndarray:
    return FIRE(coords, pot.lennard_jones_derivative)

def test_basin_hopping_finds_lj4_global_minimum():
    initial_coords = np.array([[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.5, 1.2, 0.0],
                               [0.5, 0.3, 0.3]])

    # Parameters for BH
    beta = 1.0
    max_iter = 100
    delta = 0.5

    optimized_coords, optimized_energy, iterations, accepted_moves = bh.basin_hopping(
        input_coords=initial_coords,
        potential_function=pot.lennard_jones_potential,
        local_optimizer_wrapper=local_minimization_wrapper,
        beta=beta,
        max_iter=max_iter,
        perturbation_type="random",
        delta=delta
    )

    known_global_min_energy = -6.0

    # Assert that the optimized energy is close to the known global minimum
    assert np.isclose(optimized_energy, known_global_min_energy, atol=1e-2)
    
def test_basin_hopping_invalid_parameters():
    initial_coords = np.array([[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.5, 1.2, 0.0]])

    # Test invalid beta
    with pytest.raises(ValueError):
        bh.basin_hopping(initial_coords, pot.lennard_jones_potential, local_minimization_wrapper,
                         beta=-1.0, max_iter=10)

    # Test invalid max_iter
    with pytest.raises(ValueError):
        bh.basin_hopping(initial_coords, pot.lennard_jones_potential, local_minimization_wrapper,
                         beta=1.0, max_iter=0)

    # Test unsupported perturbation type
    with pytest.raises(ValueError):
        bh.basin_hopping(initial_coords, pot.lennard_jones_potential, local_minimization_wrapper,
                         beta=1.0, max_iter=10, perturbation_type="unsupported")

    # Test missing delta for random perturbation
    with pytest.raises(ValueError):
        bh.basin_hopping(initial_coords, pot.lennard_jones_potential, local_minimization_wrapper,
                         beta=1.0, max_iter=10, perturbation_type="random", delta=None)
    
def test_basin_hopping_escape_local_minima_lj7_cluster():
    a = 2**(1/6) 
    initial_coords = np.array([[a, 0.0, 0.0],
                              [-a, 0.0, 0.0],
                              [0.0, a, 0.0],
                              [0.0, -a, 0.0],
                              [0.0, 0.0, a],
                              [0.0, 0.0, -a],
                              [a, a, a]])
    
    local_min_coords = local_minimization_wrapper(initial_coords)
    local_min_energy = pot.cluster_potential(local_min_coords, pot.lennard_jones_potential)


    #parameters for BH
    beta = 1.0
    max_iter = 200
    delta = 0.5

    optimized_coords, optimized_energy, iterations, accepted_moves = bh.basin_hopping(
        input_coords=local_min_coords,
        potential_function=pot.lennard_jones_potential,
        local_optimizer_wrapper=local_minimization_wrapper,
        beta=beta,
        max_iter=max_iter,
        perturbation_type="random",
        delta=delta
        )

    known_global_min_energy = -16.505384

    assert np.isclose(local_min_energy, -15.935043, atol=1e-2)
    assert optimized_energy < local_min_energy
    assert np.isclose(optimized_energy, known_global_min_energy, atol=1e-2)    
    