import numpy as np
import pytest
from src.main_code.global_optimizers import ml_basin_hopping as mlbh
from src.main_code.local_optimizers.bfgs import LBFGS
from src.main_code.potentials_calculations import (
    make_lennard_jones_potential,
    make_lennard_jones_derivative,
    cluster_potential,
)
   
@pytest.mark.parametrize("n_atoms, known_global_min_energy", "max_iter", [
        (10, -28.422532, 100),
        (25, -102.372663, 500),
        (50, -244.549926, 1000),
        (80, -428.083564, 2000),
    ])    
    
def test_ml_basin_hopping_increasing_lj_cluster(n_atoms, known_global_min_energy, max_iter=100):
    np.random.seed(42)
    initial_coords = np.random.rand(n_atoms, 3) * 5.0

    # Parameters for ML-BH
    beta = 1.0
    delta = 0.5

    optimized_coords, optimized_energy, iterations, accepted_moves = mlbh.basin_hopping_opt(
        input_coords=initial_coords,
        potential_function=make_lennard_jones_potential(),
        local_optimizer_wrapper=LBFGS(make_lennard_jones_derivative()),
        beta=beta,
        max_iter=max_iter,
        perturbation_type="random",
        delta=delta
    )

    # Assert that the optimized energy is close to the known global minimum
    assert np.isclose(optimized_energy, known_global_min_energy, atol=1e-2)

    assert np.isclose(optimized_energy,
                      cluster_potential(optimized_coords, make_lennard_jones_potential()),
                      atol=1e-6,
                      rtol=0)
    
def test_ml_basin_hopping_invalid_parameters():
    initial_coords = np.array([[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.5, 1.2, 0.0]])
    potential = make_lennard_jones_potential()
    optimizer = LBFGS(make_lennard_jones_derivative())
    
    with pytest.raises(ValueError):
        mlbh.basin_hopping_opt(initial_coords, potential, optimizer, beta=-1.0, max_iter=10)
        
    with pytest.raises(ValueError):
        mlbh.basin_hopping_opt(initial_coords, potential, optimizer, beta=1.0, max_iter=0)
    
    with pytest.raises(ValueError):
        mlbh.basin_hopping_opt(initial_coords, potential, optimizer, beta=1.0, max_iter=10, perturbation_type="unsupported")
    
    with pytest.raises(ValueError):
        mlbh.basin_hopping_opt(initial_coords, potential, optimizer, beta=1.0, max_iter=10, perturbation_type="random", delta=-0.5)
        
    with pytest.raises(ValueError):
        mlbh.basin_hopping_opt(initial_coords, potential, optimizer, beta=1.0, max_iter=10, perturbation_type="random", delta=None)
    
    with pytest.raises(ValueError):
        mlbh.basin_hopping_opt(np.empty((0, 3)), potential, optimizer, beta=1.0, max_iter=10)
        
