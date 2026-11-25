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
        (10, -28.422, 100),
        (25, -80.336, 500),
        (50, -196.831, 1000),
        (80, -397.492, 2000),
    ])    
    
def test_ml_basin_hopping_increasing_lj_cluster(n_atoms, known_global_min_energy, max_iter=100):
    np.random.seed(42)
    initial_coords = np.random.rand(n_atoms, 3) * 5.0

    # Parameters for ML-BH
    beta = 1.0
    delta = 0.5

    optimized_coords, optimized_energy, iterations, accepted_moves = mlbh.ml_basin_hopping(
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
    
    
    
