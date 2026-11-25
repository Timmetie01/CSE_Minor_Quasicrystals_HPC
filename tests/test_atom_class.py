from src.main_code.classes import Optimizer, LennardJones
from src.main_code.local_optimizers import fire
import numpy as np
import pytest

def test_optimizers_callable():
    """
    Fast test verifying that every local and global method in Optimizer
    can be called without crashing and returns valid (N, 3) coordinate arrays.
    Mostly checks if none of the arguments have changed. If they have, change the atom class.
    """
    np.random.seed(42)
    pot = LennardJones(epsilon=1.0, sigma=1.0)
    opt = Optimizer(pot)

    n_atoms = 3
    init_coords = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])

    def _assert_valid_coords(coords, label):
        assert isinstance(coords, np.ndarray), f"{label}: expected np.ndarray, got {type(coords)}"
        assert coords.shape == (n_atoms, 3), f"{label}: expected shape {(n_atoms, 3)}, got {coords.shape}"
        assert np.all(np.isfinite(coords)), f"{label}: returned NaN or Inf values"
        assert pot.potential(coords) == pytest.approx(-3)

    # 1. Local optimizers
    res_fire = opt.FIRE(init_coords.copy())
    _assert_valid_coords(res_fire, "FIRE")

    res_fire_steps, steps_fire = opt.FIRE(init_coords.copy(), keep_steps=True)
    _assert_valid_coords(res_fire_steps, "FIRE (keep_steps)")
    assert steps_fire.ndim == 3 and steps_fire.shape[1:] == (n_atoms, 3)

    res_bfgs = opt.BFGS(init_coords.copy(), print_convergence=False)
    _assert_valid_coords(res_bfgs, "BFGS")

    res_lbfgs = opt.LBFGS(init_coords.copy(), print_convergence=False)
    _assert_valid_coords(res_lbfgs, "LBFGS")

    # 2. Global optimizers (minimal iterations & log_info=None to avoid polluting highscores)
    res_bh_str = opt.basin_hopping(
        n_atoms,
        local_optimizer="LBFGS",
        local_optimizer_kwargs=None,
        max_iter=3,
        delta=0.5,
        log_info=None,
    )
    _assert_valid_coords(res_bh_str, "basin_hopping (LBFGS string)")

    res_bh_fire = opt.basin_hopping(
        n_atoms,
        local_optimizer=fire.FIRE,
        local_optimizer_kwargs=None,
        max_iter=2,
        init_coords=init_coords,
        log_info=None,
    )
    _assert_valid_coords(res_bh_fire, "basin_hopping (FIRE callable)")

    res_ga = opt.genetic_algorithm(
        n_atoms,
        Pop_size=4,
        local_optimizer="LBFGS",
        local_optimizer_kwargs=None,
        number_of_iterations=20,
        elitism=1,
        seed=42,
        log_info=None,
    )
    _assert_valid_coords(res_ga, "genetic_algorithm")

    res_pt = opt.parallel_tempering(
        n_atoms,
        M=3,
        R_c=3.0,
        swap_every=2,
        N_sweeps=50,
        quench=True,
        local_optimizer="BFGS",
        local_optimizer_kwargs=None,
        log_info=None,
    )
    _assert_valid_coords(res_pt, "parallel_tempering")

    res_sa = opt.simulated_annealing(
        n_atoms,
        local_optimizer="FIRE",
        local_optimizer_kwargs=None,
        steps_per_atom=10,
        adapt_every=5,
        seed=42,
        log_info=None,
    )
    _assert_valid_coords(res_sa, "simulated_annealing")
