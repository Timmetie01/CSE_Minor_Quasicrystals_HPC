import numpy as np
import matplotlib.pyplot as plt
from src.main_code.classes import LennardJones, OscillatingPair, RawOscillatingPair, Optimizer
import time as t
from data.data_handling import read_LJwales_data
import pytest



def compare_single_cluster_size(N_atoms, N_runs):
    #Arrays constisting of the times [0,:] and if they converged to the global minimum [1,:] (1=True, 0=False)
    basin_hopping = np.zeros((2, N_runs))
    genetic_algorithm = np.zeros((2, N_runs))
    parallel_tempering = np.zeros((2, N_runs))
    parallel_tempering_numba = np.zeros((2, N_runs))
    simulated_annealing = np.zeros((2, N_runs))

    Potential = LennardJones(1,1)
    Opt = Optimizer(Potential)
    opt_pot = Potential.potential(read_LJwales_data(N_atoms))

    for i in range(N_runs):
        t0 = t.time()
        coords = Opt.basin_hopping(N_atoms)
        basin_hopping[0,i] = t.time() - t0
        basin_hopping[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

        t0 = t.time()
        coords = Opt.genetic_algorithm(N_atoms)
        genetic_algorithm[0,i] = t.time() - t0
        genetic_algorithm[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

        t0 = t.time()
        coords = Opt.parallel_tempering(N_atoms)
        parallel_tempering[0,i] = t.time() - t0
        parallel_tempering[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)
 
        # t0 = t.time()
        # coords = Opt.parallel_tempering_numba(N_atoms)
        # parallel_tempering_numba[0,i] = t.time() - t0
        # parallel_tempering_numba[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

        t0 = t.time()
        coords = Opt.simulated_annealing(N_atoms)
        simulated_annealing[0,i] = t.time() - t0
        simulated_annealing[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

    plt.hist([basin_hopping[0,:], genetic_algorithm[0,:], parallel_tempering[0,:], parallel_tempering_numba[0,:], simulated_annealing[0,:]],
            label=["Basin Hopping", "Genetic Algorithm", "Parallel Tempering", "Parallel Tempering (Numba)", "Simulated Annealing"],
            bins=15
    )
    plt.title("Histogram of Global Optimizer Runtimes")
    plt.grid(ls='--')
    plt.xlabel("Time spent running (s)")
    plt.ylabel("Amount of times occured")
    plt.xlim(0, 50)
    plt.legend()
    plt.show()


compare_single_cluster_size(20, 10)





