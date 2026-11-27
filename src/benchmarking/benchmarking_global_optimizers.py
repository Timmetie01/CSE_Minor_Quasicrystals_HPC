import numpy as np
import matplotlib.pyplot as plt
from src.main_code.classes import LennardJones, OscillatingPair, RawOscillatingPair, Optimizer
from src.main_code.local_optimizers import bfgs
import time as t
from tqdm import tqdm
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

    #Ensure all jit-compiled functions are done compiling
    njit_init = Opt.basin_hopping(N_atoms, bfgs.lbfgs_njit)
    njit_init = Opt.genetic_algorithm(N_atoms) 
    #njit_init = Opt.parallel_tempering(N_atoms)
    njit_init = Opt.parallel_tempering_numba(N_atoms)
    njit_init = Opt.simulated_annealing(N_atoms) 

    for i in range(N_runs):
        t0 = t.time()
        coords = Opt.basin_hopping(N_atoms, bfgs.lbfgs_njit)
        basin_hopping[0,i] = t.time() - t0
        basin_hopping[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

        t0 = t.time()
        coords = Opt.genetic_algorithm(N_atoms)
        genetic_algorithm[0,i] = t.time() - t0
        genetic_algorithm[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

        #t0 = t.time()
        #coords = Opt.parallel_tempering(N_atoms)
        #parallel_tempering[0,i] = t.time() - t0
        #parallel_tempering[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)
 
        t0 = t.time()
        coords = Opt.parallel_tempering_numba(N_atoms)
        parallel_tempering_numba[0,i] = t.time() - t0
        parallel_tempering_numba[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

        t0 = t.time()
        coords = Opt.simulated_annealing(N_atoms)
        simulated_annealing[0,i] = t.time() - t0
        simulated_annealing[1,i] = (1 if Potential.potential(coords) == pytest.approx(opt_pot) else 0)

    # plt.hist([basin_hopping[0,:], genetic_algorithm[0,:], parallel_tempering[0,:], parallel_tempering_numba[0,:], simulated_annealing[0,:]],
    #         label=["Basin Hopping", "Genetic Algorithm", "Parallel Tempering", "Parallel Tempering (Numba)", "Simulated Annealing"],
    #         bins=25
    # )
    # plt.title("Histogram of Global Optimizer Runtimes")
    # plt.grid(ls='--')
    # plt.xlabel("Time spent running (s)")
    # plt.ylabel("Amount of times occured")
    # plt.xlim(0, 50)
    # plt.legend()
    # plt.show()

    methods = [basin_hopping, genetic_algorithm, parallel_tempering_numba, simulated_annealing]
    labels = ["Basin Hopping", "Genetic Algorithm", "Parallel Tempering (Numba)", "Simulated Annealing"]

    times = [m[0, :] for m in methods]
    hits = [int(np.sum(m[1, :])) for m in methods]
    hit_rates = [100.0 * h / N_runs for h in hits]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Runtime distribution + individual run dots
    ax1.boxplot(times, tick_labels=labels, showfliers=False)
    for idx, t_arr in enumerate(times, start=1):
        ax1.scatter(np.full_like(t_arr, idx), t_arr, alpha=0.7, zorder=3)
    ax1.set_title(f"Runtime Distribution (N={N_atoms}, {N_runs} runs)")
    ax1.set_ylabel("Time spent running (s)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(ls="--", alpha=0.6)

    # Right: Global minimum hit count / percentage
    bars = ax2.bar(labels, hit_rates, color="steelblue")
    ax2.axhline(95, color="red", ls="--", label="95% target")
    ax2.set_ylim(0, 110)
    ax2.set_title("Global Minimum Success Rate")
    ax2.set_ylabel("Reached Global Minimum (%)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(ls="--", alpha=0.6)
    ax2.legend()

    for bar, h in zip(bars, hits):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{h}/{N_runs}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()













#compare_single_cluster_size(50, 2)





