import numpy as np
import time
import pytest
import matplotlib.pyplot as plt
from src.main_code.potentials_calculations import cluster_potential, cluster_potential_gradient
from tests.test_potential import simplified_cluster_potential, simplified_cluster_potential_gradient


def cluster_potential_benchmark():
    '''
    Benchmarking function that compares the execution time of the optimized cluster_potential function against a simplified, elementary implementation using double for loops. 
    It measures the execution time for different numbers of particles and plots the results. It also verifies that both implementations yield the same potential value for the same set of coordinates, for the comparison to make sense
    '''
    atom_count = np.arange(2, 200, 2)
    t_elementary_pot = np.zeros_like(atom_count, dtype=float)
    t_optimized_pot = np.zeros_like(atom_count, dtype=float)
    t_elementary_grad = np.zeros_like(atom_count, dtype=float)
    t_optimized_grad = np.zeros_like(atom_count, dtype=float)

    np.random.seed(0)  #reproducibility
    for i, n_particles in enumerate(atom_count):
        coords = np.random.rand(n_particles, 3) * 5 #Slightly spaced out coordinates 

        t_start = time.time()
        simplified_potential = simplified_cluster_potential(coords)
        t_elementary_pot[i] = time.time() - t_start

        t_start = time.time()
        optimized_potential = cluster_potential(coords)
        t_optimized_pot[i] = time.time() - t_start

        t_start = time.time()
        simplified_grad = simplified_cluster_potential_gradient(coords)
        t_elementary_grad[i] = time.time() - t_start

        t_start = time.time()
        optimized_grad = cluster_potential_gradient(coords)
        t_optimized_grad[i] = time.time() - t_start

        assert optimized_potential == pytest.approx(simplified_potential), "The two methods should yield the same result."
        assert optimized_grad == pytest.approx(simplified_grad), "The two methods should yield the same result."

    plt.plot(atom_count, t_elementary_pot, label='Simplified Potential calculator', marker='o')
    plt.plot(atom_count, t_optimized_pot, label='Optimized Potential calculator', marker='x')
    plt.plot(atom_count, t_elementary_grad, label='Simplified Gradient calculator', marker='s')
    plt.plot(atom_count, t_optimized_grad, label='Optimized Gradient calculator', marker='+')
    plt.xlabel('Number of Particles')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison of Potential and Potential Gradient Calculators')
    plt.grid(ls='--')
    plt.legend()
    plt.show()

cluster_potential_benchmark()