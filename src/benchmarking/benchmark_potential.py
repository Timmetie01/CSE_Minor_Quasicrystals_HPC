import numpy as np
import time
import pytest
import matplotlib.pyplot as plt
from src.main_code.potentials_calculations import cluster_potential
from tests.test_potential import simplified_cluster_potential


def cluster_potential_benchmark():
    '''
    Benchmarking function that compares the execution time of the optimized cluster_potential function against a simplified, elementary implementation using double for loops. 
    It measures the execution time for different numbers of particles and plots the results. It also verifies that both implementations yield the same potential value for the same set of coordinates, for the comparison to make sense
    '''
    atom_count = np.arange(2, 300, 2)
    t_elementary_implementation = np.zeros_like(atom_count, dtype=float)
    t_optimized_implementation = np.zeros_like(atom_count, dtype=float)

    np.random.seed(0)  #reproducibility
    for i, n_particles in enumerate(atom_count):
        coords = np.random.rand(n_particles, 3) * 5 #Slightly spaced out coordinates 

        t_start = time.time()
        simplified_potential = simplified_cluster_potential(coords)
        t_elementary_implementation[i] = time.time() - t_start

        t_start = time.time()
        optimized_potential = cluster_potential(coords)
        t_optimized_implementation[i] = time.time() - t_start

        assert optimized_potential == pytest.approx(simplified_potential), "The two methods should yield the same result."

    plt.plot(atom_count, t_elementary_implementation, label='Elementary Implementation (double for loop)', marker='o')
    plt.plot(atom_count, t_optimized_implementation, label='Optimized Implementation', marker='x')
    plt.xlabel('Number of Particles')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.show()

cluster_potential_benchmark()