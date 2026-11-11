import numpy as np
from src.main_code.potentials_calculations import lennard_jones_potential, cluster_potential
import pytest
import math as m
import time
import matplotlib.pyplot as plt

def simplified_cluster_potential(coords: np.ndarray, potential_func=lennard_jones_potential):
    '''
    Calculate the total Lennard-Jones potential for a cluster of particles in a simple and correct but slow version, to benchmark against the optimized version.

    :param coords: array_like, shape (N, 3), 
        Coordinates of N particles in 3D space.
    :param potential_func: callable, optional,
        Function to calculate the potential between two particles.
    :returns: float,
        Total potential of the cluster.
    '''

    total_potential = 0.0
    num_particles = len(coords)

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            distance = np.linalg.norm(coords[i] - coords[j])
            total_potential += potential_func(distance)

    return total_potential


def test_cluster_potential(potential_func=lennard_jones_potential):
    for i in range(2, 103, 10):
        np.random.seed(0)
        coords = np.random.rand(i, 3) * 5  # Random coordinates for i particles
        assert cluster_potential(coords, potential_func) == pytest.approx(simplified_cluster_potential(coords, potential_func)), "The two methods should yield the same result."

 
def test_single_particle(potential_func=lennard_jones_potential):  
    np.random.seed(0)  
    coords = np.random.rand(1, 3)
    assert cluster_potential(coords, potential_func) == simplified_cluster_potential(coords, potential_func) == 0, "One particle should yield a potential of 0."


def test_same_position_particles(potential_func=lennard_jones_potential): 
    '''
    Test the behavior of the cluster_potential function when particles are at the same position. It should raise a RuntimeWarning and return NaN, as the potential is undefined (or could be considered infinite) for zero distance.
    '''
    
    coords = np.array([[1, 1, 1], [1, 1, 1]])  # Two particles at the same position
    with pytest.warns(RuntimeWarning):
        assert m.isnan(cluster_potential(coords, potential_func)), "The potential should be NaN for particles at the same position, and give a RuntimeWarning."

    coords = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3]])  # Should still fail and raise warnings if not all but some particles are at the same position
    with pytest.warns(RuntimeWarning):
        assert m.isnan(cluster_potential(coords, potential_func)), "The potential should be NaN for particles at the same position, and give a RuntimeWarning."

    coords = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])  # Should still fail and raise warnings if some particles are at different same positions
    with pytest.warns(RuntimeWarning):
        assert m.isnan(cluster_potential(coords, potential_func)), "The potential should be NaN for particles at the same position, and give a RuntimeWarning."

def test_potential_with_table():
    '''
    Using the tables found at https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ/tables.150.html, check if the potential calculator agrees on the values obtained when putting atoms at the specified coordinates.
    '''

    # For 2 particles at distance 1.122462048309373, the potential should be -1.0
    coords_2_particles = np.array([[0, 0, 0], [1.122462048309373, 0, 0]])
    assert cluster_potential(coords_2_particles, lennard_jones_potential) == pytest.approx(-1.0), "The potential for 2 particles at the specified distance should be -1.0."

    coords_3_particles = np.array([[0.4391356726, 0.1106588251, -0.4635601962], [-0.5185079933, 0.3850176090, 0.0537084789],[0.0793723207, -0.4956764341, 0.4098517173]])
    assert cluster_potential(coords_3_particles, lennard_jones_potential) == pytest.approx(-3.0), "The potential for 3 particles at the specified coordinates should be -3.0."

    coords_10_particles =    np.array([[-0.112678561569957,   1.154350068036701,  -0.194840194577019],
                    [0.455529927957375,  -0.141698933988423,   1.074987039970359],
                    [-1.076845089999501,  -0.472850203002737,  -0.000759561321676],
                    [0.772283646029137,   0.650565133523509,   0.318426126854726],
                    [-0.156611774846419,  -0.917108951862921,   0.505406908904027],
                    [-0.704034080497556,   0.331528029107990,  -0.717967548406147],
                    [ 0.749396644213383,  -0.438279561655135,   0.048302291736783],
                    [-0.145155893839899,  -0.631595300103604,  -0.579684879295956],
                    [0.417694325594403,   0.318313250559209,  -0.692983073948749],
                    [-0.199579143040966,   0.146776469285409,   0.23911289008365]])

    assert cluster_potential(coords_10_particles, lennard_jones_potential) == pytest.approx(-28.422532), "The potential for 10 particles at the specified coordinates should be -28.422532."