import numpy as np
from src.main_code import potentials_calculations

class AtomCluster:
    def __init__(self, coords: np.ndarray, potential_function: callable = potentials_calculations.make_lennard_jones_potential(epsilon=1, sigma=1)):
        '''
        Initialize the AtomCluster with coordinates.

        :param coords: array_like, shape (N, 3), 
            Coordinates of N particles in 3D space.
        :param potential_function: callable, optional,
            Function to calculate the potential based on float or array-like distance r. Defaults to Lennard-Jones potential.
        '''
        self.coords = np.asarray(coords)
        self.atom_count = self.coords.shape[0]

        self.potential_function = potential_function

    def potential(self):
        '''
        Calculate the total potential for the cluster using the provided potential function.

        :param potential_func: callable, 
            Function to calculate the potential based on float or array-like distance r. if None, defaults to self.potential_function. If that is none, defaults to Lennard-Jones potential.
        :returns: float, 
            Total potential of the cluster.
        '''

        return potentials_calculations.cluster_potential(self.coords, self.potential_function)
