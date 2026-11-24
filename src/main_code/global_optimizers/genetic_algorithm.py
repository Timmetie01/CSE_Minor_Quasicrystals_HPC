from src.main_code.local_optimizers.fire import FIRE
from src.main_code.potentials_calculations import make_lennard_jones_potential, cluster_potential
from functools import partial
import numpy as np

def GA(initial_population, selection, crossover, mutation, local_optimizer,
       number_of_iterations=100):
    """
    Performs global geometry optimization of a particle cluster using a modifiable genetic algorithm.
    
    :param initial_population: list of np-arrays
        List of clusters to start the optimization from
    :param selection: callable
        Function to select parent pairs from the population
    :param crossover: callable
        Function to perform crossover between parent pairs
    :param mutation: callable
        Function to perform mutation on offspring
    :param local_optimizer: callable
        Function to optimize the geometry of individual clusters
    :param number_of_iterations: int, optional
        Number of iterations for the genetic algorithm (default is 100)
    :returns: list of np-arrays
        List of optimized clusters after the specified number of iterations"""
    
    if not isinstance(initial_population, list) or len(initial_population) < 2:
        raise ValueError("initial_population must be a list of at least two clusters")
    if not all(callable(f) for f in (selection, crossover, mutation, local_optimizer)):
        raise ValueError("selection, crossover, mutation and local_optimizer must be callable")
    if not isinstance(number_of_iterations, int) or number_of_iterations < 0:
        raise ValueError("number_of_iterations must be a non-negative integer")

    Np = len(initial_population)

    # relax the starting population once
    current_population = [local_optimizer(c) for c in initial_population]

    for _ in range(number_of_iterations):
        parent_pairs = selection(current_population)            # Np pairs
        children = [crossover(pair) for pair in parent_pairs]   # Np children
        children = [mutation(c) for c in children]              # mutate each child
        current_population = [local_optimizer(c) for c in children]  # relax each child

    return current_population

def truncation_selection(population,potential_func, cutoff_perc,N,rng = None):
    '''
        Perfoms truncation selection on a population of clusters based on their potential energies.
    
        :param population: list of np-arrays
            List of clusters to select from
        :param potential_func: callable
            Function to calculate the potential energy of a cluster
        :param cutof_perc: float between in [0,1)
            Percentage of the population to remove
        :param N: positive integer > 0
            Number of parent pairs to create
        :returns: List of np-arrays,
            List of parent pairs
        :raises ValueError: invalid argument
            if parameters are not of specified type and bound 
        :raises ValueError: Cutoff percentage is too low,
        atleast 2 survivors must remain after truncation          
        '''   
    if not isinstance(population, list) or not all(isinstance(cluster, np.ndarray) for cluster in population):  # maybe remove once optemizing 
        raise ValueError("population must be a list of np-arrays")
    if not callable(potential_func):
        raise ValueError("potential_func must be a callable")
    if not isinstance(cutoff_perc, (int, float)) or not 0 <= cutoff_perc < 1:
        raise ValueError("cutof_perc must be a float between 0 and 1")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if rng is None:
        rng = np.random.default_rng()
    # initialisation and sorting population
    parent_pair_pool = []
    Np = len(population)
    fitness_list = [
        (cluster_potential(cluster, potential_func), cluster)
        for cluster in population
    ]

    fitness_list.sort(key=lambda x: x[0])
    # Truncate population leaving only the best individuals
    cutoff_amount = int(Np * cutoff_perc)
    survivor_amount = Np - cutoff_amount

    if survivor_amount < 2:
        raise ValueError("At least two individuals must remain after truncation")

    truncated_list = [
        cluster for energy, cluster in fitness_list[:survivor_amount]
    ]
    # Randomly select parent pairs from the truncated list without replacement using indices
    for _ in range(N):
        i, j = rng.choice(survivor_amount, size=2, replace=False)
        parent_pair_pool.append((truncated_list[i], truncated_list[j]))

    return parent_pair_pool

def rcube_initial_population(N, K, r):
    '''
    random initilization of a population with N clusters all containing K atoms bounded inside a cube around the origin with side length r

    :param N: positive integer > 0 
         population size
    :param K: positive integer > 0
        cluster size
    :param r: positive float or integer > 0
        side length of bounding cube
    :returns: List of np-arrays,
        containing the randomly initialised clusters    
    :raises ValueError: invalid argument
        if parameters are not of specified type and bound           
    '''  
    if not isinstance(N,int) or N <= 0:
        raise ValueError("N must be a integer value greater then 0")
    if not isinstance(K,int) or K <= 0:
            raise ValueError("K must be a integer value greater then 0")
    if not isinstance(r,(int,float)) or r <= 0:
        raise ValueError("r must be a float value greater then 0")
    
    population = [np.random.uniform(-r/2, r/2, size=(K, 3)) for _ in range(N)]
    return population

def cut_splice(parent_pair, max_attempts=1000, rng=None):
    '''
    Cut and splice crossover operator for two clusters of the same size.
    Both clusters are centered on their centers of mass, then cut by a random
    plane through the origin. Atoms on the positive side of the plane from
    parent 1 are combined with atoms on the negative side from parent 2.
    Planes are redrawn until the child has the same number of atoms as the parents.

    :param parent_pair: tuple or list of two np-arrays of shape (M, 3)
        the parent clusters to crossover
    :param max_attempts: int
        maximum number of random planes to try
    :param rng: np.random.Generator or None
        random number generator (for reproducibility)
    :returns: np-array of shape (M, 3)
        the offspring cluster (centered on its center of mass)
    :raises ValueError: if the parent clusters have different sizes or
        parameters are not of the specified type and bound
    :raises RuntimeError: if no valid cut is found within max_attempts
    '''
    if (not isinstance(parent_pair, (list, tuple)) or len(parent_pair) != 2
            or not all(isinstance(c, np.ndarray) for c in parent_pair)):
        raise ValueError("parent_pair must be a list or tuple of two numpy array clusters")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")

    parent1, parent2 = parent_pair
    M = len(parent1)
    if len(parent2) != M:
        raise ValueError("Parent clusters must have the same number of atoms")
    if parent1.ndim != 2 or parent1.shape[1] != 3 or parent2.shape != parent1.shape:
        raise ValueError("Each parent must have shape (M, 3)")


    if rng is None:
        rng = np.random.default_rng()


    # centering the clusters on their centers of mass
    parent1 = parent1 - parent1.mean(axis=0)
    parent2 = parent2 - parent2.mean(axis=0)

    for _ in range(max_attempts):
        # random plane through the origin: uniform random unit normal vector
        normal = rng.normal(size=3)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm

        keep1 = parent1 @ normal >= 0   # atoms of parent 1 on the positive side
        keep2 = parent2 @ normal < 0    # atoms of parent 2 on the negative side

        # only accept if the child has the same number of atoms as the parents
        if keep1.sum() + keep2.sum() == M:
            child = np.vstack((parent1[keep1], parent2[keep2]))
            return child - child.mean(axis=0)  # re-center the offspring

    raise RuntimeError(f"No valid cut found in {max_attempts} attempts")

def random_displacement(cluster, prob=0.25, max_frac=0.1, rng = None):
    '''
    Mutation operator: with probability `prob`, displace every atom by a random
    vector whose length is at most `max_frac` times the average pairwise
    distance of the cluster.

    :param cluster: np-array of shape (M, 3)
    :param prob: float in [0, 1]
        probability that the mutation is applied at all
    :param max_frac: float > 0
        maximum displacement as a fraction of the average pairwise distance
    :param rng: np.random.Generator or None
        random number generator to use
    :returns: np-array of shape (M, 3)
        a new cluster (the input is never modified)
    '''
    if not isinstance(cluster, np.ndarray) or cluster.ndim != 2 or cluster.shape[1] != 3:
        raise ValueError("cluster must be a numpy array of shape (M, 3)")
    if not 0 <= prob <= 1:
        raise ValueError("prob must be between 0 and 1")
    if max_frac <= 0:
        raise ValueError("max_frac must be positive")

    if rng is None:
        rng = np.random.default_rng()
    #only mutate cluster with probability prob.
    if rng.random() >= prob:
        return cluster.copy()  

    # no pairwise distances to define a scale
    M = len(cluster)
    if M < 2:
        return cluster.copy()  

    # average pairwise distance over all unique pairs similair to potential energy calculation 
    diffs = cluster[:, None, :] - cluster[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    avg_dist = dists[np.triu_indices(M, k=1)].mean()
    max_disp = max_frac * avg_dist

    # uniform random direction on the 3d sphere and random length in [0, max_disp]

    # each row of directions is a random vector in 3D where each component is drawn from a standard normal distribution after which it is normalized.
    directions = rng.normal(size=(M, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    #uniform random lengths in [0, max_disp] for each atom
    lengths = max_disp * rng.random(size=(M, 1))

    return cluster + directions * lengths 



""" #removed later,
#perform GA with an initial population of 3 clusters, each containing 10 atoms with 5+1 generations   

rng = np.random.default_rng(42)

initial_population = rcube_initial_population(3, 10, 5.0)
selection       = partial(truncation_selection, potential_func = make_lennard_jones_potential(1,1),cutoff_perc=0.5, N=3, rng=rng)
crossover       = partial(cut_splice,max_attempts=1000, rng=rng)
mutation        = partial(random_displacement, rng=rng, prob=0.1)
local_optimizer = partial(FIRE, potential_derivative_func = make_lennard_jones_potential(1,1), Nmax=10_000, Nmin=5, f_inc=1.1, f_dec=0.5, alpha0=0.1, f_alpha=0.99, err_tol=1e-5, dt=0.002, max_step=0.05)


final = GA(initial_population, selection, crossover, mutation, local_optimizer, 5)
print("Final population:")
for cluster in final:
    print(cluster)  """


    

    


    
