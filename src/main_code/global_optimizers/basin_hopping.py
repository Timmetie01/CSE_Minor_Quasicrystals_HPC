import numpy as np
from collections.abc import Callable
from src.main_code import potentials_calculations as pot
from sklearn.ensemble import RandomForestClassifier

def basin_hopping(input_coords: np.ndarray, potential_function: Callable,
                  local_optimizer_wrapper: Callable, beta: float, max_iter: int,
                  perturbation_type: str = "random", delta: float | None = None) -> tuple[np.ndarray, float, int, int]:
    '''
    Performs global geometry optimization of a particle cluster using the
    basin-hopping algorithm with Metropolis acceptance.

    :param input_coords: array_like, shape (N, 3)
        Initial coordinates of N particles in 3D space.
        The original coordinates are not modified.
    :param potential_function: callable
        Function calculating the pair potential between two particles.
    :param local_optimizer_wrapper: callable
        Function that takes coordinates and returns locally optimized coordinates.
    :param beta: float
        Inverse temperature parameter used in the Metropolis criterion (1/(k_b*T)).
        Must be finite and non-negative.
    :param max_iter: int
        Maximum number of basin-hopping iterations. Must be a positive integer.
    :param perturbation_type: str, optional
        Method used to perturb the cluster. Default is "random".
    :param delta: float, optional
        Maximum displacement in each Cartesian direction for random perturbations.
        Must be finite and positive when using random perturbation.
    :returns: tuple of (np.ndarray, float, int, int)
        Coordinates of the lowest-energy configuration, its potential energy,
        number of iterations performed, and number of accepted moves.
    :raises TypeError:
        If beta or delta is not numeric, or max_iter is not an integer.
    :raises ValueError:
        If beta is negative or non-finite, max_iter is not positive,
        perturbation_type is unsupported, delta is missing, non-positive,
        or non-finite, or coordinates do not have shape (N, 3) with N >= 1.
    '''

    # Validate beta
    if isinstance(beta, (bool, np.bool_)) or not isinstance(beta, (int, float, np.integer, np.floating)):
        raise TypeError("beta must be a numeric value")

    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")

    # Validate max_iter
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be an integer")

    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    # Validate perturbation type
    if perturbation_type != "random":
        raise ValueError("Unsupported perturbation type")

    # Validate delta
    if delta is None:
        raise ValueError("delta must be provided for random perturbation")

    if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, (int, float, np.integer, np.floating)):
        raise TypeError("delta must be a numeric value")

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")

    # Convert coordinates to a floating-point array and protect original input
    coords = np.array(input_coords, dtype=float, copy=True)

    # Validate coordinate dimensions
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N, 3)")

    if len(coords) == 0:
        raise ValueError("At least one particle is required")

    if not np.all(np.isfinite(coords)):
        raise ValueError("Coordinates must contain only finite values")

    # Initial local optimization
    coords = local_optimizer_wrapper(coords)
    energy = pot.cluster_potential(coords, potential_function)

    best_coords = coords.copy()
    best_energy = energy

    n_iter = 0
    n_accepted = 0

    while n_iter < max_iter:
        n_iter += 1

        # Perturb current coordinates
        pert_coords = random_perturbation(coords, delta)

        # Locally optimize perturbed coordinates
        new_coords = local_optimizer_wrapper(pert_coords)
        new_energy = pot.cluster_potential(new_coords, potential_function)

        delta_energy = new_energy - energy

        # Metropolis acceptance criterion
        if delta_energy < 0.0 or np.random.rand() < np.exp(-beta * delta_energy):
            n_accepted += 1
            coords = new_coords
            energy = new_energy

            # Update lowest-energy configuration
            if energy < best_energy:
                best_coords = coords.copy()
                best_energy = energy

    return best_coords, best_energy, n_iter, n_accepted


def random_perturbation(coords: np.ndarray, delta: float) -> np.ndarray:
    '''
    Randomly perturbs the position of a single particle in a cluster.

    :param coords: array_like, shape (N, 3)
        Coordinates of N particles in 3D space.
    :param delta: float
        Maximum displacement in each Cartesian direction.
    :returns: ndarray, shape (N, 3)
        Perturbed coordinates of the cluster. Original coordinates are unchanged.
    '''

    new_coords = coords.copy()

    idx = np.random.randint(len(new_coords))
    new_coords[idx] += np.random.uniform(-delta, delta, 3)
    #new_coords += np.random.uniform(-delta, delta, new_coords.shape) 

    return new_coords

def generate_candidates(coords: np.ndarray, delta: float, n_candidates: int) -> list:
    '''
    Generates a list of candidate configurations by randomly perturbing the input coordinates.

    :param coords: array_like, shape (N, 3)
        Coordinates of N particles in 3D space.
    :param delta: float
        Maximum displacement in each Cartesian direction for random perturbations.
    :param n_candidates: int
        Number of candidate configurations to generate.
    :returns: list of ndarray
        List containing n_candidates perturbed coordinate arrays.
    '''
    candidates = []
    for _ in range(n_candidates):
        perturbed_coords = random_perturbation(coords, delta)
        candidates.append(perturbed_coords)
    return candidates

def calculate_features(coords: np.ndarray, candidate: np.ndarray, potential_function: Callable, current_energy: float):
    '''
    Calculates features for a candidate configuration based on the current coordinates and energy.
    '''
    difference = candidate - coords
    idx = np.argmax(np.linalg.norm(difference, axis=1))

    displacement = np.linalg.norm(difference[idx])
    
    distances = np.linalg.norm(candidate - candidate[idx], axis=1)
    distances = np.delete(distances, idx)
    
    min_distance = np.min(distances) if len(distances) > 0 else 0.0
    max_distance = np.max(distances) if len(distances) > 0 else 0.0
    
    candidate_energy = pot.cluster_potential(candidate, potential_function)
    energy_change = candidate_energy - current_energy

    return np.array([displacement, min_distance, max_distance, energy_change])

def train_ml_model(coords: np.ndarray, current_energy: float,
                   potential_function: Callable,
                   local_optimizer_wrapper: Callable,
                   delta: float, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    '''
    
    X = []
    y = []
    
    for _ in range(n_samples):
        candidate = random_perturbation(coords, delta)
    
        features = calculate_features(coords, candidate, potential_function, current_energy)
        X.append(features)
        
        opt_coords = local_optimizer_wrapper(candidate)
        opt_energy = pot.cluster_potential(opt_coords, potential_function)
        
        # CHANGE THIS (LOSS OF INFORMATION)
        y.append(1 if opt_energy < current_energy else 0)

    return np.array(X), np.array(y)

def ml_basin_hopping(input_coords: np.ndarray, potential_function: Callable,
                  local_optimizer_wrapper: Callable, beta: float, max_iter: int,
                delta: float, n_candidates: int) -> tuple[np.ndarray, float, int, int]:
    '''
    
    '''
    # add checks
    
    coords = np.array(input_coords, dtype=float, copy=True)
    coords = local_optimizer_wrapper(coords)
    
    energy = pot.cluster_potential(coords, potential_function)
    
    X, y = train_ml_model(coords, energy, potential_function, local_optimizer_wrapper, delta, n_candidates)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    