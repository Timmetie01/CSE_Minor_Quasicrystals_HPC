import shap
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
from collections.abc import Callable
from sklearn.ensemble import RandomForestClassifier
from src.main_code.local_optimizers.bfgs import LBFGS
from src.main_code import potentials_calculations as pot
from src.main_code.global_optimizers import basin_hopping as bh
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, average_precision_score, roc_auc_score, classification_report,confusion_matrix)

OPTIMAL_DELTAS = {
    10: 1.6,
    20: 1.8,
    30: 1.8,
    50: 1.9
}

FEATURE_NAMES = [
    "pert_energy_change",
    "minimum_distance_after",
    "force_magnitude_after",
    "displacement_magnitude",
    "coordination_before",
    "coordination_after",
    "neighbor_connections_changed",
    "mean_three_nearest_distances",
    "distance_from_center_before",
    "radial_displacement",
    "local_energy_before",
    "current_total_energy",
    "minimum_distance_before",
    "mean_three_nearest_distances_before",
    "radius_of_gyration_before",
    "second_shell_coordination_before",
    "mean_neighbor_distance_change",
    "nearest_neighbor_alignment",
    "local_distance_std_before",
    "distance_to_neighbor_centroid_after",
]

LJ_GLOBAL_MINIMA = {
    10: -28.422532,
    20: -77.177043,
    30: -128.286571,
    40: -185.249839,
    50: -244.549926,
    60: -305.875476,
    70: -366.892251,
    80: -428.083564,
    90: -492.433908,
    100: -557.039820
}

# Helper functions

def atom_distances(coords: np.ndarray, atom_idx: int) -> np.ndarray:
    """Returns distances from the selected atom to all other atoms."""
    differences = coords - coords[atom_idx]
    distances = np.linalg.norm(differences, axis=1)
    return np.delete(distances, atom_idx)


def atom_pair_energy(coords: np.ndarray, atom_idx: int,
                     position: np.ndarray, potential_function: Callable) -> float:
    """Calculates the selected atom's interaction energy at a given position."""
    other_atoms = np.delete(coords, atom_idx, axis=0)
    distances = np.linalg.norm(other_atoms - position, axis=1)

    return float(sum(potential_function(float(r)) for r in distances))


# Feature 1: log transformed Energy change before local optimization

def pert_energy_change(coords: np.ndarray, pert_coords: np.ndarray,
                       atom_idx: int, potential_function: Callable) -> float:
    """Returns the signed log-transformed energy change before local optimization."""

    old_energy = atom_pair_energy(
        coords, atom_idx, coords[atom_idx], potential_function
    )

    new_energy = atom_pair_energy(
        pert_coords, atom_idx, pert_coords[atom_idx], potential_function
    )

    delta_energy = new_energy - old_energy

    if not np.isfinite(delta_energy):
        raise ValueError("Perturbation energy change is not finite")

    return float(np.sign(delta_energy) * np.log1p(abs(delta_energy)))


# Feature 2: Minimum distance after perturbation

def minimum_distance_after(pert_coords: np.ndarray, atom_idx: int) -> float:
    """Distance from the moved atom to its nearest neighbour after perturbation."""
    distances = atom_distances(pert_coords, atom_idx)
    return float(np.min(distances))


# Feature 3: Force magnitude after perturbation

def force_magnitude_after(
    pert_coords: np.ndarray,
    atom_idx: int,
    potential_derivative_function: Callable = pot.lennard_jones_derivative
) -> float:

    other_atoms = np.delete(pert_coords, atom_idx, axis=0)

    vectors = pert_coords[atom_idx] - other_atoms

    distances = np.linalg.norm(vectors, axis=1)

    force = np.zeros(3)

    for j in range(len(distances)):

        force -= (
            potential_derivative_function(distances[j])
            * vectors[j] / distances[j]
        )

    return float(np.linalg.norm(force))


# Feature 4: Displacement magnitude

def displacement_magnitude(coords: np.ndarray, pert_coords: np.ndarray,
                           atom_idx: int) -> float:
    """Euclidean distance moved by the selected atom."""
    displacement = pert_coords[atom_idx] - coords[atom_idx]
    return float(np.linalg.norm(displacement))


# Feature 5: Coordination number before perturbation

def coordination_before(coords: np.ndarray, atom_idx: int,
                        cutoff: float) -> int:
    """Number of neighbours within the cutoff before perturbation."""
    distances = atom_distances(coords, atom_idx)
    return int(np.count_nonzero(distances < cutoff))


# Feature 6: Coordination number after perturbation

def coordination_after(pert_coords: np.ndarray, atom_idx: int,
                       cutoff: float) -> int:
    """Number of neighbours within the cutoff after perturbation."""
    distances = atom_distances(pert_coords, atom_idx)
    return int(np.count_nonzero(distances < cutoff))


# Feature 7: Number of neighbour connections changed

def neighbor_connections_changed(coords: np.ndarray, pert_coords: np.ndarray,
                                 atom_idx: int, cutoff: float) -> int:
    """Counts neighbours gained or lost due to the perturbation."""
    distances_before = atom_distances(coords, atom_idx)
    distances_after = atom_distances(pert_coords, atom_idx)

    neighbors_before = distances_before < cutoff
    neighbors_after = distances_after < cutoff

    return int(np.count_nonzero(neighbors_before != neighbors_after))


# Feature 8: Mean distance to three nearest neighbours after perturbation

def mean_three_nearest_distances(pert_coords: np.ndarray,
                                 atom_idx: int) -> float:
    """Mean distance to up to three nearest neighbours after perturbation."""
    distances = np.sort(atom_distances(pert_coords, atom_idx))
    return float(np.mean(distances[:3]))


# Feature 9: Distance from cluster centre before perturbation

def distance_from_center_before(coords: np.ndarray, atom_idx: int) -> float:
    """Distance of selected atom from the cluster centre before perturbation."""
    center = np.mean(coords, axis=0)
    return float(np.linalg.norm(coords[atom_idx] - center))


# Feature 10: Radial displacement

def radial_displacement(coords: np.ndarray, pert_coords: np.ndarray,
                        atom_idx: int) -> float:
    """Signed displacement towards or away from the original cluster centre."""
    center = np.mean(coords, axis=0)

    radial_vector = coords[atom_idx] - center
    radial_distance = np.linalg.norm(radial_vector)

    if radial_distance == 0:
        return 0.0

    displacement = pert_coords[atom_idx] - coords[atom_idx]
    radial_direction = radial_vector / radial_distance

    return float(np.dot(displacement, radial_direction))


# Feature 11: Local potential energy before perturbation

def local_energy_before(coords: np.ndarray, atom_idx: int,
                        potential_function: Callable) -> float:
    """Sum of pairwise interaction energies involving the selected atom."""
    return atom_pair_energy(
        coords, atom_idx, coords[atom_idx], potential_function
    )


# Feature 12: Current total cluster energy

def current_total_energy(coords: np.ndarray,
                         potential_function: Callable) -> float:
    """Total potential energy of the current cluster before perturbation."""
    return float(pot.cluster_potential(coords, potential_function))


# Feature 13: Minimum distance before perturbation

def minimum_distance_before(coords: np.ndarray, atom_idx: int) -> float:
    """Distance from the selected atom to its nearest neighbour before perturbation."""
    distances = atom_distances(coords, atom_idx)
    return float(np.min(distances))


# Feature 14: Mean distance to three nearest neighbours before perturbation

def mean_three_nearest_distances_before(coords: np.ndarray,
                                        atom_idx: int) -> float:
    """Mean distance to up to three nearest neighbours before perturbation."""
    distances = np.sort(atom_distances(coords, atom_idx))
    return float(np.mean(distances[:3]))


# Feature 15: Radius of gyration before perturbation

def radius_of_gyration_before(coords: np.ndarray) -> float:
    """Measures the spatial compactness of the current cluster."""
    center = np.mean(coords, axis=0)
    squared_distances = np.sum((coords - center)**2, axis=1)

    return float(np.sqrt(np.mean(squared_distances)))


# Feature 16: Second-shell coordination number

def second_shell_coordination_before(coords: np.ndarray, atom_idx: int,
                                     cutoff: float,
                                     second_cutoff: float) -> int:
    """Counts neighbours between the first and second distance cutoffs."""
    distances = atom_distances(coords, atom_idx)

    return int(np.count_nonzero(
        (distances >= cutoff) & (distances < second_cutoff)
    ))


# Feature 17: Mean distance change to original neighbours

def mean_neighbor_distance_change(coords: np.ndarray, pert_coords: np.ndarray,
                                  atom_idx: int, cutoff: float) -> float:
    """Mean signed distance change to atoms that were originally neighbours."""
    distances_before = atom_distances(coords, atom_idx)
    distances_after = atom_distances(pert_coords, atom_idx)

    neighbors = distances_before < cutoff

    if not np.any(neighbors):
        return 0.0

    distance_changes = distances_after[neighbors] - distances_before[neighbors]

    return float(np.mean(distance_changes))


# Feature 18: Alignment with nearest-neighbour direction

def nearest_neighbor_alignment(coords: np.ndarray, pert_coords: np.ndarray,
                               atom_idx: int) -> float:
    """Cosine of angle between displacement and direction to nearest neighbour."""
    other_atoms = np.delete(coords, atom_idx, axis=0)
    vectors = other_atoms - coords[atom_idx]

    distances = np.linalg.norm(vectors, axis=1)
    nearest_vector = vectors[np.argmin(distances)]

    displacement = pert_coords[atom_idx] - coords[atom_idx]

    norm_displacement = np.linalg.norm(displacement)
    norm_nearest = np.linalg.norm(nearest_vector)

    if norm_displacement == 0 or norm_nearest == 0:
        return 0.0

    return float(
        np.dot(displacement, nearest_vector)
        / (norm_displacement * norm_nearest)
    )


# Feature 19: Standard deviation of original neighbour distances

def local_distance_std_before(coords: np.ndarray, atom_idx: int,
                              cutoff: float) -> float:
    """Measures the variation in distances to the atom's original neighbours."""
    distances = atom_distances(coords, atom_idx)
    neighbors = distances[distances < cutoff]

    if len(neighbors) == 0:
        return 0.0

    return float(np.std(neighbors))


# Feature 20: Distance to centroid of original nearest neighbours

def distance_to_neighbor_centroid_after(coords: np.ndarray,
                                        pert_coords: np.ndarray,
                                        atom_idx: int) -> float:
    """Distance from perturbed atom to centroid of its original three nearest neighbours."""
    other_atoms = np.delete(coords, atom_idx, axis=0)

    distances = atom_distances(coords, atom_idx)
    nearest_indices = np.argsort(distances)[:3]

    neighbor_center = np.mean(other_atoms[nearest_indices], axis=0)

    return float(np.linalg.norm(pert_coords[atom_idx] - neighbor_center))

# Combine all 20 features into a NumPy array
def calculate_features(coords: np.ndarray, pert_coords: np.ndarray,
                       potential_function: Callable,
                       cutoff: float = 1.35,
                       second_cutoff: float = 2.0) -> np.ndarray:
    """
    Calculates 16 features for a single-atom perturbation.

    :param coords: ndarray, shape (N, 3)
        Locally optimized coordinates before perturbation.
    :param pert_coords: ndarray, shape (N, 3)
        Coordinates after perturbation but before local optimization.
    :param potential_function: callable
        Pair potential taking interatomic distance as input.
    :param cutoff: float
        First-shell neighbour cutoff. Default is 1.35.
    :param second_cutoff: float
        Second-shell neighbour cutoff. Default is 2.0.
    :returns: ndarray, shape (20,)
        Numerical feature vector describing the proposed perturbation.
    :raises ValueError:
        If coordinates, cutoffs, or calculated features are invalid,
        or exactly one atom was not moved.
    """

    coords = np.asarray(coords, dtype=float)
    pert_coords = np.asarray(pert_coords, dtype=float)

    # Validate coordinate dimensions
    if coords.shape != pert_coords.shape or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Both coordinate arrays must have shape (N, 3)")

    if len(coords) < 2:
        raise ValueError("At least two atoms are required")

    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(pert_coords)):
        raise ValueError("Coordinates must contain only finite values")

    # Validate neighbour cutoffs
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Cutoff must be finite and positive")

    if not np.isfinite(second_cutoff) or second_cutoff <= cutoff:
        raise ValueError("Second cutoff must be finite and greater than cutoff")

    # Identify the moved atom
    changed_atoms = np.flatnonzero(np.any(coords != pert_coords, axis=1))

    if len(changed_atoms) != 1:
        raise ValueError("Exactly one atom must be perturbed")

    atom_idx = int(changed_atoms[0])

    # Calculate all 20 features
    features = np.array([
        pert_energy_change(coords, pert_coords, atom_idx, potential_function),
        minimum_distance_after(pert_coords, atom_idx),
        force_magnitude_after(pert_coords, atom_idx, potential_function),
        displacement_magnitude(coords, pert_coords, atom_idx),
        coordination_before(coords, atom_idx, cutoff),
        coordination_after(pert_coords, atom_idx, cutoff),
        neighbor_connections_changed(coords, pert_coords, atom_idx, cutoff),
        mean_three_nearest_distances(pert_coords, atom_idx),
        distance_from_center_before(coords, atom_idx),
        radial_displacement(coords, pert_coords, atom_idx),
        local_energy_before(coords, atom_idx, potential_function),
        current_total_energy(coords, potential_function),
        minimum_distance_before(coords, atom_idx),
        mean_three_nearest_distances_before(coords, atom_idx),
        radius_of_gyration_before(coords),
        second_shell_coordination_before(coords, atom_idx, cutoff, second_cutoff),
        mean_neighbor_distance_change(coords, pert_coords, atom_idx, cutoff),
        nearest_neighbor_alignment(coords, pert_coords, atom_idx),
        local_distance_std_before(coords, atom_idx, cutoff),
        distance_to_neighbor_centroid_after(coords, pert_coords, atom_idx),
    ], dtype=float)

    if not np.all(np.isfinite(features)):
        raise ValueError("Feature calculation produced non-finite values")

    return features

# records fearures of single perturbation + additional information after relaxation, returning dictionary. Uses calculate_features function
def create_observation(
    coords,
    pert_coords, 
    new_coords,
    potential_function, 
    best_energy_before,
    iteration, 
    run_id, 
    accepted,
    epsilon=1e-5
):
    # Features calculated from the initial and perturbed coordinates
    features = calculate_features(
        coords, pert_coords, potential_function
    )

    # Energies before and after local optimization
    initial_energy = pot.cluster_potential(coords, potential_function)
    new_energy = pot.cluster_potential(new_coords, potential_function)

    delta_energy = new_energy - initial_energy
    success = int(delta_energy < -epsilon)

    # Construct one complete observation
    observation = dict(zip(FEATURE_NAMES, features))

    observation.update({
        "energy_after_optimization": new_energy,
        "delta_energy": delta_energy,
        "success": success,
        "best_energy_before": best_energy_before,
        "iteration": iteration,
        "run_id": run_id,
        "accepted": accepted
    })

    return observation

# create a random cluster
def random_cluster(n_atoms: int, 
                   characteristic_distance: float = 2**(1/6),
                   density: float = 0.5, 
                   min_separation: float = 0.8,
                   seed: int | None = None) -> np.ndarray:
    """
    Generates random atomic coordinates inside a sphere.

    :param n_atoms: int
        Number of atoms to generate.
    :param characteristic_distance: float
        Preferred interatomic distance of the pair potential.
    :param density: float
        Dimensionless atomic density relative to characteristic_distance.
        Default is 0.5.
    :param min_separation: float
        Minimum allowed separation as a fraction of characteristic_distance.
        Default is 0.8.
    :param seed: int or None
        Random seed for reproducibility.
    :returns: ndarray, shape (n_atoms, 3)
        Random initial atomic coordinates.
    :raises ValueError:
        If input parameters are invalid.
    :raises RuntimeError:
        If the atoms cannot be placed within the allowed attempts.
    """

    if not isinstance(n_atoms, (int, np.integer)) or isinstance(n_atoms, (bool, np.bool_)):
        raise ValueError("n_atoms must be an integer")

    if n_atoms < 1:
        raise ValueError("n_atoms must be positive")

    if not np.isfinite(characteristic_distance) or characteristic_distance <= 0:
        raise ValueError("characteristic_distance must be finite and positive")

    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive")

    if not np.isfinite(min_separation) or min_separation < 0:
        raise ValueError("min_separation must be finite and non-negative")

    rng = np.random.default_rng(seed)

    # Sphere radius based on dimensionless density
    radius = characteristic_distance * (
        3 * n_atoms / (4 * np.pi * density)
    ) ** (1 / 3)

    minimum_distance = min_separation * characteristic_distance

    coords = []

    for i in range(n_atoms):
        for attempt in range(10000):

            # Generate a random point inside the surrounding cube
            point = rng.uniform(-radius, radius, 3)

            # Reject points outside the sphere
            if np.linalg.norm(point) > radius:
                continue

            # Reject points too close to existing atoms
            if all(np.linalg.norm(point - atom) >= minimum_distance
                   for atom in coords):
                coords.append(point)
                break

        else:
            raise RuntimeError(
                "Could not place all atoms; try a lower density"
            )

    return np.array(coords)

#create random perturbation
def random_perturbation(coords: np.ndarray, delta: float = 1.8) -> np.ndarray:
    '''
    Randomly perturbs the position of a single particle in a cluster.

    :param coords: array_like, shape (N, 3)
        Coordinates of N particles in 3D space.
    :param delta: float
        Maximum displacement in each Cartesian direction. default is 1.8
    :returns: ndarray, shape (N, 3)
        Perturbed coordinates of the cluster. Original coordinates are unchanged.
    '''

    new_coords = coords.copy()

    idx = np.random.randint(len(new_coords))
    new_coords[idx] += np.random.uniform(-delta, delta, 3)
    #new_coords += np.random.uniform(-delta, delta, new_coords.shape) 

    return new_coords

# main basin hopping optimiser, can return a list of dictionaries of perturbation observations. Does not have timing functionality
def basin_hopping_opt(input_coords: np.ndarray, potential_function: Callable = pot.lennard_jones_potential,
                      local_optimizer_wrapper: Callable = LBFGS, beta: float =1.0, max_iter: int = 3000,
                      perturbation_type: str = "random", delta: float = 1.8, record_perturbations_features : bool = False,
                      run_id: int | None = None) -> tuple[np.ndarray, float, int, int] | tuple[np.ndarray, float, int, int, list[dict]]:
    
    '''
    Performs global geometry optimization of a particle cluster using the
    basin-hopping algorithm with Metropolis acceptance. Optionally records
    observations from attempted perturbations for supervised ML training.

    :param input_coords: array_like, shape (N, 3)
        Initial coordinates of N particles in 3D space.
        The original coordinates are not modified.
    :param potential_function: callable
        Function calculating the pair potential between two particles.
        Default is Lennard Jones
    :param local_optimizer_wrapper: callable
        Function that takes coordinates and returns locally optimized coordinates.
        Default is LBFGS
    :param beta: float
        Inverse temperature parameter used in the Metropolis criterion (1/(k_b*T)).
        Must be finite and non-negative. Default is 1
    :param max_iter: int
        Maximum number of basin-hopping iterations. Must be a positive integer.
        Default is 3000
    :param perturbation_type: str, optional
        Method used to perturb the cluster. Default is "random".
    :param delta: float, optional
        Maximum displacement in each Cartesian direction for random perturbations.
        Must be finite and positive. Default is 1.8
    :param record_perturbations_features: bool, optional
        Whether to record an observation for every attempted perturbation,
        including accepted and rejected moves. Default is False.
    :param run_id: int or None, optional
        Identifier for the basin-hopping run, used to group observations
        from the same optimization trajectory. Default is None.

    :returns: tuple
        If record_run=False, returns a tuple of (np.ndarray, float, int, int):
            - Coordinates of the lowest-energy configuration.
            - Potential energy of the lowest-energy configuration.
            - Number of iterations performed.
            - Number of accepted moves.

        If record_run=True, additionally returns a list of dictionaries
        containing one observation per attempted perturbation. Each
        observation includes the ML features, energy outcomes, binary
        success label, Metropolis acceptance decision, and run metadata.

    :raises TypeError:
        If beta or delta is not numeric, or max_iter is not an integer.
    :raises ValueError:
        If beta is negative or non-finite, max_iter is not positive,
        perturbation_type is unsupported, delta is missing, non-positive,
        or non-finite, or coordinates do not have shape (N, 3) with N >= 1
        and finite values.
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

    observations = []

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
        if(perturbation_type == "random"):
            pert_coords = random_perturbation(coords, delta)

        # Locally optimize perturbed coordinates
        new_coords = local_optimizer_wrapper(pert_coords)
        new_energy = pot.cluster_potential(new_coords, potential_function)

        delta_energy = new_energy - energy

        # Metropolis acceptance criterion
        accepted = (
            delta_energy < 0.0
            or np.random.rand() < np.exp(-beta * delta_energy)
        )
        if (record_perturbations_features):
            # Record the observation, regardless of acceptance
            obs = create_observation(
                coords=coords,
                pert_coords=pert_coords,
                new_coords=new_coords,
                best_energy_before=best_energy,
                iteration=n_iter,
                run_id=run_id,
                potential_function=potential_function,
                accepted=accepted
            )
            observations.append(obs)

        # Update the current state only if accepted
        if accepted:
            n_accepted += 1
            coords = new_coords
            energy = new_energy

            if energy < best_energy:
                best_coords = coords.copy()
                best_energy = energy

        
    if (record_perturbations_features):
        return best_coords, best_energy, n_iter, n_accepted, observations
    
    return best_coords, best_energy, n_iter, n_accepted

#lean basin hopping algo with timing functionality for benchmarking against clusters consisting of multiples of ten atoms. Stops when target energy is reached
def bh_rec_std_clusters(
    input_coords: np.ndarray,
    local_optimizer_wrapper: Callable = LBFGS,
    beta: float = 1.0,
    delta: float = 1.8,
    tolerance: float = 1e-5
) -> tuple[np.ndarray, float, int, int, float]:

    """
    Performs basin-hopping optimization of a Lennard-Jones cluster until
    the known global minimum energy is reached.

    Uses Metropolis acceptance and random single-atom perturbations.
    The global minimum is obtained from LJ_GLOBAL_MINIMA.

    :param input_coords: np.ndarray, shape (N, 3)
        Initial coordinates of the cluster.
        N must be a multiple of 10 between 10 and 100.
    :param local_optimizer_wrapper: callable
        Function that takes coordinates and returns locally optimized
        coordinates. Default is LBFGS
    :param beta: float
        Inverse temperature used in Metropolis acceptance.
        Default is 1.0.
    :param delta: float
        Maximum displacement in each Cartesian direction.
        Default is 1.8.
    :param tolerance: float
        Maximum absolute energy difference from the reference global
        minimum for convergence. Default is 1e-5.

    :returns: tuple
        - best_coords: np.ndarray
            Coordinates of the global minimum configuration found.
        - best_energy: float
            Energy of that configuration.
        - n_iter: int
            Number of basin-hopping iterations performed.
        - n_accepted: int
            Number of accepted perturbations.
        - total_time: float
            Total execution time in seconds, including initial
            local optimization.

    :raises ValueError:
        If the number of atoms is unsupported.

    :note:
        There is no maximum iteration limit. The algorithm continues
        until the reference energy is reached within tolerance.
    """

    coords = np.array(input_coords, dtype=float, copy=True)

    # Validate cluster size
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N, 3)")

    n_atoms = len(coords)

    if n_atoms not in LJ_GLOBAL_MINIMA:
        raise ValueError(
            "Number of atoms must be a multiple of 10 between 10 and 100."
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError("Coordinates must contain finite values")

    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    target_energy = LJ_GLOBAL_MINIMA[n_atoms]

    # Start timing
    start_time = perf_counter()

    # Initial local optimization
    coords = local_optimizer_wrapper(coords)
    energy = pot.cluster_potential(
        coords, pot.lennard_jones_potential
    )

    best_coords = coords.copy()
    best_energy = energy

    n_iter = 0
    n_accepted = 0

    # Continue until the global minimum is reached
    while abs(best_energy - target_energy) > tolerance:

        n_iter += 1

        # Random perturbation
        pert_coords = random_perturbation(coords, delta)

        # Local optimization
        new_coords = local_optimizer_wrapper(pert_coords)
        new_energy = pot.cluster_potential(
            new_coords, pot.lennard_jones_potential
        )

        delta_energy = new_energy - energy

        # Metropolis acceptance
        accepted = (
            delta_energy < 0.0
            or np.random.rand() < np.exp(-beta * delta_energy)
        )

        if accepted:
            n_accepted += 1
            coords = new_coords
            energy = new_energy

            if energy < best_energy:
                best_coords = coords.copy()
                best_energy = energy

    # Stop timing
    total_time = perf_counter() - start_time

    return best_coords, best_energy, n_iter, n_accepted, total_time

# finds optimal delta based on runtime over a list of possible delta values
def find_optimal_delta(n_atoms: int, k: int = 40) -> float:

    delta_values = [ 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]

    best_delta = None
    best_time = float("inf")

    def local_optimizer(coords):
        return LBFGS(
            coords,
            pot.lennard_jones_derivative,
            print_convergence=False
        )

    for delta in delta_values:

        run_times = []

        for i in range(k):

            seed = n_atoms * 1000 * k + i
            np.random.seed(seed)

            initial_coords = random_cluster(
                n_atoms=n_atoms,
                characteristic_distance=2**(1/6),
                density=0.5,
                min_separation=0.8,
                seed=seed
            )

            _, _, _, _, total_time = bh_rec_std_clusters(
                input_coords=initial_coords,
                local_optimizer_wrapper=local_optimizer,
                beta=1.0,
                delta=delta
            )

            print(f"run {i} done for delta = {delta}, with time = {total_time}")
            run_times.append(total_time)

        median_time = np.median(run_times)

        if median_time < best_time:
            best_time = median_time
            best_delta = delta

        print(f"median for delta {delta} = {median_time}")

    return best_delta

def benchmark_bh(n_atoms: int, k: int = 40):

    base_dir = Path("data/records/LJ/1.0_1.0/methods/basin_hopping")
    data_dir = base_dir / "run_times_for_global_min"
    plots_dir = base_dir / "plots_run_times"

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i in range(k):

        seed = n_atoms * 1000 * k + i
        np.random.seed(seed)

        initial_coords = random_cluster(
            n_atoms=n_atoms,
            characteristic_distance=2**(1/6),
            density=0.5,
            min_separation=0.8,
            seed=seed
        )

        best_coords, best_energy, n_iter, n_accepted, total_time = (
            bh_rec_std_clusters(
                input_coords=initial_coords,
                local_optimizer_wrapper=lambda coords: LBFGS(
                    coords,
                    pot.lennard_jones_derivative,
                    print_convergence=False
                ),
                beta=1.0,
                delta=1.8
            )
        )

        records.append({
            "run_id": i,
            "best_energy": best_energy,
            "iterations": n_iter,
            "accepted_iterations": n_accepted,
            "time_seconds": total_time
        })

        # Save results after every run
        df = pd.DataFrame(records)
        df.to_csv(data_dir / f"{n_atoms} atoms.csv", index=False)

        print(f"LJ{n_atoms} | Run {i+1}/{k} | Time: {total_time:.2f}s")

    # Plot distribution of execution times
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["time_seconds"],
        bins="auto",
        color="steelblue",
        edgecolor="white"
    )

    ax.axvline(
        df["time_seconds"].mean(),
        color="darkred",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {df['time_seconds'].mean():.2f}s"
    )

    ax.set_title(f"LJ{n_atoms} — Time to Global Minimum")
    ax.set_xlabel("Time to global minimum (seconds)")
    ax.set_ylabel("Number of runs")

    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(plots_dir / f"{n_atoms} atoms.png", dpi=200)
    plt.close(fig)

    print(f"LJ{n_atoms} complete. Dataset and plot saved.")

# generate basin hopping data for ml training and records it in data base.
def generate_data(n_atoms : int, n_bh_runs : int = 10, n_bh_iter : int = 500, output_dir : str = "data/records/LJ/1.0_1.0/methods/ml_assisted_bh/training_datasets" ):
    output_dir = Path(output_dir)
    def local_optimizer(coords):
        return LBFGS(coords, pot.lennard_jones_derivative, print_convergence=False)

    obs_list = []

    for i in range(n_bh_runs):

        initial_coords = random_cluster(
            n_atoms=n_atoms,
            characteristic_distance=2**(1/6),
            density=0.5,
            min_separation=0.8,
            seed=i*10 + i
        )


        best_coords, best_energy, n_iter, n_accepted, observations = basin_hopping_opt(
            input_coords=initial_coords,
            potential_function=pot.lennard_jones_potential,
            local_optimizer_wrapper=local_optimizer,
            beta=1.0,
            max_iter=n_bh_iter,
            delta=1.8,
            record_perturbations_features=True,
            run_id= i
        )
        obs_list.extend(observations)
        print(
        f"Run {i+1}/{n_bh_runs} | "
        f"Best energy: {best_energy:.6f} | "
        f"Accepted: {n_accepted}/{n_iter}")

        
    df = pd.DataFrame(obs_list)
    df.to_csv(output_dir / f"ml_bh_{n_atoms}atoms_{n_bh_runs}runs_{n_bh_iter}iter.csv", index=False)

    print("File added to " + str(output_dir))

def train_rf_model(
    n_atoms: int,
    n_bh_runs: int,
    n_bh_iter: int,
    data_file_path: str = "data/records/LJ/1.0_1.0/methods/ml_assisted_bh/training_datasets",
    n_estimators: int = 200,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
):
    
    if(data_file_path == "data/records/LJ/1.0_1.0/methods/ml_assisted_bh/training_datasets"):
        folder_path = Path(data_file_path)
        file_path = folder_path / f"ml_bh_{n_atoms}atoms_{n_bh_runs}runs_{n_bh_iter}iter.csv"
    else:
        file_path = Path(data_file_path)

    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)
    print("Number of runs:", df["run_id"].nunique())
    print("Success rate:", df["success"].mean())
    print("Acceptance rate:", df["accepted"].mean())
    print("Missing values:", df.isna().sum().sum())

    print("Near-zero energy changes:",
        np.mean(np.abs(df["delta_energy"]) < 1e-5))

    print("\nSuccess rate per run:")
    print(df.groupby("run_id")["success"].mean())

    # Select inputs and dependent variable
    X = df[FEATURE_NAMES]
    y = df["success"].astype(int)

    # Split by independent basin-hopping run
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=df["run_id"])
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("Training success rate:", y_train.mean())
    print("Testing success rate:", y_test.mean())

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both classes.")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test, X_train, y_train

# input model and test, train sets -> loads feature importance plot into data base
def feature_importance_rf_model(
    model,
    X_test,
    Y_test,
    X_train,
    n_atoms: int,
    n_bh_runs: int,
    n_bh_iter: int,
    data_folder_path: str = "data/records/LJ/1.0_1.0/methods/ml_assisted_bh"
):

    # Create output directories
    base_dir = Path(data_folder_path)

    importance_dir = base_dir / "feature_importance"
    probability_dir = base_dir / "pred_prob_vs_realised"

    importance_dir.mkdir(parents=True, exist_ok=True)
    probability_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{n_atoms}atoms_{n_bh_runs}runs_{n_bh_iter}iter"

    # Predicted probabilities vs realised outcomes
    y_prob = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "predicted_probability": y_prob,
        "actual_outcome": Y_test.to_numpy()
    })

    results = results.sort_values(
        by="predicted_probability",
        ascending=False
    )

    results.to_csv(
        probability_dir / f"{file_name}.csv",
        index=False
    )

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    feature_importance.to_csv(
        importance_dir / f"{file_name}.csv",
        index=False
    )

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color="steelblue"
    )

    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title(
        f"LJ{n_atoms} - Random Forest Feature Importance"
    )

    fig.tight_layout()

    fig.savefig(
        importance_dir / f"{file_name}.png",
        dpi=300
    )

    plt.close(fig)

    print(f"LJ{n_atoms}: Model analysis saved.")


# main basin hopping optimiser, can return a list of dictionaries of perturbation observations. Does not have timing functionality
def rf_basin_hopping_opt(
    input_coords: np.ndarray,
    model,
    potential_function: Callable = pot.lennard_jones_potential,
    local_optimizer_wrapper: Callable = LBFGS,
    beta: float = 1.0,
    max_iter: int = 3000,
    delta: float = 1.8,
    k: int = 20,
    record_run: bool = False,
    run_id: int | None = None
) -> tuple[np.ndarray, float, int, int, float, float] | tuple[np.ndarray, float, int, int, float, float, list[dict]]:

    '''
    Performs ML-assisted global geometry optimization of a particle
    cluster using probability-weighted perturbation selection and
    Metropolis acceptance.

    At each iteration, k random perturbation candidates are generated
    from the current configuration. A trained classification model
    predicts the probability of success for each candidate.

    One candidate is selected with probability proportional to its
    predicted probability of success and subsequently locally optimized.

    :param input_coords: array_like, shape (N, 3)
        Initial coordinates of N particles in 3D space.
        The original coordinates are not modified.

    :param model:
        Trained classification model predicting the probability
        of a successful perturbation.

    :param potential_function: callable
        Function calculating the pair potential between two particles.
        Default is Lennard Jones.

    :param local_optimizer_wrapper: callable
        Function that takes coordinates and returns locally
        optimized coordinates. Default is LBFGS.

    :param beta: float
        Inverse temperature parameter used in the Metropolis
        acceptance criterion. Must be finite and non-negative.
        Default is 1.0.

    :param max_iter: int
        Maximum number of basin-hopping iterations.
        Must be a positive integer. Default is 3000.

    :param delta: float
        Maximum displacement in each Cartesian direction.
        Must be finite and positive. Default is 1.8.

    :param k: int
        Number of random perturbation candidates generated
        per basin-hopping iteration.
        Must be a positive integer. Default is 20.

    :param record_run: bool
        Whether to record an observation for each selected
        perturbation, regardless of acceptance.
        Default is False.

    :param run_id: int or None
        Identifier for the basin-hopping trajectory.
        Default is None.

    :returns: tuple
        If record_run=False, returns:
            - Best coordinates.
            - Best energy.
            - Number of iterations performed.
            - Number of accepted moves.
            - Total execution time in seconds.
            - Time to reach the best configuration in seconds.

        If record_run=True, additionally returns:
            - List of recorded observations.
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

    # Validate delta
    if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, (int, float, np.integer, np.floating)):
        raise TypeError("delta must be a numeric value")

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")

    # Validate k
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Validate model classes
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Model must be trained on classes 0 and 1")

    # Convert coordinates and protect original input
    coords = np.array(input_coords, dtype=float, copy=True)

    # Validate coordinate dimensions
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N, 3)")

    if len(coords) == 0:
        raise ValueError("At least one particle is required")

    if not np.all(np.isfinite(coords)):
        raise ValueError("Coordinates must contain only finite values")

    observations = []

    # Start timer
    start_time = perf_counter()

    # Initial local optimization
    coords = local_optimizer_wrapper(coords)
    energy = pot.cluster_potential(coords, potential_function)

    best_coords = coords.copy()
    best_energy = energy

    time_to_best = perf_counter() - start_time

    n_iter = 0
    n_accepted = 0

    while n_iter < max_iter:

        n_iter += 1

        candidates = []
        candidate_features = []

        # Generate k random perturbation candidates
        for _ in range(k):

            candidate = random_perturbation(coords, delta)

            features = calculate_features(
                coords,
                candidate,
                potential_function
            )

            candidates.append(candidate)
            candidate_features.append(features)

        # Construct feature matrix
        X_candidates = pd.DataFrame(
            candidate_features,
            columns=FEATURE_NAMES
        )

        # Predict probability of success for each candidate
        probabilities = model.predict_proba(X_candidates)[:, 1]

        # Normalize probabilities for weighted selection
        if probabilities.sum() > 0:

            selection_probabilities = probabilities / probabilities.sum()

        else:

            # Uniform selection if all predicted probabilities are zero
            selection_probabilities = np.ones(k) / k

        # Select candidate using probability-weighted sampling
        selected_idx = np.random.choice(
            k,
            p=selection_probabilities
        )

        #selected_idx = np.argmax(probabilities)

        pert_coords = candidates[selected_idx]

        # Locally optimize only the selected candidate
        new_coords = local_optimizer_wrapper(pert_coords.copy())
        new_energy = pot.cluster_potential(new_coords, potential_function)

        delta_energy = new_energy - energy

        # Metropolis acceptance criterion
        accepted = (
            delta_energy < 0.0
            or np.random.rand() < np.exp(-beta * delta_energy)
        )

        # Record the selected perturbation, regardless of acceptance
        if record_run:

            obs = create_observation(
                coords=coords,
                pert_coords=pert_coords,
                new_coords=new_coords,
                potential_function=potential_function,
                best_energy_before=best_energy,
                iteration=n_iter,
                run_id=run_id,
                accepted=accepted
            )

            observations.append(obs)

        # Update the current state only if accepted
        if accepted:

            n_accepted += 1
            coords = new_coords
            energy = new_energy

            # Update best configuration and time to best
            if energy < best_energy:

                best_coords = coords.copy()
                best_energy = energy

                time_to_best = perf_counter() - start_time

    # Calculate total execution time
    total_time = perf_counter() - start_time

    if record_run:

        return (
            best_coords,
            best_energy,
            n_iter,
            n_accepted,
            total_time,
            time_to_best,
            observations
        )

    return (
        best_coords,
        best_energy,
        n_iter,
        n_accepted,
        total_time,
        time_to_best
    )

#lean rf basin hopping algo with timing functionality for benchmarking against clusters consisting of multiples of ten atoms. Stops when target energy is reached
def rf_bh_rec_std_clusters(
    input_coords: np.ndarray,
    model,
    local_optimizer_wrapper: Callable = LBFGS,
    beta: float = 1.0,
    delta: float = 1.8,
    k: int = 20,
    tolerance: float = 1e-5
) -> tuple[np.ndarray, float, int, int, float]:

    """
    Performs ML-assisted basin-hopping optimization of a Lennard-Jones
    cluster until the known global minimum energy is reached.

    At each iteration, k random single-atom perturbations are generated.
    A trained Random Forest predicts their probabilities of success.
    One candidate is selected with probability proportional to its
    predicted probability of success.

    :param input_coords: np.ndarray, shape (N, 3)
        Initial coordinates of the cluster.
        N must be a multiple of 10 between 10 and 100.
    :param model:
        Trained Random Forest classifier.
    :param local_optimizer_wrapper: callable
        Function that takes coordinates and returns locally optimized
        coordinates. Default is LBFGS.
    :param beta: float
        Inverse temperature used in Metropolis acceptance.
        Default is 1.0.
    :param delta: float
        Maximum displacement in each Cartesian direction.
        Default is 1.8.
    :param k: int
        Number of perturbation candidates generated per iteration.
        Default is 10.
    :param tolerance: float
        Maximum absolute energy difference from the reference global
        minimum for convergence. Default is 1e-5.

    :returns: tuple
        - best_coords: np.ndarray
            Coordinates of the global minimum configuration found.
        - best_energy: float
            Energy of that configuration.
        - n_iter: int
            Number of basin-hopping iterations performed.
        - n_accepted: int
            Number of accepted perturbations.
        - total_time: float
            Total execution time in seconds, including initial local
            optimization, feature extraction and ML predictions.

    :note:
        There is no maximum iteration limit. The algorithm continues
        until the reference energy is reached within tolerance.
    """

    coords = np.array(input_coords, dtype=float, copy=True)

    # Validate cluster size
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N, 3)")

    n_atoms = len(coords)

    if n_atoms not in LJ_GLOBAL_MINIMA:
        raise ValueError(
            "Number of atoms must be a multiple of 10 between 10 and 100."
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError("Coordinates must contain finite values")

    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")

    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Model must be trained on classes 0 and 1")

    target_energy = LJ_GLOBAL_MINIMA[n_atoms]

    # Start timing
    start_time = perf_counter()

    # Initial local optimization
    coords = local_optimizer_wrapper(coords)
    energy = pot.cluster_potential(
        coords, pot.lennard_jones_potential
    )

    best_coords = coords.copy()
    best_energy = energy

    n_iter = 0
    n_accepted = 0

    # Continue until the global minimum is reached
    while abs(best_energy - target_energy) > tolerance:

        n_iter += 1

        candidates = []
        candidate_features = []

        # Generate k random perturbation candidates
        for _ in range(k):

            candidate = random_perturbation(coords, delta)

            features = calculate_features(
                coords,
                candidate,
                pot.lennard_jones_potential
            )

            candidates.append(candidate)
            candidate_features.append(features)

        # Construct feature matrix
        X_candidates = pd.DataFrame(
            candidate_features,
            columns=FEATURE_NAMES
        )

        # Predict probability of success
        probabilities = model.predict_proba(X_candidates)[:, 1]

        #  Normalize probabilities
        if probabilities.sum() > 0:
            selection_probabilities = probabilities / probabilities.sum()
        else:
            selection_probabilities = np.ones(k) / k

        # Probability-weighted candidate selection
        selected_idx = np.random.choice(
            k,
            p=selection_probabilities
        )

        #selected_idx = np.argmax(probabilities) # <- this can be changed
        pert_coords = candidates[selected_idx]

        # Local optimization of selected candidate
        new_coords = local_optimizer_wrapper(pert_coords)
        new_energy = pot.cluster_potential(
            new_coords, pot.lennard_jones_potential
        )

        delta_energy = new_energy - energy

        # Metropolis acceptance
        accepted = (
            delta_energy < 0.0
            or np.random.rand() < np.exp(-beta * delta_energy)
        )

        if accepted:
            n_accepted += 1
            coords = new_coords
            energy = new_energy

            if energy < best_energy:
                best_coords = coords.copy()
                best_energy = energy

    # Stop timing
    total_time = perf_counter() - start_time

    return best_coords, best_energy, n_iter, n_accepted, total_time


# runs the lean rf_bh_rec_std_clusters basin hopping and records the run times, creates a dataset and a plot, both stored in data base
def benchmark_rf_bh(
    n_atoms: int,
    model,
    k: int = 20,
    n_runs: int = 40,
    delta: float = 1.8,
    beta: float = 1.0,
    tolerance: float = 1e-5
):

    # Validate inputs
    if n_atoms not in LJ_GLOBAL_MINIMA:
        raise ValueError("Global minimum not available for this cluster size")

    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    # Create output directories
    base_dir = Path(
        "data/records/LJ/1.0_1.0/methods/ml_assisted_bh"
    )

    data_dir = base_dir / "run_times_for_global_min"
    plots_dir = base_dir / "plots_run_times"

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{n_atoms}atoms_{k}candidates_{n_runs}runs"

    records = []

    # Define local optimizer
    def local_optimizer(coords):
        return LBFGS(
            coords,
            pot.lennard_jones_derivative,
            print_convergence=False
        )

    # Perform independent ML-assisted basin-hopping runs
    for i in range(n_runs):

        seed = n_atoms * 33 + i*k + i
        np.random.seed(seed)

        initial_coords = random_cluster(
            n_atoms=n_atoms,
            characteristic_distance=2**(1/6),
            density=0.5,
            min_separation=0.8,
            seed=seed
        )

        best_coords, best_energy, n_iter, n_accepted, total_time = (
            rf_bh_rec_std_clusters(
                input_coords=initial_coords,
                model=model,
                local_optimizer_wrapper=local_optimizer,
                beta=beta,
                delta=delta,
                k=k,
                tolerance=tolerance
            )
        )

        # Record results
        records.append({
            "run_id": i,
            "best_energy": best_energy,
            "iterations": n_iter,
            "accepted_iterations": n_accepted,
            "time_seconds": total_time
        })

        # Save results after each completed run
        df = pd.DataFrame(records)

        df.to_csv(
            data_dir / f"{file_name}.csv",
            index=False
        )

        print(
            f"LJ{n_atoms} | "
            f"k={k} | "
            f"Run {i+1}/{n_runs} | "
            f"Time: {total_time:.4f}s | "
            f"Iterations: {n_iter}",
            flush=True
        )

    # Calculate summary statistics
    mean_time = df["time_seconds"].mean()
    median_time = df["time_seconds"].median()
    std_time = df["time_seconds"].std()

    print(f"\nLJ{n_atoms} | k={k}")
    print(f"Mean runtime: {mean_time:.4f}s")
    print(f"Median runtime: {median_time:.4f}s")
    print(f"Standard deviation: {std_time:.4f}s")
    print(f"Mean iterations: {df['iterations'].mean():.2f}")
    print(f"Median iterations: {df['iterations'].median():.2f}")

    # Plot runtime distribution
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["time_seconds"],
        bins="auto",
        color="steelblue",
        edgecolor="white"
    )

    ax.axvline(
        mean_time,
        color="darkred",
        linestyle="--",
        label=f"Mean: {mean_time:.2f}s"
    )

    ax.axvline(
        median_time,
        color="black",
        linestyle="--",
        label=f"Median: {median_time:.2f}s"
    )

    ax.set_title(
        f"LJ{n_atoms} - ML-Assisted Basin Hopping (k={k})"
    )

    ax.set_xlabel("Time to global minimum (seconds)")
    ax.set_ylabel("Number of runs")

    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        plots_dir / f"{file_name}.png",
        dpi=300
    )

    plt.close(fig)

    print(f"\nLJ{n_atoms} complete. Dataset and plot saved.")

    
# calculate compute time of all features, using 100 runs per feature, records data and plots feature computaion time. stores both in data base.
def benchmark_feature_cost(n_atoms: int):

    """
    Benchmarks the computational cost of all 20 ML features
    by evaluating each feature 100 times.

    Features are ranked from most to least computationally
    expensive, and the results are saved to a CSV file.

    :param n_atoms: int
        Number of atoms in the Lennard-Jones cluster.
    """

    n_iterations = 100
    delta = 1.8
    cutoff = 1.5
    second_cutoff = 2.0

    potential_function = pot.lennard_jones_potential

    # Generate initial cluster
    coords = random_cluster(
        n_atoms=n_atoms,
        characteristic_distance=2**(1/6),
        density=0.5,
        min_separation=0.8,
        seed=42
    )

    # Locally optimize initial cluster
    coords = LBFGS(
        coords,
        pot.lennard_jones_derivative,
        print_convergence=False
    )

    # Generate 100 random perturbations before timing
    rng = np.random.default_rng(42)

    candidates = []

    for _ in range(n_iterations):

        atom_idx = rng.integers(n_atoms)

        pert_coords = coords.copy()
        pert_coords[atom_idx] += rng.uniform(-delta, delta, 3)

        candidates.append((pert_coords, atom_idx))

    # Define feature calculations
    features = {
        "pert_energy_change":
            lambda p, i: pert_energy_change(
                coords, p, i, potential_function
            ),

        "minimum_distance_after":
            lambda p, i: minimum_distance_after(p, i),

        "force_magnitude_after":
            lambda p, i: force_magnitude_after(
                p, i, potential_function
            ),

        "displacement_magnitude":
            lambda p, i: displacement_magnitude(coords, p, i),

        "coordination_before":
            lambda p, i: coordination_before(coords, i, cutoff),

        "coordination_after":
            lambda p, i: coordination_after(p, i, cutoff),

        "neighbor_connections_changed":
            lambda p, i: neighbor_connections_changed(
                coords, p, i, cutoff
            ),

        "mean_three_nearest_distances":
            lambda p, i: mean_three_nearest_distances(p, i),

        "distance_from_center_before":
            lambda p, i: distance_from_center_before(coords, i),

        "radial_displacement":
            lambda p, i: radial_displacement(coords, p, i),

        "local_energy_before":
            lambda p, i: local_energy_before(
                coords, i, potential_function
            ),

        "current_total_energy":
            lambda p, i: current_total_energy(
                coords, potential_function
            ),

        "minimum_distance_before":
            lambda p, i: minimum_distance_before(coords, i),

        "mean_three_nearest_distances_before":
            lambda p, i: mean_three_nearest_distances_before(
                coords, i
            ),

        "radius_of_gyration_before":
            lambda p, i: radius_of_gyration_before(coords),

        "second_shell_coordination_before":
            lambda p, i: second_shell_coordination_before(
                coords, i, cutoff, second_cutoff
            ),

        "mean_neighbor_distance_change":
            lambda p, i: mean_neighbor_distance_change(
                coords, p, i, cutoff
            ),

        "nearest_neighbor_alignment":
            lambda p, i: nearest_neighbor_alignment(
                coords, p, i
            ),

        "local_distance_std_before":
            lambda p, i: local_distance_std_before(
                coords, i, cutoff
            ),

        "distance_to_neighbor_centroid_after":
            lambda p, i: distance_to_neighbor_centroid_after(
                coords, p, i
            )
    }

    records = []

    # Benchmark each feature independently
    for feature_name, feature_function in features.items():

        # Warm-up call outside the timed loop
        feature_function(*candidates[0])

        # Start timer
        start_time = perf_counter()

        for pert_coords, atom_idx in candidates:

            feature_function(pert_coords, atom_idx)

        # Stop timer
        total_time = perf_counter() - start_time

        records.append({
            "feature": feature_name,
            "total_time_seconds": total_time,
            "mean_time_seconds": total_time / n_iterations,
            "mean_time_microseconds": (
                total_time / n_iterations * 1e6
            )
        })

    # Create results DataFrame
    df = pd.DataFrame(records)

    # Rank from most to least computationally expensive
    df = df.sort_values(
        by="total_time_seconds",
        ascending=False
    ).reset_index(drop=True)

    df.insert(0, "rank", df.index + 1)

    # Calculate percentage of total feature computation time
    df["percentage_of_total_time"] = (
        df["total_time_seconds"]
        / df["total_time_seconds"].sum()
        * 100
    )

    # Create output directory
    output_dir = Path(
        "data/records/LJ/1.0_1.0/methods/"
        "ml_assisted_bh/compute_expense_features"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    file_path = output_dir / f"{n_atoms} atom cluster.csv"

    df.to_csv(file_path, index=False)

    # Plot feature computational expense
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(
        df["feature"],
        df["percentage_of_total_time"],
        color="steelblue"
    )

    ax.invert_yaxis()

    ax.set_xlabel("Percentage of total feature computation time (%)")
    ax.set_ylabel("Feature")
    ax.set_title(f"LJ{n_atoms} - Feature Computational Expense")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{n_atoms} atom cluster.png",
        dpi=300
    )

    plt.close(fig)

    print(df.to_string(index=False))

    print(f"\nResults saved to {file_path}")

    return df



if __name__ == "__main__":
    # to do: 


    # 5. After creating dataset, train first rf model. Look into parameter tuning
    # 6. Try XGBoost and parameter tuning.
    # 7. Look in to SHAP
    def local_optimizer_wrapper(coords):
        return LBFGS(coords, print_convergence = False)
    
    rf_model_30, X_test, Y_test, X_train, Y_train = train_rf_model(30, 10, 1500)
    
    #benchmark_ml_bh(30, rf_model_30, k = 35, n_runs = 100)
    explainer = shap.TreeExplainer(rf_model_30)

    shap_test = explainer(X_test)[:, :, 1]


    # Define output directory
    output_dir = Path(
        "data/records/LJ/1.0_1.0/methods/ml_assisted_bh/feature_importance"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate beeswarm plot
    shap.plots.beeswarm(
        shap_test,
        max_display=20,
        plot_size=(14, 9),
        show=False
    )

    plt.subplots_adjust(left=0.35)
    plt.title("LJ30 - SHAP Feature Importance")

    plt.show()
    plt.close()
    








   

    







    



    
    



    





        
    

