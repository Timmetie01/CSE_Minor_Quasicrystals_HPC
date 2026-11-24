import numpy as np
from collections.abc import Callable
from src.main_code import potentials_calculations as pot
from sklearn.ensemble import RandomForestClassifier
from src.main_code.global_optimizers import basin_hopping as bh


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

def force_magnitude_after(pert_coords: np.ndarray, atom_idx: int,
                          potential_function: Callable,
                          h: float = 1e-5) -> float:
    """Force magnitude on the moved atom, calculated using central differences."""
    position = pert_coords[atom_idx].copy()
    gradient = np.zeros(3)

    for axis in range(3):
        position_plus = position.copy()
        position_minus = position.copy()

        position_plus[axis] += h
        position_minus[axis] -= h

        energy_plus = atom_pair_energy(
            pert_coords, atom_idx, position_plus, potential_function
        )
        energy_minus = atom_pair_energy(
            pert_coords, atom_idx, position_minus, potential_function
        )

        gradient[axis] = (energy_plus - energy_minus) / (2 * h)

    return float(np.linalg.norm(gradient))


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
    Calculates 20 features for a single-atom perturbation.

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



def basin_hopping_opt(input_coords: np.ndarray, potential_function: Callable,
                      local_optimizer_wrapper: Callable, beta: float, max_iter: int,
                      perturbation_type: str = "random", delta: float | None = None, record_run : bool = False,
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
        Must be finite and positive.
    :param record_run: bool, optional
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
        if (record_run):
            # Record the observation, regardless of acceptance
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

            if energy < best_energy:
                best_coords = coords.copy()
                best_energy = energy

        
    if (record_run):
        return best_coords, best_energy, n_iter, n_accepted, observations
    
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



def random_cluster(n_atoms: int, 
                   characteristic_distance: float,
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


if __name__ == "__main__":
    from src.main_code.local_optimizers.bfgs import LBFGS
    from pathlib import Path
    import pandas as pd

    output_dir = Path("data/records/LJ/1.0_1.0/methods/ml_assisted_bh")

    def local_optimizer(coords):
        return LBFGS(coords, pot.lennard_jones_derivative)

    initial_coords = random_cluster(
        n_atoms=50,
        characteristic_distance=2**(1/6),
        density=0.5,
        min_separation=0.8,
        seed=42
    )


    best_coords, best_energy, n_iter, n_accepted, observations = basin_hopping_opt(
        input_coords=initial_coords,
        potential_function=pot.lennard_jones_potential,
        local_optimizer_wrapper=local_optimizer,
        beta=1.0,
        max_iter=15000,
        delta=1,
        record_run=True,
        run_id=50
    )

    df = pd.DataFrame(observations)

    print("Best energy:", best_energy)
    print("Iterations:", n_iter)
    print("Accepted moves:", n_accepted)
    print("Observations:", len(observations))
    print("\nFirst observation:")
    print(df.head())

    df.to_csv(output_dir / f"ml_bh_50.csv", index=False)

    
   

