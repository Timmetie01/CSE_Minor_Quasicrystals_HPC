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
from src.main_code.global_optimizers.basin_hopping_library import basin_hopping as bh
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, average_precision_score, roc_auc_score, classification_report,confusion_matrix)
from src.main_code.global_optimizers.basin_hopping_library.basin_hopping import (calculate_features, create_observation,  benchmark_feature_cost)



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
    return atom_pair_energy(coords, atom_idx, coords[atom_idx], potential_function)


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
                       potential_function: Callable = pot.lennard_jones_potential,
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

# records fearures of single perturbation + additional information after relaxation, returning dictionary. Uses calculate_features function
def create_observation(
    coords : np.ndarray,
    pert_coords : np.ndarray, 
    new_coords : np.ndarray,
    best_energy_before : float,
    iteration : int, 
    run_id : int, 
    accepted : bool ,
    potential_function : Callable = pot.lennard_jones_potential,
    epsilon : float = 1e-5
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










