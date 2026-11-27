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

from src.main_code.global_optimizers.basin_hopping_library.feature_extraction import (calculate_features, create_observation,  benchmark_feature_cost)

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


if __name__ == "__main__":
    # # to do: 


    # # 5. After creating dataset, train first rf model. Look into parameter tuning
    # # 6. Try XGBoost and parameter tuning.
    # # 7. Look in to SHAP
    def local_optimizer_wrapper(coords):
        return LBFGS(coords, print_convergence = False)
    
    #rf_model_30, X_test, Y_test, X_train, Y_train = train_rf_model(30, 10, 1500)
    
    # #benchmark_ml_bh(30, rf_model_30, k = 35, n_runs = 100)
    # explainer = shap.TreeExplainer(rf_model_30)

    # shap_test = explainer(X_test)[:, :, 1]


    # # Define output directory
    # output_dir = Path(
    #     "data/records/LJ/1.0_1.0/methods/ml_assisted_bh/feature_importance"
    # )

    # output_dir.mkdir(parents=True, exist_ok=True)

    # # Generate beeswarm plot
    # shap.plots.beeswarm(
    #     shap_test,
    #     max_display=20,
    #     plot_size=(14, 9),
    #     show=False
    # )

    # plt.subplots_adjust(left=0.35)
    # plt.title("LJ30 - SHAP Feature Importance")

    # plt.show()
    # plt.close()
    
