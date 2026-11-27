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
