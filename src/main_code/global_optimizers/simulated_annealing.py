import time

import numpy as np

import src.main_code.potentials_calculations as pt
from src.main_code.local_optimizers.fire import FIRE


def atom_energy(x, i, v):
    """
    Energy of atom i with all the other atoms.
    :param x: array_like, shape (N, 3)
        3D coordinates of the N particles in the cluster.
    :param i: int
        Index of the atom whose energy is calculated.
    :param v: callable
        Pair potential V(r) between two atoms.
    :returns: float
        Sum of V(r_ij) over all atoms j != i.
    """
    others = np.delete(x, i, axis=0)
    r = np.linalg.norm(others - x[i], axis=1)
    return np.sum(v(r))


def random_in_sphere(n, radius, rng):
    """
    Place n points uniformly inside a sphere (rejection sampling).
    :param n: int
        Number of points.
    :param radius: float
        Radius of the sphere, centred at the origin.
    :param rng: numpy.random.Generator
        Random number generator.
    :returns: ndarray, shape (n, 3)
        Coordinates of the points.
    """
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:       # outside: draw again
            points.append(p)
    return np.array(points)


def simulated_annealing(n_atoms, v, dv, local_optimizer=FIRE,
                        local_optimizer_kwargs=None,
                        T_start=1.0, T_end=0.01,
                        steps_per_atom=20_000, step_size=0.2, radius=None,
                        target_acceptance=0.45, adapt_every=1000, seed=0):
    """
    Find a low-energy structure of a cluster with simulated annealing,
    followed by a FIRE minimisation.
    :param n_atoms: int
        Number of atoms in the cluster.
    :param v: callable
        Pair potential V(r).
    :param dv: callable
        Derivative dV/dr, used by FIRE.
    :param T_start: float, optional
        Starting temperature (reduced units). Should be above the cluster's
        melting temperature, so the random start is forgotten.
    :param T_end: float, optional
        Final temperature, low enough that the structure is frozen.
    :param steps_per_atom: int, optional
        Monte Carlo moves per atom; total moves = steps_per_atom * n_atoms.
    :param step_size: float, optional
        Initial maximum displacement per move. Adapted during the run.
    :param radius: float or None, optional
        Radius of the container sphere. If None, it grows with n_atoms.
    :param target_acceptance: float, optional
        Fraction of accepted moves that the step-size adaptation aims for.
    :param adapt_every: int, optional
        Number of moves between step-size adjustments.
    :param seed: int, optional
        Seed for the random number generator.
    :returns: tuple (ndarray of shape (n_atoms, 3), float)
        Final coordinates and their energy.
    """
    rng = np.random.default_rng(seed)

    # The container grows with the cluster size
    if radius is None:
        radius = 1.0 + 0.7 * n_atoms ** (1 / 3)

    # Random starting positions inside the container
    x = random_in_sphere(n_atoms, radius, rng)

    # Geometric cooling: T goes from T_start to T_end over n_steps
    n_steps = steps_per_atom * n_atoms
    alpha = (T_end / T_start) ** (1 / n_steps)

    accepted = 0
    for step in range(n_steps):
        T = T_start * alpha ** step

        # Pick a random atom and move it a little
        i = rng.integers(n_atoms)
        old_energy = atom_energy(x, i, v)
        old_pos = x[i].copy()
        x[i] += rng.uniform(-step_size, step_size, 3)

        if np.linalg.norm(x[i]) > radius:
            # Outside the container: reject, so the cluster can't fall apart
            x[i] = old_pos
        else:
            # Metropolis rule: always accept downhill,
            # accept uphill with probability exp(-dE/T)
            dE = atom_energy(x, i, v) - old_energy
            if dE > 0 and rng.random() > np.exp(-dE / T):
                x[i] = old_pos
            else:
                accepted += 1

        # Adapt the step size so about target_acceptance of moves are accepted
        if (step + 1) % adapt_every == 0:
            rate = accepted / adapt_every
            step_size *= 1.1 if rate > target_acceptance else 0.9
            step_size = min(max(step_size, 0.01), 1.0)
            accepted = 0

    # Slide to the bottom of the valley with the chosen local optimizer
    if local_optimizer_kwargs is None:
        local_optimizer_kwargs = {}
    x = local_optimizer(x, dv, **local_optimizer_kwargs)
    return x, pt.cluster_potential(x, v)


def best_of_runs(n_atoms, v, dv, n_runs=5, first_seed=0, **kwargs):
    """
    Run simulated annealing n_runs times with different seeds and keep the
    lowest energy. Extra keyword arguments go to simulated_annealing.
    :returns: tuple (ndarray, float, list of float)
        Best coordinates, best energy, and the energies of all runs.
    """
    best_x, best_E, energies = None, np.inf, []
    for seed in range(first_seed, first_seed + n_runs):
        x, E = simulated_annealing(n_atoms, v, dv, seed=seed, **kwargs)
        energies.append(E)
        if E < best_E:
            best_x, best_E = x, E
    return best_x, best_E, energies


if __name__ == "__main__":
    v = pt.make_lennard_jones_potential()
    dv = pt.make_lennard_jones_derivative()

    # Known Lennard-Jones global minima (Cambridge Cluster Database)
    known = {13: -44.326801, 25: -102.372663}

    for n_atoms, E_known in known.items():
        t0 = time.perf_counter()
        x, E_best, energies = best_of_runs(n_atoms, v, dv, n_runs=10)
        successes = sum(abs(E - E_known) < 1e-4 for E in energies)
        print(f"N = {n_atoms}: best E = {E_best:.6f} (known {E_known}), "
              f"{successes}/{len(energies)} runs found it, "
              f"{time.perf_counter() - t0:.0f} s")
