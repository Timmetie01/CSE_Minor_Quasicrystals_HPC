import numpy as np
from src.main_code import potentials_calculations
from src.main_code.local_optimizers import fire, bfgs
from src.main_code.global_optimizers import basin_hopping, genetic_algorithm, parallel_tempering, simulated_annealing
from collections.abc import Callable
from data.data_handling import highscores
from typing import Optional, Union

class Potential:
    def __init__(self, potential_func: Callable, potential_derivative: Callable, potential_str, potential_parameters):
        self.potential_func = potential_func
        self.potential_derivative = potential_derivative
        self.potential_str = potential_str
        self.potential_parameters = potential_parameters

    def potential(self, coords):
        return potentials_calculations.cluster_potential(coords, self.potential_func)

    def gradient(self, coords):
        return potentials_calculations.cluster_potential_gradient(coords, self.potential_derivative)

def LennardJones(epsilon=1, sigma=1):
    potential_func = potentials_calculations.make_lennard_jones_potential(epsilon, sigma)
    potential_derivative = potentials_calculations.make_lennard_jones_derivative(epsilon, sigma)
    return Potential(potential_func, potential_derivative, "LJ", (epsilon, sigma))

def OscillatingPair(frequency, phase_shift):
    potential_func = potentials_calculations.make_tapered_oscillating_pair_potential(frequency, phase_shift)
    potential_derivative = potentials_calculations.make_tapered_oscillating_pair_derivative(frequency, phase_shift)
    return Potential(potential_func, potential_derivative, "OP", (frequency, phase_shift))

def RawOscillatingPair(frequency, phase_shift):
    potential_func = potentials_calculations.make_oscillating_pair_potential(frequency, phase_shift)
    potential_derivative = potentials_calculations.make_oscillating_pair_derivative(frequency, phase_shift)
    return Potential(potential_func, potential_derivative, "raw_OP", (frequency, phase_shift))


class Optimizer:
    """
    Unified interface for local and global geometry optimization of atomic clusters.

    All parameters from the underlying local optimizers (FIRE, BFGS, L-BFGS) and
    global optimizers (Basin Hopping, Genetic Algorithm, Parallel Tempering,
    Simulated Annealing) are directly exposed with consistent signatures.
    """

    def __init__(self, Potential: Potential):
        """
        Initialize the Optimizer with a pair potential.

        :param Potential: Potential
            Instance of type Potential. Use helper functions such as ``LennardJones(1, 1)``
            or ``OscillatingPair(1, 0)`` to initialize.
        """
        self.potential_func = Potential.potential_func
        self.potential_derivative = Potential.potential_derivative
        self.potential_str = Potential.potential_str
        self.potential_parameters = Potential.potential_parameters

    # =========================================================================
    # Local Optimizers

    def FIRE(
        self,
        coords: np.ndarray,
        keep_steps: bool = False,
        Nmax: int = 10_000,
        Nmin: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        alpha0: float = 0.1,
        f_alpha: float = 0.99,
        err_tol: float = 1e-5,
        dt: float = 0.002,
        max_step: float = 0.05,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Minimize the potential energy of a particle cluster using the Fast Inertial
        Relaxation Engine (FIRE) algorithm with Velocity Verlet integration.
        https://journals-aps-org.tudelft.idm.oclc.org/prl/pdf/10.1103/PhysRevLett.97.170201

        :param coords: array_like, shape (N, 3)
            Initial 3D coordinates of the N particles in the cluster.
        :param keep_steps: bool, optional
            If True, records and returns the full trajectory of coordinates across all
            iterations alongside the final coordinates. Default is False.
        :param Nmax: int, optional
            Maximum number of optimization iterations. Default is 10_000.
        :param Nmin: int, optional
            Minimum number of consecutive steps with positive power (P > 0) required
            before increasing the time step and decreasing alpha. Default is 5.
        :param f_inc: float, optional
            Multiplicative factor used to increase the time step when P > 0 for at least
            Nmin steps (must be > 1). Default is 1.1.
        :param f_dec: float, optional
            Multiplicative factor used to decrease the time step when P <= 0
            (must be in (0, 1)). Default is 0.5.
        :param alpha0: float, optional
            Initial and reset value for the velocity-mixing parameter alpha (must be in (0, 1]).
            Default is 0.1.
        :param f_alpha: float, optional
            Multiplicative decay factor applied to alpha when accelerating (must be in (0, 1]).
            Default is 0.99.
        :param err_tol: float, optional
            Convergence threshold for the maximum absolute force component on any particle.
            Default is 1e-5.
        :param dt: float, optional
            Initial integration time step (maximum time step is capped at ``10 * dt``).
            Default is 0.002.
        :param max_step: float, optional
            Maximum allowed spatial displacement per particle in a single iteration to
            prevent numerical instability from large initial forces. Default is 0.05.
        :returns: ndarray of shape (N, 3), or tuple of (ndarray, ndarray)
            Optimized 3D coordinates of the N particles at a local energy minimum.
            If ``keep_steps=True``, returns ``(opt_coords, step_arr)``.
        """
        return fire.FIRE(
            coords,
            self.potential_derivative,
            keep_steps=keep_steps,
            Nmax=Nmax,
            Nmin=Nmin,
            f_inc=f_inc,
            f_dec=f_dec,
            alpha0=alpha0,
            f_alpha=f_alpha,
            err_tol=err_tol,
            dt=dt,
            max_step=max_step,
        )

    def BFGS(
        self,
        coords: np.ndarray,
        tol: float = 1e-5,
        max_iter: int = 1000,
        max_step: float = 0.1,
        keep_steps: bool = False,
        print_convergence: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Minimize the potential energy of a particle cluster using the Broyden-Fletcher-
        Goldfarb-Shanno (BFGS) quasi-Newton algorithm with an Armijo backtracking
        and quadratic interpolation line search.
        https://doi.org/10.1007/978-0-387-40065-5

        :param coords: array_like, shape (N, 3)
            Initial 3D coordinates of the N particles in the cluster.
        :param tol: float, optional
            Convergence threshold for the maximum absolute gradient component on any
            particle. Default is 1e-5.
        :param max_iter: int, optional
            Maximum number of BFGS optimization iterations. Default is 1000.
        :param max_step: float, optional
            Maximum allowed spatial displacement per particle in a single iteration to
            prevent numerical instability from large initial forces. Default is 0.1.
        :param keep_steps: bool, optional
            If True, records and returns the trajectory of coordinates across all
            iterations alongside the final coordinates. Default is False.
        :param print_convergence: bool, optional
            If True, prints convergence status or emits a warning if ``max_iter`` is
            reached without converging. Default is True.
        :returns: ndarray of shape (N, 3), or tuple of (ndarray, ndarray)
            Optimized 3D coordinates of the N particles at a local energy minimum.
            If ``keep_steps=True``, returns ``(opt_coords, step_list)`` of shape ``(n_iter + 1, N, 3)``.
        """
        return bfgs.BFGS(
            coords,
            self.potential_derivative,
            self.potential_func,
            tol=tol,
            max_iter=max_iter,
            max_step=max_step,
            keep_steps=keep_steps,
            print_convergence=print_convergence,
        )

    def LBFGS(
        self,
        coords: np.ndarray,
        tol: float = 1e-5,
        max_iter: int = 1000,
        max_step: float = 0.1,
        m: int = 10,
        keep_steps: bool = False,
        print_convergence: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Minimize the potential energy of an atomic cluster using the Limited-Memory
        Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm with two-loop recursion,
        quadratic-interpolated Armijo backtracking, and per-atom step capping.
        https://doi.org/10.1007/978-0-387-40065-5

        :param coords: array_like, shape (N, 3)
            Initial 3D coordinates of the N particles in the cluster.
        :param tol: float, optional
            Convergence threshold for the maximum absolute gradient component on any
            particle. Default is 1e-5.
        :param max_iter: int, optional
            Maximum number of L-BFGS optimization iterations. Default is 1000.
        :param max_step: float, optional
            Maximum allowed spatial displacement per particle in a single iteration to
            prevent numerical instability from large initial forces. Default is 0.1.
        :param m: int, optional
            Number of past curvature pairs ``(s, y)`` retained in history. Default is 10.
        :param keep_steps: bool, optional
            If True, records and returns the trajectory of coordinates across all
            iterations alongside the final coordinates. Default is False.
        :param print_convergence: bool, optional
            If True, prints convergence status or emits a warning if ``max_iter`` is
            reached without converging. Default is True.
        :returns: ndarray of shape (N, 3), or tuple of (ndarray, ndarray)
            Optimized 3D coordinates of the N particles at a local energy minimum.
            If ``keep_steps=True``, returns ``(opt_coords, step_list)`` of shape ``(n_iter + 1, N, 3)``.
        """
        return bfgs.LBFGS(
            coords,
            self.potential_derivative,
            self.potential_func,
            tol=tol,
            max_iter=max_iter,
            max_step=max_step,
            m=m,
            keep_steps=keep_steps,
            print_convergence=print_convergence,
        )

    # =========================================================================
    # Internal Functions

    def _resolve_local_func(self, local_optimizer: str | Callable) -> Callable:
        """Resolve string names ('FIRE', 'BFGS', 'LBFGS') to module functions."""
        if isinstance(local_optimizer, str):
            mapping = {
                "fire": fire.FIRE,
                "bfgs": bfgs.BFGS,
                "lbfgs": bfgs.LBFGS,
            }
            key = local_optimizer.lower()
            if key not in mapping:
                raise ValueError(
                    f"Unknown local_optimizer '{local_optimizer}'. Expected one of {list(mapping)} or a callable."
                )
            return mapping[key]
        return local_optimizer

    def _build_single_cluster_local_opt(
        self,
        local_optimizer: str | Callable,
        local_optimizer_kwargs: Optional[dict] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Normalize any local optimizer into a uniform ``opt(coords) -> opt_coords`` callable,
        forwarding ``local_optimizer_kwargs`` and suppressing repetitive convergence prints
        inside global loops unless explicitly enabled.
        """
        func = self._resolve_local_func(local_optimizer)
        kwargs = dict(local_optimizer_kwargs or {})

        if func in (bfgs.BFGS, bfgs.LBFGS):
            kwargs.setdefault("print_convergence", False)
            return lambda coords: func(
                coords, self.potential_derivative, self.potential_func, **kwargs
            )
        return lambda coords: func(coords, self.potential_derivative, **kwargs)

    def _log_highscore(self, opt_coords: np.ndarray, method_name: str, log_info: Optional[str]) -> None:
        """Record global optimization results to highscores unless ``log_info`` is None."""
        if log_info is not None:
            highscores(
                opt_coords,
                method_name,
                self.potential_str,
                self.potential_parameters,
                log_info,
            )

    # =========================================================================
    # Global Optimizers

    def basin_hopping(
        self,
        N_atoms: int,
        local_optimizer: str | Callable = bfgs.LBFGS,
        local_optimizer_kwargs: Optional[dict] = None,
        beta: float = 1.0,
        max_iter: int = 10_000,
        perturbation_type: str = "random",
        delta: float = 1.5,
        init_coords: Optional[np.ndarray] = None,
        log_info: Optional[str] = "",
    ) -> np.ndarray:
        """
        Perform global geometry optimization of a particle cluster using the
        Basin-Hopping algorithm with Metropolis acceptance.

        :param N_atoms: int
            Number of particles in the cluster (ignored if ``init_coords`` is provided).
        :param local_optimizer: callable or {"FIRE", "BFGS", "LBFGS"}, optional
            Local minimizer applied after each perturbation. Default is ``bfgs.LBFGS``.
        :param local_optimizer_kwargs: dict, optional
            Extra keyword arguments forwarded to ``local_optimizer`` (e.g., ``{"tol": 1e-6}``).
        :param beta: float, optional
            Inverse temperature parameter ``1 / (k_B * T)`` used in the Metropolis criterion.
            Must be finite and non-negative. Default is 1.0.
        :param max_iter: int, optional
            Maximum number of basin-hopping iterations. Must be a positive integer.
            Default is 10_000.
        :param perturbation_type: str, optional
            Method used to perturb the cluster. Default is ``"random"``.
        :param delta: float, optional
            Maximum displacement in each Cartesian direction for random perturbations.
            Must be finite and positive. Default is 1.5.
        :param init_coords: array_like, shape (N_atoms, 3), optional
            Custom initial coordinates. If None, spawns random coordinates in a cube of
            side length ``2 * N_atoms ** (1/3)``.
        :param log_info: str or None, optional
            Metadata string logged to ``highscores``. Set to None to disable logging.
            Default is ``""``.
        :returns: ndarray, shape (N_atoms, 3)
            Coordinates of the lowest-energy configuration found.
        """
        if init_coords is None:
            coords = np.random.random((N_atoms, 3)) * (2 * N_atoms ** (1 / 3))
        else:
            coords = np.asarray(init_coords, dtype=float)

        opt_wrapper = self._build_single_cluster_local_opt(
            local_optimizer, local_optimizer_kwargs
        )

        opt_coords, _, _, _ = basin_hopping.basin_hopping(
            coords,
            self.potential_func,
            opt_wrapper,
            beta=beta,
            max_iter=max_iter,
            perturbation_type=perturbation_type,
            delta=delta,
        )

        self._log_highscore(opt_coords, "basin_hopping", log_info)
        return opt_coords

    def genetic_algorithm(
        self,
        N_atoms: int,
        Pop_size: int = 50,
        local_optimizer: str | Callable = bfgs.LBFGS,
        local_optimizer_kwargs: Optional[dict] = None,
        number_of_iterations: int = 100,
        elitism: int = 0,
        cutoff_perc: float = 0.5,
        max_attempts: int = 1000,
        mutation_prob: float = 0.25,
        max_frac: float = 0.15,
        cube_length: Optional[float] = None,
        seed: Optional[int] = None,
        log_info: Optional[str] = "",
    ) -> np.ndarray:
        """
        Perform global geometry optimization of a particle cluster using a Genetic Algorithm
        with truncation selection, cut-and-splice crossover, random displacement mutation,
        and local relaxation.

        :param N_atoms: int
            Number of particles per cluster.
        :param Pop_size: int, optional
            Number of clusters in the population (must be >= 2). Default is 50.
        :param local_optimizer: callable or {"FIRE", "BFGS", "LBFGS"}, optional
            Local minimizer used to relax initial and offspring clusters.
            Default is ``bfgs.LBFGS``.
        :param local_optimizer_kwargs: dict, optional
            Extra keyword arguments forwarded to ``local_optimizer``.
        :param number_of_iterations: int, optional
            Number of generations to evolve. Default is 100.
        :param elitism: int, optional
            Number of lowest-energy clusters copied directly into the next generation
            (must be ``0 <= elitism < Pop_size``). Default is 0.
        :param cutoff_perc: float, optional
            Fraction of highest-energy individuals removed during truncation selection,
            in ``[0, 1)``. Default is 0.5.
        :param max_attempts: int, optional
            Maximum random cutting planes attempted during cut-and-splice crossover.
            Default is 1000.
        :param mutation_prob: float, optional
            Probability in ``[0, 1]`` that an offspring undergoes mutation. Default is 0.25.
        :param max_frac: float, optional
            Maximum atomic displacement during mutation as a fraction of the cluster's
            average pairwise distance. Default is 0.15.
        :param cube_length: float, optional
            Side length of the bounding cube for initial population generation.
            If None, defaults to ``N_atoms ** (1/3)``.
        :param seed: int, optional
            Random seed for reproducible selection, crossover, and mutation. Default is None.
        :param log_info: str or None, optional
            Metadata string logged to ``highscores``. Set to None to disable logging.
            Default is ``""``.
        :returns: ndarray, shape (N_atoms, 3)
            Coordinates of the lowest-energy cluster in the final population.
        """
        rng = np.random.default_rng(seed)

        selection = genetic_algorithm.make_operators(
            genetic_algorithm.truncation_selection,
            potential_func=self.potential_func,
            cutoff_perc=cutoff_perc,
            rng=rng,
        )
        crossover = genetic_algorithm.make_operators(
            genetic_algorithm.cut_splice,
            max_attempts=max_attempts,
            rng=rng,
        )
        mutation = genetic_algorithm.make_operators(
            genetic_algorithm.random_displacement,
            rng=rng,
            prob=mutation_prob,
            max_frac=max_frac,
        )

        single_opt = self._build_single_cluster_local_opt(
            local_optimizer, local_optimizer_kwargs
        )
        pop_local_optimizer = lambda clusters: [single_opt(c) for c in clusters]

        if cube_length is None:
            cube_length = N_atoms ** (1 / 3)

        initial_population = genetic_algorithm.rcube_initial_population(
            Pop_size, N_atoms, cube_length
        )

        coord_arr = genetic_algorithm.GA(
            initial_population,
            selection,
            crossover,
            mutation,
            pop_local_optimizer,
            number_of_iterations=number_of_iterations,
            elitism=elitism,
        )

        opt_coords = min(
            coord_arr,
            key=lambda c: potentials_calculations.cluster_potential(c, self.potential_func),
        )

        self._log_highscore(opt_coords, "genetic_algorithm", log_info)
        return opt_coords

    def parallel_tempering(
        self,
        N_atoms: int,
        M: int = 10,
        temps: Union[str, np.ndarray] = "Auto",
        T_min: float = 0.05,
        T_max: float = 0.40,
        deltas: Union[str, np.ndarray] = "Auto",
        d_ref: float = 0.1,
        T_ref: float = 0.05,
        R_c: float = 5.0,
        swap_every: int = 10,
        N_sweeps: int = 2000,
        quench: bool = False,
        local_optimizer: str | Callable = bfgs.LBFGS,
        local_optimizer_kwargs: Optional[dict] = None,
        log_info: Optional[str] = "",
    ) -> np.ndarray:
        """
        Perform global geometry optimization of a particle cluster using adaptive
        Parallel Tempering (Replica Exchange) Monte Carlo.

        :param N_atoms: int
            Number of particles in the cluster.
        :param M: int, optional
            Number of temperature replicas. Default is 10.
        :param temps: "Auto" or array_like of shape (M,), optional
            Strictly increasing temperature ladder. If ``"Auto"``, generates a geometric
            ladder ``np.geomspace(T_min, T_max, M)``. Default is ``"Auto"``.
        :param T_min: float, optional
            Lowest anchor temperature used when ``temps="Auto"``. Default is 0.05.
        :param T_max: float, optional
            Highest initial temperature used when ``temps="Auto"``. Default is 0.40.
        :param deltas: "Auto" or array_like of shape (M,), optional
            Maximum single-atom Monte Carlo displacement per temperature slot. If ``"Auto"``,
            computes ``make_deltas(temps, d_ref=d_ref, T_ref=T_ref)``. Default is ``"Auto"``.
        :param d_ref: float, optional
            Reference step size at ``T_ref`` when ``deltas="Auto"``. Default is 0.1.
        :param T_ref: float, optional
            Reference temperature for ``d_ref`` when ``deltas="Auto"``. Default is 0.05.
        :param R_c: float, optional
            Maximum confining spherical radius of the cluster. Default is 5.0.
        :param swap_every: int, optional
            Frequency (in sweeps) at which adjacent temperature swaps and ladder updates
            are attempted. Default is 10.
        :param N_sweeps: int, optional
            Total number of Monte Carlo sweeps per replica (each sweep performs ``N_atoms``
            trial moves). Default is 2000.
        :param quench: bool, optional
            If True, locally relaxes the lowest-energy configuration to T=0 at the end of
            the run using ``local_optimizer``. Default is False.
        :param local_optimizer: callable or {"FIRE", "BFGS", "LBFGS"}, optional
            Local minimizer used when ``quench=True``. Default is ``bfgs.LBFGS``.
        :param local_optimizer_kwargs: dict, optional
            Extra keyword arguments forwarded to ``local_optimizer`` when ``quench=True``.
        :param log_info: str or None, optional
            Metadata string logged to ``highscores``. Set to None to disable logging.
            Default is ``""``.
        :returns: ndarray, shape (N_atoms, 3)
            Coordinates of the lowest-energy configuration found.
        """
        cluster_pot_fn = lambda coords: potentials_calculations.cluster_potential(
            coords, self.potential_func
        )

        energies, coords = parallel_tempering.make_start(
            N_atoms, M, R_c, cluster_pot_fn
        )

        if isinstance(temps, str) and temps == "Auto":
            temps = np.geomspace(T_min, T_max, M)
        if isinstance(deltas, str) and deltas == "Auto":
            deltas = parallel_tempering.make_deltas(temps, d_ref=d_ref, T_ref=T_ref)

        _, opt_coords = parallel_tempering.parallel_tempering_run(
            coords,
            energies,
            M=M,
            temps=temps,
            deltas=deltas,
            R_c=R_c,
            swap_every=swap_every,
            N_sweeps=N_sweeps,
            potential_func=self.potential_func,
        )

        if quench:
            opt_wrapper = self._build_single_cluster_local_opt(
                local_optimizer, local_optimizer_kwargs
            )
            opt_coords = opt_wrapper(opt_coords)

        self._log_highscore(opt_coords, "parallel_tempering", log_info)
        return opt_coords

    def simulated_annealing(
        self,
        N_atoms: int,
        local_optimizer: str | Callable = bfgs.LBFGS,
        local_optimizer_kwargs: Optional[dict] = None,
        T_start: float = 1.0,
        T_end: float = 0.01,
        steps_per_atom: int = 20_000,
        step_size: float = 0.2,
        radius: Optional[float] = None,
        target_acceptance: float = 0.45,
        adapt_every: int = 1000,
        seed: int = 0,
        log_info: Optional[str] = "",
    ) -> np.ndarray:
        """
        Perform global geometry optimization of a particle cluster using Simulated
        Annealing with adaptive step sizes, followed by local minimization.

        :param N_atoms: int
            Number of particles in the cluster.
        :param local_optimizer: callable or {"FIRE", "BFGS", "LBFGS"}, optional
            Local minimizer used to relax the final annealed structure.
            Default is ``bfgs.LBFGS``.
        :param local_optimizer_kwargs: dict, optional
            Extra keyword arguments forwarded to ``local_optimizer``.
        :param T_start: float, optional
            Initial temperature in reduced units. Default is 1.0.
        :param T_end: float, optional
            Final temperature in reduced units. Default is 0.01.
        :param steps_per_atom: int, optional
            Monte Carlo moves per atom (total moves = ``steps_per_atom * N_atoms``).
            Default is 20_000.
        :param step_size: float, optional
            Initial maximum Cartesian displacement per move (adapted during the run).
            Default is 0.2.
        :param radius: float or None, optional
            Radius of the confining sphere. If None, defaults to ``1.0 + 0.7 * N_atoms ** (1/3)``.
        :param target_acceptance: float, optional
            Target Metropolis acceptance fraction for step-size adaptation. Default is 0.45.
        :param adapt_every: int, optional
            Number of Monte Carlo moves between step-size adjustments. Default is 1000.
        :param seed: int, optional
            Seed for the random number generator. Default is 0.
        :param log_info: str or None, optional
            Metadata string logged to ``highscores``. Set to None to disable logging.
            Default is ``""``.
        :returns: ndarray, shape (N_atoms, 3)
            Locally minimized coordinates of the final annealed cluster.
        """
        single_opt = self._build_single_cluster_local_opt(
            local_optimizer, local_optimizer_kwargs
        )
        # simulated_annealing.simulated_annealing calls local_optimizer(x, dv, **local_optimizer_kwargs)
        sa_local_opt = lambda coords, _dv, **kw: single_opt(coords)

        opt_coords, _ = simulated_annealing.simulated_annealing(
            N_atoms,
            self.potential_func,
            self.potential_derivative,
            local_optimizer=sa_local_opt,
            local_optimizer_kwargs=None,
            T_start=T_start,
            T_end=T_end,
            steps_per_atom=steps_per_atom,
            step_size=step_size,
            radius=radius,
            target_acceptance=target_acceptance,
            adapt_every=adapt_every,
            seed=seed,
        )

        self._log_highscore(opt_coords, "simulated_annealing", log_info)
        return opt_coords

if __name__ == "__main__":
    from data.data_handling import read_LJwales_data
    import pytest
    N_atoms = 10

    Pot = LennardJones()
    optcoords = read_LJwales_data(N_atoms)
    globopt = Optimizer(Pot)


    print(Pot.potential(globopt.basin_hopping(N_atoms)) == pytest.approx(Pot.potential(optcoords)))
    print(Pot.potential(globopt.genetic_algorithm(N_atoms, 20)) == pytest.approx(Pot.potential(optcoords)))
    print(Pot.potential(globopt.parallel_tempering(N_atoms)) == pytest.approx(Pot.potential(optcoords)))
    print(Pot.potential(globopt.simulated_annealing(N_atoms)) == pytest.approx(Pot.potential(optcoords)))
