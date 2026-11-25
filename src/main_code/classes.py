import numpy as np
from src.main_code import potentials_calculations
from src.main_code.local_optimizers import fire, bfgs
from src.main_code.global_optimizers import basin_hopping, genetic_algorithm, parallel_tempering, simulated_annealing
from collections.abc import Callable

class Potential:
    def __init__(self, potential_func: Callable, potential_derivative: Callable, potential_str):
        self.potential_func = potential_func
        self.potential_derivative = potential_derivative
        self.potential_str = potential_str

def LennardJones(epsilon=1, sigma=1):
    potential_func = potentials_calculations.make_lennard_jones_potential(epsilon, sigma)
    potential_derivative = potentials_calculations.make_lennard_jones_derivative(epsilon, sigma)
    return Potential(potential_func, potential_derivative, "LJ")

def OscillatingPair(frequency=1, phase_shift=0):
    potential_func = potentials_calculations.make_oscillating_pair_potential(frequency, phase_shift)
    potential_derivative = potentials_calculations.make_oscillating_pair_derivative(frequency, phase_shift)
    return Potential(potential_func, potential_derivative, "OP")

def RawOscillatingPair(frequency=1, phase_shift=0):
    potential_func = potentials_calculations.make_oscillating_pair_potential(frequency, phase_shift)
    potential_derivative = potentials_calculations.make_oscillating_pair_derivative(frequency, phase_shift)
    return Potential(potential_func, potential_derivative, "raw_OP")


class AtomCluster:
    def __init__(self, coords: np.ndarray, Potential: Potential):
        '''
        Initialize the AtomCluster with coordinates.

        :param coords: array_like, shape (N, 3), 
            Coordinates of N particles in 3D space.
        :param Potential: class,
            Class of type Potential. Use functions LennardJones or OscillatingPair to initialize. So syntax is: Potential=LennardJones(1, 1) or similar
        '''
        self.coords = np.asarray(coords)
        self.atom_count = self.coords.shape[0]

        self.potential_func = Potential.potential_func
        self.potential_derivative = Potential.potential_derivative

    def potential(self, coords=None):
        '''
        Calculate the total potential for the cluster using the provided potential function.

        :param coords: np.ndarray (Nx3)
            coordinates to calculate the potential of. Defaults to None, which makes the function use self.coords
        :returns: float, 
            Total potential of the cluster.
        '''

        if coords == None: coords = self.coords
        return potentials_calculations.cluster_potential(coords, self.potential_func)

    def potential_gradient(self, coords=None):
        '''
        Calculate the total potential for the cluster using the provided potential function.
        
        :param coords: np.ndarray (Nx3)
            coordinates to calculate the gradient with. Defaults to None, which makes the function use self.coords
        :returns: np.ndarray (Nx3), 
            gradient of the potential for each atom
        '''

        if coords == None: coords = self.coords
        return potentials_calculations.cluster_potential_gradient(coords, self.potential_derivative)

    def FIRE(self, coords=None, Nmax=10000, Nmin=5, f_inc=1.1, f_dec=0.5, alpha0=0.1, f_alpha=0.99, err_tol=1e-5, dt=0.002, max_step=0.05):
        '''
        Minimize the potential energy of a particle cluster using the Fast Inertial
        Relaxation Engine (FIRE) algorithm with Velocity Verlet integration.
        https://journals-aps-org.tudelft.idm.oclc.org/prl/pdf/10.1103/PhysRevLett.97.170201 
    
        :param coords: array_like, shape (N, 3) or None
            Initial 3D coordinates of the N particles in the cluster. If None, uses self.coords and overwrites self.coords with optimized coordinates
        :param potential_derivative_func: callable
            Function that calculates the derivative of the pair potential with respect
            to inter-particle distance r.
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
            (must be < 1). Default is 0.5.
        :param alpha0: float, optional
            Initial and reset value for the velocity-mixing parameter alpha. Default is 0.1.
        :param f_alpha: float, optional
            Multiplicative decay factor applied to alpha when accelerating. Default is 0.99.
        :param err_tol: float, optional
            Convergence threshold for the maximum absolute force component on any particle.
            Default is 1e-5.
        :param dt: float, optional
            Initial integration time step (maximum time step is capped at 10 * dt).
            Default is 0.002.
        :param max_step: float, optional
            Maximum allowed spatial displacement per particle in a single iteration to
            prevent numerical instability from large initial forces. Default is 0.05.
        :returns: ndarray, shape (N, 3)
            Optimized 3D coordinates of the N particles at a local energy minimum. If coords was None, self.coords will be set to optimized coordinates
        '''
        change_own_coords = False
        if coords == None: 
            coords = self.coords
            change_own_coords = True

        opt_coords = fire.FIRE(coords, self.potential_derivative, Nmax=Nmax, Nmin=Nmin, f_inc=f_inc, f_dec=f_dec, alpha0=alpha0, f_alpha=f_alpha, err_tol=err_tol, dt=dt, max_step=max_step)

        if change_own_coords:
            self.coords = opt_coords
        return opt_coords

    def BFGS(self, coords=None, tol=1e-5, max_iter=1000, max_step=0.1):
        '''
        STILL NEEDS DOCUMENTATION
        
        '''
        change_own_coords = False
        if coords == None: 
            coords = self.coords
            change_own_coords = True

        opt_coords = bfgs.BFGS(coords, self.potential_derivative, self.potential_func, tol=tol, max_iter=max_iter, max_step=max_step)

        if change_own_coords:
            self.coords = opt_coords
        return opt_coords

    def LBFGS(self, coords=None, tol=1e-5, max_iter=1000, max_step=0.1):
        '''
        STILL NEEDS DOCUMENTATION AND VERIFICATION
        
        '''
        change_own_coords = False
        if coords == None: 
            coords = self.coords
            change_own_coords = True

        opt_coords = bfgs.LBFGS(coords, self.potential_derivative, self.potential_func, tol=tol, max_iter=max_iter, max_step=max_step)

        if change_own_coords:
            self.coords = opt_coords
        return opt_coords


    def basin_hopping(self, coords=None, local_optimizer=bfgs.LBFGS, beta=200, max_iter=1000):
        '''
        NOT IMPLEMENTED HERE YET

        raises: NotImplementedError
        '''

        change_own_coords = False
        if coords == None: 
            coords = self.coords
            change_own_coords = True

        def local_optimizer_wrapper():
            if local_optimizer == bfgs.BFGS or local_optimizer == bfgs.LBFGS:
                def opt(coords):
                    return local_optimizer(coords, self.potential_derivative, self.potential_func)
            else:
                def opt(coords):
                    return local_optimizer(coords, self.potential_derivative)
            return opt

        opt_coords = basin_hopping.basin_hopping(
            coords, self.potential_func, 
            local_optimizer_wrapper(), beta=beta, max_iter=max_iter)

        if change_own_coords:
            self.coords = opt_coords
        return opt_coords

    def genetic_algorithm(self, N_atoms, Pop_size, local_optimizer=bfgs.LBFGS, cutoff_perc=0.5, mutation_prob=0.25, max_frac=0.15):
        '''
        NOT IMPLEMENTED HERE YET

        raises: NotImplementedError

        I have to import from testing file:
        '''

        change_own_coords = False
        if coords == None: 
            coords = self.coords
            change_own_coords = True

        def local_optimizer_wrapper():
            if local_optimizer == bfgs.BFGS or local_optimizer == bfgs.LBFGS:
                def opt(coords):
                    return local_optimizer(coords, self.potential_derivative, self.potential_func)
            else:
                def opt(coords):
                    return local_optimizer(coords, self.potential_derivative)
            return opt

        rng = np.random.default_rng()
        
        selection = genetic_algorithm.make_operators(
            genetic_algorithm.truncation_selection,
            potential_func=self.potential_func,
            cutoff_perc=cutoff_perc,
            rng=rng,
        )
        crossover = genetic_algorithm.make_operators(
            genetic_algorithm.cut_splice,
            max_attempts=1000,
            rng=rng,
        )
        mutation = genetic_algorithm.make_operators(
            genetic_algorithm.random_displacement,
            rng=rng,
            prob=mutation_prob,
            max_frac=max_frac,
        )
        local_optimizer = genetic_algorithm.make_population_local_optimizer(
            local_optimizer,
            potential_derivative_func=self.potential_derivative,
        )

        initial_population = genetic_algorithm.rcube_initial_population(Pop_size, N_atoms,  N_atoms ** (1/3))

        
        opt_coords = genetic_algorithm.GA(initial_population, genetic_algorithm.truncation_selection, crossover, mutation, local_optimizer_wrapper())

        if change_own_coords:
            self.coords = opt_coords
        return opt_coords

    def parallel_tempering(self, N_atoms, M=10, temps="Auto", deltas="Auto", R_c=5, swap_every=10, N_sweeps=2000):
        '''
        NOT IMPLEMENTED HERE YET

        raises: NotImplementedError
        Hardcoded potential...
        '''
        def cluster_potential_wrapper():
            def pot(coords):
                return potentials_calculations.cluster_potential(coords, self.potential_func)
            return pot

        energies, coords = parallel_tempering.make_start(N_atoms, M, R_c, cluster_potential_wrapper())

        if temps == "Auto":
            temps = np.geomspace(0.05, 0.40, M)
        if deltas == "Auto":
            deltas = parallel_tempering.make_deltas(temps)

        change_own_coords = False
        if coords == None: 
            coords = self.coords
            change_own_coords = True

        opt_coords = parallel_tempering.parallel_tempering_run(
            coords, energies, 
            M=M, temps=temps, deltas=deltas, R_c=R_c, swap_every=swap_every, N_sweeps=N_sweeps, potential_func=self.potential_func
        )

        if change_own_coords:
            self.coords = opt_coords
        return opt_coords

    def simulated_annealing(self, N_atoms, local_optimizer=bfgs.LBFGS, set_own_coords=False):
        '''
        NOT IMPLEMENTED HERE YET

        Done?
        
        '''

        def local_optimizer_wrapper():
            if local_optimizer == bfgs.BFGS or local_optimizer == bfgs.LBFGS:
                def opt(coords, pot_derivative):
                    return local_optimizer(coords, pot_derivative, self.potential_func)
            else:
                def opt(coords, pot_derivative):
                    return local_optimizer(coords, pot_derivative)
            return opt

        opt_coords = simulated_annealing.simulated_annealing(
            len(N_atoms), self.potential_func, self.potential_derivative,
            local_optimizer_wrapper(), None
            )

        if set_own_coords:
            self.coords = opt_coords
        return opt_coords
