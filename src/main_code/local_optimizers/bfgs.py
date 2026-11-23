from src.main_code.potentials_calculations import cluster_potential_gradient, cluster_potential, lennard_jones_derivative, lennard_jones_potential
import numpy as np
import warnings


def BFGS(
        x, 
        potential_derivative_func: callable = lennard_jones_derivative, 
        potential_func: callable = lennard_jones_potential, 
        tol: float = 1e-5, 
        max_iter: int = 1000, 
        max_step:float = 0.1, 
        keep_steps: bool = False,
        print_convergence = True
        ):
    
    '''
    Minimize the potential energy of a particle cluster using the Broyden-Fletcher-
    Goldfarb-Shanno (BFGS) quasi-Newton algorithm with an Armijo backtracking
    and quadratic interpolation line search.
    https://doi.org/10.1007/978-0-387-40065-5

    :param x: array_like, shape (N, 3)
        Initial 3D coordinates of the N particles in the cluster.
    :param potential_derivative_func: callable
        Function that calculates the derivative of the pair potential with respect
        to inter-particle distance r. Default is lennard_jones_derivative.
    :param potential_func: callable
        Function that calculates the pair potential energy as a function of
        inter-particle distance r. Default is lennard_jones_potential.
    :param tol: float
        Convergence threshold for the maximum absolute gradient component on any
        particle. Default is 1e-5.
    :param max_iter: int
        Maximum number of BFGS optimization iterations. Default is 1000.
    :param max_step: float
        Maximum allowed spatial displacement per particle in a single iteration to
        prevent numerical instability from large initial forces. Default is 0.1.
    :param keep_steps: bool
        If True, records and returns the trajectory of coordinates across all
        iterations alongside the final coordinates. Default is False.
    :returns: ndarray, shape (N, 3) or tuple of (ndarray, ndarray)
        Returns the optimized 3D coordinates of the N particles at a local energy 
        minimum, shape (N, 3).
        If keep_steps is True, also returns step_list, which has shape 
        (n_iter + 1, N, 3) and contains the coordinates at each intermediate iteration.
    '''

    validate_bfgs_data(x, potential_derivative_func, potential_func, max_iter, tol, max_step)

    # 1: Bookkeeping
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    x = np.array(x, dtype=np.float64).ravel()
    
    if keep_steps:
        step_list = [x.reshape(shape).copy()]

    def pot(x_flat): return cluster_potential(x_flat.reshape(shape), potential_func)
    def grad(x_flat): return cluster_potential_gradient(x_flat.reshape(shape), potential_derivative_func).ravel()

    
    # 2: Defining initial parameters
    H0 = 0.02                                                                   #initial inverse Hessian scaling factor
    H = H0 * np.eye(len(x))                                                     #First step is a scaled steepest descent
    outer_buf = np.empty_like(H)                                                #Buffer for outer products to minimize memory-allocation overhead
    E = pot(x)
    gradx = grad(x)
    c1 = 1e-4                                                                   #Armijo condition constant/sufficient decrease parameter
    n_iter = 0


    # 3: Main loop
    for k in range(max_iter):
        # 3.1: Checking convergence criterium
        if np.max(np.abs(gradx)) < tol:
            break


        # 3.2: Calculating directional derivative (scaling search direction vector p such that the largest displacement becomes max_step if neccessary)
        p = -H @ gradx
        max_disp = np.max(np.linalg.norm(p.reshape(shape), axis=-1))
        was_clamped = max_disp > max_step
        if was_clamped:
            p *= max_step / max_disp
        direc_deriv = gradx @ p
        
        if direc_deriv >= 0:                                                #Resetting the Hessian matrix with a scaled steepest descent iteration
            H = H0 * np.eye(len(x))                                         #if it becomes nonsensical
            p = -H @ gradx
            max_disp = np.max(np.linalg.norm(p.reshape(shape), axis=-1))
            if max_disp > max_step:
                p *= max_step / max_disp
            direc_deriv = gradx @ p


        # 3.3: Line search to obtain step length alpha
        alpha = 1.0
        for j in range(20):
            s = alpha * p
            E_new = pot(x + s)
            if E_new <= E + c1 * alpha * direc_deriv:                       #Checking Armijo condition (sufficient decrease in potential energy)
                break
            
            denom = 2.0 * (E_new - E - direc_deriv * alpha)                 #Applying quadratic interpolation of E(alpha) to estimate optimal step length
            if denom > 1e-14 and not was_clamped:
                alpha_interp = -(direc_deriv * alpha * alpha) / denom
                alpha = np.clip(alpha_interp, 0.1 * alpha, 0.5 * alpha)
            else:                                                           #Fallback to standard backtracking halving if quadratic interpolation is 
                alpha *= 0.5                                                #ill-conditioned

        
        # 3.4: Updating parameters
        x += s

        E = E_new

        gradx_new = grad(x)
        
        y = gradx_new - gradx

        ys = y @ s
        if ys > 1e-14:
            rho = 1.0 / ys
            Hy = H @ y
            yHy = y @ Hy
            np.multiply.outer(s, s, out=outer_buf)
            H += (rho + rho * rho * yHy) * outer_buf
            np.multiply.outer(s, Hy, out=outer_buf)
            H -= rho * outer_buf
            H -= rho * outer_buf.T

        gradx = gradx_new

        n_iter += 1
        if keep_steps:
            step_list.append(x.reshape(shape).copy())


    # 4: Returning coordinates
    if print_convergence and np.max(np.abs(gradx)) >= tol:
        warnings.warn("BFGS optimizer reached its max iteration count without converging, returning non-optimal coordinates...", RuntimeWarning)
    elif print_convergence:
        print(f"BFGS optimizer converged in {n_iter} iterations, returning optimal coordinates...")

    final_coords = x.reshape(shape)

    if not keep_steps:
        return final_coords
    return final_coords, np.asarray(step_list, dtype=np.float64)


def validate_bfgs_data(x_arr, potential_derivative_func, potential_func, max_iter, tol, max_step):
    # 1. Validate x (convert before checking .shape so lists/tuples work)
    try:
        x_arr = np.asarray(x_arr, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError("x must be convertible to a floating-point NumPy array.") from e

    if x_arr.ndim != 2 or x_arr.shape[1] != 3:
        raise ValueError(f"x must have shape (N, 3), got {x_arr.shape}.")
    if x_arr.shape[0] == 0:
        raise ValueError("x must contain at least one particle (N >= 1).")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must not contain NaN or infinite values.")

    # 2. Validate callables
    if not callable(potential_derivative_func):
        raise TypeError("potential_derivative_func must be callable.")
    if not callable(potential_func):
        raise TypeError("potential_func must be callable.")

    # 3. Validate max_iter (reject booleans since isinstance(True, int) is True)
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")

    # 4. Validate floating-point parameters
    for name, val in [("tol", tol), ("max_step", max_step)]:
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise TypeError(f"{name} must be a real number, got {type(val).__name__}.")
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}.")
        if val <= 0.0:
            raise ValueError(f"{name} must be > 0, got {val}.")




from collections import deque
from collections.abc import Callable
from src.main_code.potentials_calculations import (
    cluster_potential_gradient,
    cluster_potential,
    lennard_jones_derivative,
    lennard_jones_potential,
)
import numpy as np


def LBFGS(
    x: np.ndarray,
    potential_derivative_func: Callable = lennard_jones_derivative,
    potential_func: Callable = lennard_jones_potential,
    tol: float = 1e-5,
    max_iter: int = 1000,
    max_step: float = 0.1,
    m: int = 10,
    keep_steps: bool = False,
    print_convergence = True
    ):

    '''
    Minimize the potential energy of an atomic cluster using the Limited-Memory
    Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm with two-loop recursion, 
    quadratic-interpolated Armijo backtracking, and per-atom step capping.
    https://doi.org/10.1007/978-0-387-40065-5

    :param x: array_like, shape (N, 3)
        Initial 3D coordinates of the N particles in the cluster.
    :param potential_derivative_func: callable
        Function that calculates the derivative of the pair potential with respect
        to inter-particle distance r. Default is lennard_jones_derivative.
    :param potential_func: callable
        Function that calculates the pair potential energy as a function of
        inter-particle distance r. Default is lennard_jones_potential.
    :param tol: float
        Convergence threshold for the maximum absolute gradient component on any
        particle. Default is 1e-5.
    :param max_iter: int
        Maximum number of BFGS optimization iterations. Default is 1000.
    :param max_step: float
        Maximum allowed spatial displacement per particle in a single iteration to
        prevent numerical instability from large initial forces. Default is 0.1.
    :param m : int
        Number of past curvature pairs (s, y) retained in history. Default is 10.
    :param keep_steps: bool
        If True, records and returns the trajectory of coordinates across all
        iterations alongside the final coordinates. Default is False.
    :returns: ndarray, shape (N, 3) or tuple of (ndarray, ndarray)
        Returns the optimized 3D coordinates of the N particles at a local energy 
        minimum, shape (N, 3).
        If keep_steps is True, also returns step_list, which has shape 
        (n_iter + 1, N, 3) and contains the coordinates at each intermediate iteration.
    '''

    # 1: Bookkeeping
    shape = x.shape
    x = np.array(x, dtype=float).ravel()

    if keep_steps:
        step_list = [x.reshape(shape).copy()]

    def pot(x_flat: np.ndarray) -> float:
        return cluster_potential(x_flat.reshape(shape), potential_func)

    def grad(x_flat: np.ndarray) -> np.ndarray:
        return cluster_potential_gradient(x_flat.reshape(shape), potential_derivative_func).ravel()


    # 2: Defining initial parameters and history buffers
    H0 = 0.02                                                                   #Initial inverse Hessian scaling factor
    c1 = 1e-4                                                                   #Armijo condition constant/sufficient decrease parameter

    s_hist: deque[np.ndarray] = deque(maxlen=m)                                 #Fixed-length ring buffers storing recent (s, y, rho) curvature pairs
    y_hist: deque[np.ndarray] = deque(maxlen=m)
    rho_hist: deque[float] = deque(maxlen=m)
    alpha_buf = np.empty(m, dtype=float)

    E = pot(x)
    gradx = grad(x)

    n_iter = 0


    # 3: Main loop
    for k in range(max_iter):
        # 3.1: Checking convergence criterium
        if np.max(np.abs(gradx)) < tol:
            break


        
        # 3.2: Two-loop recursion to compute directional derivative
        q = gradx.copy()
        hist_len = len(s_hist)


        #3.2.1: Looping backward through curvature history
        for idx in range(hist_len - 1, -1, -1):
            a = rho_hist[idx] * (s_hist[idx] @ q)
            alpha_buf[idx] = a
            q -= a * y_hist[idx]

        if hist_len > 0:                                                        #Dynamic Barzilai-Borwein initial Hessian scaling
            y_last = y_hist[-1]
            gamma = (s_hist[-1] @ y_last) / (y_last @ y_last)
            r = gamma * q
        else:
            r = H0 * q


        #3.2.2: Looping forward through curvature history
        for idx in range(hist_len):
            beta = rho_hist[idx] * (y_hist[idx] @ r)
            r += (alpha_buf[idx] - beta) * s_hist[idx]

        p = -r

        if gradx @ p >= 0.0:                                                    #Resetting the search direction vector p with a scaled steepest 
            s_hist.clear()                                                      #descent iteration if neccessary and wiping corrupted history
            y_hist.clear()
            rho_hist.clear()
            p = -H0 * gradx

        max_disp = np.max(np.linalg.norm(p.reshape(shape), axis=-1))            #Scaling search direction vector p such that the largest displacement 
        was_clamped = max_disp > max_step                                       #becomes max_step if neccessary
        if was_clamped:
            p *= max_step / max_disp

        direc_deriv = gradx @ p


        # 3.4: Line search to obtain step length alpha
        alpha = 1.0
        for _ in range(20):
            s = alpha * p
            E_new = pot(x + s)
            if E_new <= E + c1 * alpha * direc_deriv:                           #Checking Armijo condition (sufficient decrease in potential energy)
                break

            denom = 2.0 * (E_new - E - direc_deriv * alpha)                     #Applying quadratic interpolation of E(alpha) to estimate optimal step length
            if denom > 1e-14 and not was_clamped:
                alpha_interp = -(direc_deriv * alpha * alpha) / denom
                alpha = np.clip(alpha_interp, 0.1 * alpha, 0.5 * alpha)
            else:                                                               #Fallback to standard backtracking halving if quadratic interpolation is
                alpha *= 0.5                                                    #ill-conditioned


        # 3.5: Updating parameters and curvature history
        x += s
        E = E_new
        gradx_new = grad(x)

        y = gradx_new - gradx
        ys = y @ s
        if ys > 1e-14:
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / ys)

        gradx = gradx_new
        n_iter += 1
        if keep_steps:
            step_list.append(x.reshape(shape).copy())


    # 4: Returning coordinates
    if print_convergence and np.max(np.abs(gradx)) >= tol:
        warnings.warn("L-BFGS optimizer reached its max iteration count without converging, returning non-optimal coordinates...", RuntimeWarning)
    elif print_convergence:
        print(f"L-BFGS optimizer converged in {n_iter} iterations, returning optimal coordinates...")

    final_coords = x.reshape(shape)

    if not keep_steps:
        return final_coords
    return final_coords, np.asarray(step_list, dtype=np.float64)