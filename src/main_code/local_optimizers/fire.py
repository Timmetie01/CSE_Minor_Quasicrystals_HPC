import numpy as np
from src.main_code.potentials_calculations import cluster_potential_gradient
from collections.abc import Callable

def FIRE(coords: np.ndarray, potential_derivative_func: Callable, keep_steps: bool = False,
         Nmax: int = 10_000, Nmin: int = 5, f_inc: float = 1.1, f_dec: float = 0.5, 
         alpha0: float = 0.1, f_alpha: float = 0.99, err_tol: float = 1e-5 , 
         dt: float = 0.002, max_step: float = 0.05):
    '''
    Minimize the potential energy of a particle cluster using the Fast Inertial
    Relaxation Engine (FIRE) algorithm with Velocity Verlet integration.
    https://journals-aps-org.tudelft.idm.oclc.org/prl/pdf/10.1103/PhysRevLett.97.170201 

    :param coords: array_like, shape (N, 3)
        Initial 3D coordinates of the N particles in the cluster.
    :param potential_derivative_func: callable
        Function that calculates the derivative of the pair potential with respect
        to inter-particle distance r.
    :param keep_steps: boolean, optional
        Decides if the function should keep the full history of steps made. 
        If so, the function will return a tuple of (coord_loc_min, array_of_coords). 
        This can lead to arrays of many Mb's, use with caution.
        Defaults to False
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
        Optimized 3D coordinates of the N particles at a local energy minimum.
    '''

    validate_fire_data(coords, potential_derivative_func, Nmax, Nmin, f_inc, f_dec, alpha0, err_tol, f_alpha, dt, max_step)
    
    dt_max = 10 * dt
    
    vel = np.zeros_like(coords)
    alpha = alpha0
    Npos = 0

    if keep_steps:
        step_arr = np.empty((0,len(coords),3), dtype=np.float64)

    F = - cluster_potential_gradient(coords, potential_derivative_func)

    i = 0
    while i < Nmax:
        if np.max(np.abs(F)) < err_tol:
            break

        vel = vel + 0.5 * F * dt
        step = vel * dt
        max_disp = np.max(np.linalg.norm(step, axis=-1))
        if max_disp > max_step:
            step *= max_step / max_disp
            vel = step / dt

        coords = coords + vel * dt
        F = -cluster_potential_gradient(coords, potential_derivative_func)
        vel = vel + 0.5 * F * dt

        P = np.sum(F * vel)

        if P > 0:
            Npos += 1
            if Npos >= Nmin:
                dt = min(dt * f_inc, dt_max)
                alpha = alpha * f_alpha

        else:
            Npos = 0
            dt = f_dec * dt
            vel = np.zeros_like(coords)
            alpha = alpha0

        F_norm = np.linalg.norm(F)
        if F_norm > 0:
            vel = (1 - alpha) * vel + alpha * (F / F_norm) * np.linalg.norm(vel)

        if keep_steps:
            step_arr = np.append(step_arr, np.array([coords]), 0)
        i += 1

    if i == Nmax:
        raise RuntimeWarning("FIRE optimizer reached its max iteration count without converging, returning non-optimal coordinates...")

    if keep_steps:
        return coords, step_arr
    return coords

def validate_fire_data(coords: np.ndarray, potential_derivative_func: Callable,
         Nmax: int, Nmin: int, f_inc: float, f_dec: float, 
         alpha0: float, f_alpha: float, err_tol: float, 
         dt: float, max_step: float):
    # 1. Validate coords
    try:
        coords = np.asarray(coords, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError("coords must be convertible to a floating-point NumPy array.") from e

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}.")
    if coords.shape[0] == 0:
        raise ValueError("coords must contain at least one particle (N >= 1).")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must not contain NaN or infinite values.")

    # 2. Validate callable
    if not callable(potential_derivative_func):
        raise TypeError("potential_derivative_func must be callable.")

    # 3. Validate integer parameters (reject booleans since isinstance(True, int) is True in Python)
    for name, val, min_val in [("Nmax", Nmax, 1), ("Nmin", Nmin, 0)]:
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(val).__name__}.")
        if val < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {val}.")

    # 4. Validate floating-point parameters
    float_params = {
        "f_inc": f_inc,
        "f_dec": f_dec,
        "alpha0": alpha0,
        "f_alpha": f_alpha,
        "err_tol": err_tol,
        "dt": dt,
        "max_step": max_step,
    }
    for name, val in float_params.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise TypeError(f"{name} must be a real number, got {type(val).__name__}.")
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}.")

    # 5. Validate domain-specific mathematical bounds
    if f_inc <= 1.0:
        raise ValueError(f"f_inc must be > 1.0, got {f_inc}.")
    if not (0.0 < f_dec < 1.0):
        raise ValueError(f"f_dec must be strictly between 0 and 1, got {f_dec}.")
    if not (0.0 < alpha0 <= 1.0):
        raise ValueError(f"alpha0 must be in the interval (0, 1], got {alpha0}.")
    if not (0.0 < f_alpha <= 1.0):
        raise ValueError(f"f_alpha must be in the interval (0, 1], got {f_alpha}.")
    if err_tol <= 0.0:
        raise ValueError(f"err_tol must be > 0, got {err_tol}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if max_step <= 0.0:
        raise ValueError(f"max_step must be > 0, got {max_step}.")