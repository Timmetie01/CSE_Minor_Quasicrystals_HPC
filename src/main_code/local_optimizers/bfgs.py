from src.main_code.potentials_calculations import cluster_potential_gradient, cluster_potential, lennard_jones_derivative, lennard_jones_potential
import numpy as np


def BFGS(x, potential_derivative_func: callable = lennard_jones_derivative, potential_func: callable = lennard_jones_potential, tol=1e-5, max_iter=1000, max_step=0.1):
    '''
    !!!DRAFT - NOT FULLY TESTED YET!!!

    x: coordinates

    s: displacement of coordinates

    alpha: step length (chosen using a line search)

    p: search direction vector/minimiser
    
    '''

    validate_bfgs_data(x, potential_derivative_func, potential_func, max_iter, tol, max_step)

    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    x = np.array(x, dtype=np.float64).ravel()
    

    def pot(x_flat): return cluster_potential(x_flat.reshape(shape), potential_func)
    def grad(x_flat): return cluster_potential_gradient(x_flat.reshape(shape), potential_derivative_func).ravel()

    H0 = 0.02
    H = H0 * np.eye(len(x))
    outer_buf = np.empty_like(H)
    E = pot(x)
    gradx = grad(x)
    c1 = 1e-4

    for k in range(max_iter):
        if np.max(np.abs(gradx)) < tol:
            break
        
        p = -H @ gradx

        max_disp = np.max(np.linalg.norm(p.reshape(shape), axis=-1))
        was_clamped = max_disp > max_step
        if was_clamped:
            p *= max_step / max_disp

        direc_deriv = gradx @ p
        
        if direc_deriv >= 0:
            H = H0 * np.eye(len(x))
            p = -H @ gradx
            max_disp = np.max(np.linalg.norm(p.reshape(shape), axis=-1))
            if max_disp > max_step:
                p *= max_step / max_disp
            direc_deriv = gradx @ p

        alpha = 1.0
        for j in range(20):
            s = alpha * p
            E_new = pot(x + s)
            if E_new <= E + c1 * alpha * direc_deriv:
                break
            
            denom = 2.0 * (E_new - E - direc_deriv * alpha)
            if denom > 1e-14 and not was_clamped:
                alpha_interp = -(direc_deriv * alpha * alpha) / denom
                alpha = np.clip(alpha_interp, 0.1 * alpha, 0.5 * alpha)
            else:
                alpha *= 0.5

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

        if k == max_iter - 1:
            raise RuntimeWarning("BFGS optimizer reached its max iteration count without converging, returning non-optimal coordinates...")

    return x.reshape(shape)


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
) -> np.ndarray:
    """
    Minimize the potential energy of an atomic cluster using the Limited-Memory
    BFGS (L-BFGS) algorithm with two-loop recursion, quadratic-interpolated
    Armijo backtracking, and per-atom step capping.

    Parameters
    ----------
    x : np.ndarray, shape (N, 3)
        Initial 3D coordinates of the N particles in the cluster.
    potential_derivative_func : Callable, optional
        Derivative of the pair potential with respect to inter-particle distance r.
    potential_func : Callable, optional
        Pair potential function with respect to inter-particle distance r.
    tol : float, optional
        Convergence threshold for the maximum absolute force component. Default is 1e-5.
    max_iter : int, optional
        Maximum number of optimization iterations. Default is 1000.
    max_step : float, optional
        Maximum allowed spatial displacement per particle per iteration. Default is 0.1.
    m : int, optional
        Number of past curvature pairs (s, y) retained in history. Default is 10.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Optimized 3D coordinates of the N particles at a local energy minimum.
    """
    shape = x.shape
    x = np.array(x, dtype=float).ravel()

    def pot(x_flat: np.ndarray) -> float:
        return cluster_potential(x_flat.reshape(shape), potential_func)

    def grad(x_flat: np.ndarray) -> np.ndarray:
        return cluster_potential_gradient(x_flat.reshape(shape), potential_derivative_func).ravel()

    H0 = 0.02
    c1 = 1e-4

    # Fixed-length ring buffers storing recent (s, y, rho) curvature pairs
    s_hist: deque[np.ndarray] = deque(maxlen=m)
    y_hist: deque[np.ndarray] = deque(maxlen=m)
    rho_hist: deque[float] = deque(maxlen=m)
    alpha_buf = np.empty(m, dtype=float)

    E = pot(x)
    gradx = grad(x)

    for k in range(max_iter):
        if np.max(np.abs(gradx)) < tol:
            break

        # --- Two-loop recursion to compute p = -H_k @ gradx ---
        q = gradx.copy()
        hist_len = len(s_hist)

        for idx in range(hist_len - 1, -1, -1):
            a = rho_hist[idx] * (s_hist[idx] @ q)
            alpha_buf[idx] = a
            q -= a * y_hist[idx]

        # Dynamic Barzilai-Borwein initial Hessian scaling (falls back to H0 on step 0)
        if hist_len > 0:
            y_last = y_hist[-1]
            gamma = (s_hist[-1] @ y_last) / (y_last @ y_last)
            r = gamma * q
        else:
            r = H0 * q

        for idx in range(hist_len):
            beta = rho_hist[idx] * (y_hist[idx] @ r)
            r += (alpha_buf[idx] - beta) * s_hist[idx]

        p = -r

        # Descent safeguard: wipe corrupted history and fall back to scaled steepest descent
        if gradx @ p >= 0.0:
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()
            p = -H0 * gradx

        # Per-atom max displacement cap
        max_disp = np.max(np.linalg.norm(p.reshape(shape), axis=-1))
        was_clamped = max_disp > max_step
        if was_clamped:
            p *= max_step / max_disp

        direc_deriv = gradx @ p

        # Armijo backtracking with safeguarded quadratic interpolation
        alpha = 1.0
        for _ in range(20):
            s = alpha * p
            E_new = pot(x + s)
            if E_new <= E + c1 * alpha * direc_deriv:
                break

            denom = 2.0 * (E_new - E - direc_deriv * alpha)
            if denom > 1e-14 and not was_clamped:
                alpha_interp = -(direc_deriv * alpha * alpha) / denom
                alpha = np.clip(alpha_interp, 0.1 * alpha, 0.5 * alpha)
            else:
                alpha *= 0.5

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

    return x.reshape(shape)