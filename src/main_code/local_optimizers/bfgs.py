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

    # Replace your original `shape = x.shape` and `x = ...` lines with:
    shape = x_arr.shape
    x = x_arr.ravel().copy()