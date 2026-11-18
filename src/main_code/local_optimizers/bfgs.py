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

    shape = x.shape
    x = np.array(x, dtype=float).flatten()

    def pot(x_flat): return cluster_potential(x_flat.reshape(shape), potential_func)
    def grad(x_flat): return cluster_potential_gradient(x_flat.reshape(shape), potential_derivative_func).ravel()

    H = 0.02 * np.eye(len(x))
    E = pot(x)
    gradx = grad(x)

    for k in range(max_iter):
        if np.max(np.abs(gradx)) < tol:
            break
        
        p = -H @ gradx

        max_disp = np.max(np.linalg.norm(p.reshape(shape), axis=-1))
        if max_disp > max_step:
            p *= max_step / max_disp

        alpha = 1.0
        c1 = 1e-4
        direc_deriv = gradx @ p
        for j in range(20):
            s = alpha * p
            E_new = pot(x + s)
            if E_new <= E + c1 * alpha * direc_deriv:
                break
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
            H += (rho + rho * rho * yHy) * np.outer(s, s) - rho * (np.outer(s, Hy) + np.outer(Hy, s))

        gradx = gradx_new

    return x.reshape(shape)