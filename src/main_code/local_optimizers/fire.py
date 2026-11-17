import numpy as np
from src.main_code.potentials_calculations import cluster_potential, cluster_potential_gradient, lennard_jones_derivative




def FIRE(coords: np.ndarray, potential_derivative_func: callable,
         Nmax=10_000, Nmin=5, f_inc=1.1, f_dec=0.5, 
         alpha0=0.1, f_alpha=0.99, err_tol=1e-5, 
         dt=0.002, max_step=0.05):
    '''
    Minimize the potential energy of a particle cluster using the Fast Inertial
    Relaxation Engine (FIRE) algorithm with Velocity Verlet integration.
    https://journals-aps-org.tudelft.idm.oclc.org/prl/pdf/10.1103/PhysRevLett.97.170201 

    :param coords: array_like, shape (N, 3)
        Initial 3D coordinates of the N particles in the cluster.
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
        Optimized 3D coordinates of the N particles at a local energy minimum.
    '''
    
    dt_max = 10 * dt
    
    vel = np.zeros_like(coords)
    alpha = alpha0
    Npos = 0

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

        i += 1

    return coords
