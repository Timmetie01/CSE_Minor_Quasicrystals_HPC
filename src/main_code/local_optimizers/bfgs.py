from src.main_code.potentials_calculations import cluster_potential_gradient, cluster_potential
import numpy as np


def BFGS(x, potential_derivative_func: callable, potential_func: callable, tol=1e-6, max_iter=1000):
    '''
    !!!DRAFT - NOT WORKING OR TESTED YET!!!

    x: coordinates

    s: displacement of coordinates

    alpha: step length (chosen using a line search)

    p: search direction vector/minimiser
    
    '''

    I = np.eye(len(x))
    H = I
    gradx = cluster_potential_gradient(x, potential_derivative_func)

    for k in range(max_iter):
        if np.linalg.norm(gradx) < tol:
            break
        
        p = -H @ gradx

        alpha = line_search(x, p, gradx, potential_derivative_func, potential_func, c1 = 1e-4, c2 = 0.9)

        s = alpha * p

        gradx_new = cluster_potential_gradient(x+s, potential_derivative_func)

        y = gradx_new - gradx

        x += s

        if ys > 1e-14:
            H = (I - np.outer(s, y) / (y @ s)) @ H @ (I - np.outer(y, s) / (y @ s)) + np.outer(s, s) / (y @ s)

        gradx = gradx_new

    return x




def line_search(x, p, gradx, potential_derivative_func: callable, potential_func: callable, c1 = 1e-4, c2 = 0.9):
    alpha=0
    alpha_trial = 1.0
    E=cluster_potential(x, potential_func)
    E_prev = E

    def zoom(alpha_l, alpha_h, E_l):
        for j in range(20):
            alpha_trial = 0.5 * (alpha_l + alpha_h)
            s = alpha_trial * p
            E_trial = cluster_potential(x + s, potential_func)

            if (E_trial > E + c1 * alpha_trial * (gradx @ p)) or (E_trial >= E_l):
                alpha_h = alpha_trial
            else:
                gradx_trial = cluster_potential_gradient(x + s, potential_derivative_func)
                if abs(gradx_trial @ p) <= c2 * abs(gradx @ p):
                    return alpha_trial
                if (gradx_trial @ p) * (alpha_h - alpha_l) >= 0:
                    alpha_h = alpha_l
                alpha_l = alpha_trial
                E_l = E_trial

        return alpha_l

    for i in range(20):
        s = alpha_trial * p
        E_trial = cluster_potential(x+s, potential_func)

        if (E_trial <= E + c1 * alpha_trial * (gradx @ p)) and (k==0 or E_trial < E_prev):
            gradx_trial = cluster_potential_gradient(x+s, potential_derivative_func)
            if abs(gradx_trial @ p) <= c2 * abs(gradx @ p):
                return alpha_trial
            if (gradx_trial @ p) >= 0:
                return zoom(alpha_trial, alpha, E_trial)

            alpha = alpha_trial
            E_prev = E_trial
            alpha_trial *= 2.0
        else:
            return zoom(alpha, alpha_trial, E_prev)

    return alpha_trial