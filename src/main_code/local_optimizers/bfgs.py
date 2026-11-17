from src.main_code.potentials_calculations import cluster_potential_gradient
import numpy as np


def BFGS(x, potential_derivative_func: callable):
    '''
    !!!DRAFT - NOT WORKING OR TESTED YET!!!

    x: coordinates

    s: displacement of coordinates

    alpha: step length (chosen using a line search)

    p: search direction vector/minimiser
    
    '''

    H = np.eye(len(x))

    gradx = cluster_potential_gradient(x, potential_derivative_func)

    p = -H @ gradx

    alpha = 1

    s = alpha * p

    y = cluster_potential_gradient(x+s, potential_derivative_func) - gradx

    x += s

    H = (np.eye(len(s)) - np.outer(s, y) / (y @ s)) @ H @ (np.eye(len(s)) - np.outer(y, s) / (y @ s)) + np.outer(s, s) / (y @ s)