import numpy as np
from src.main_code.potentials_calculations import cluster_potential

def transform_LJwales_to_numpy():
    '''
    Reads the txt-like files from https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ/tables.150.html and saves them as numpy arrays for easier access.
    '''
    for i in range(3, 151):
        data = np.loadtxt(fr"data\LJ_data_wales\{i}", usecols=(0, 1, 2), dtype=np.float64)
        np.save(fr"data\LJ_data_wales\{i}.npy", data)

def read_LJwales_data(atom_count:int):
    '''
    https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ/tables.150.html

    :param atom_count: int
        For which amount of atoms the function reads the coordinates as reported by the source.
    :returns: np.ndarray (atom_count x 3)
        The coordinates leading to minimum Lennard-Jones potential as reported by the source.
    '''
    if atom_count != int(atom_count):
        raise FileNotFoundError(f"The atom count to be loaded should be an integer value, it was {atom_count}.")
    if atom_count > 150 or atom_count < 1:
        raise FileNotFoundError(f"Only cluster sizes from 1 to 150 atoms are available for this function, you tried {atom_count}.")
    if atom_count == 2:
        return np.array([[0,0,0],[1,0,0]])
    if atom_count == 1:
        return np.array([[0,0,0]])

    return np.load(fr"data\LJ_data_wales\{atom_count}.npy")
