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


def highscores(new_data:np.ndarray, method_str:str, potential_str='LJ', potential_parameters=(None,None)):
    '''
    Function to track the global minima we obtained. It compares to our own previous bests (if that exists).
    If 'data' has a lower potential than our current best it is saved, a message is printed to the terminal
    and the records log is updated to include the current best potential, the date at which this happened, the method etc.

    :param new_data: np.ndarray, Nx3
        blabla
    :param potential_sts: string, 'LJ' or 'OP'
        etcetc
    :potential_parameters: tuple of floats or ints

    :param method_str: string
        smthng
    '''

    atom_count = new_data.size[0]
    new_best = False
    first_attempt=False

    if potential_str == 'LJ':
        pass
    elif potential_str == 'OP':
        pass
    else:
        raise ValueError

    try:
        old_record = np.load(fr"data\group_records\{atom_count}.npy")
    except FileNotFoundError:
        first_attempt=True

    if not first_attempt:
        pass
        #if cluster_potential(new_data, potential_func) < cluster_potential(old_record, potential_func)


