import numpy as np
from datetime import datetime
import os
from pathlib import Path
from src.main_code.potentials_calculations import cluster_potential, make_lennard_jones_potential, make_oscillating_pair_potential

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
WALES_DIR = DATA_DIR / "LJ_data_wales"
RECORDS_DIR = DATA_DIR / "records"

def transform_LJwales_to_numpy():
    '''
    Reads the txt-like files from https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ/tables.150.html and saves them as numpy arrays for easier access. No need to run, since this has been done once.
    '''
    for i in range(3, 151):
        data = np.loadtxt(fr"data\LJ_data_wales\{i}", usecols=(0, 1, 2), dtype=np.float64)
        np.save(WALES_DIR / f"{i}.npy", data)

def read_LJwales_data(atom_count:int):
    '''
    https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ/tables.150.html

    :param int atom_count: 
        For which amount of atoms the function reads the coordinates as reported by the source.
    :returns np.ndarray (Nx3): 
        The coordinates leading to minimum Lennard-Jones potential as reported by the source.
    '''
    if atom_count != int(atom_count):
        raise ValueError(f"The atom count to be loaded should be an integer value, it was {atom_count}.")
    if atom_count > 150 or atom_count < 1:
        raise FileNotFoundError(f"Only cluster sizes from 1 to 150 atoms are available for this function, you tried {atom_count}.")

    #Atom count of 1 or 2 is not included in the Wales data, but elementary, so it will return the solutions to avoid errors.
    if atom_count == 2:
        return np.array([[0,0,0],[1,0,0]], dtype=np.float64)
    if atom_count == 1:
        return np.array([[0,0,0]], dtype=np.float64)

    return np.load(WALES_DIR / f"{atom_count}.npy")


def get_potential_func(potential_str: str, potential_parameters: tuple):
    """Build and return the potential energy function for 'LJ' or 'OP', based on information passed to the highscores function."""
    if potential_str.upper() == "LJ":
        return make_lennard_jones_potential(potential_parameters[0], potential_parameters[1])
    if potential_str.upper() == "OP":
        return make_oscillating_pair_potential(potential_parameters[0], potential_parameters[1])
    raise ValueError("Choose either 'LJ' (Lennard-Jones) or 'OP' (Oscillating Pair) potential.")


def append_log(log_path: str, line: str) -> None:
    """Append a single line to a chronological log file.
    Creates parent folders if needed."""
    parent = os.path.dirname(log_path)
    #Any non-empty string evaluates to True, otherwise False
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def update_atom_count_log(log_path: str, atom_count: int, line: str) -> None:
    """Write 'line' at line number 'atom_count' (starting from one). 
    Adds earlier lines if needed. Used for atom-count ordered log file."""
    parent = os.path.dirname(log_path)

    #Any non-empty string evaluates to True, otherwise False
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\r\n") for ln in f.readlines()]
    else:
        lines = []

    while len(lines) < atom_count:
        lines.append("")

    lines[atom_count - 1] = line

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def highscores(
    new_data: np.ndarray,
    method_str: str,
    potential_str: str,
    potential_parameters: tuple,
    extra_info: str,
    tol=1e-12
    ):
    '''
    Checks if the input coordinates are a new highscore (either overall, or with that GO), and if so logs it. Saves the coordinates and keeps track of both a chronological log file and one ordered by cluster size.

    :param np.ndarray  (n,3) new_data:
        The newly obtained minimum to be compared to previously calculated local/global minima.
    :param literal, "basin_hopping", "basin_hopping_ml", "genetic_algorithm", "parallel_tempering"or "simulated_annealing" method_str:
        The global optimizer used for generating the minima. Must be given to ensure proper logging.
    :param literal, "LJ" or "OP" potential_str:
        Specifies the potential type for which the minimum was found. Either Lennard-Jones or Oscillating Pair
    :param tuple, (epsilon, sigma) or (frequency, phase shift) potential_parameters:
        The parameters of the potential for which the minimum was found. For Lennard-Jones (1,1) is standard, for Oscillating Pair the parameters are more important
    :param str extra_info:
        A string of extra information that will be written at the end of the line inthe log files. Including things like execution time, the local optimizer used, other parameters etc.
    :param float, default=1e-12 tol:
        The difference by which a new minimum must beat an old minimum to be counted. To avoid floating-point error playing a significant role.
    
    '''

    coords = np.asarray(new_data, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 1:
        raise ValueError(f"Expected (N, 3) array with N > 0, got {coords.shape}.")
    potential_parameters = tuple(potential_parameters)

    n_atoms = coords.shape[0]
    pot_func = get_potential_func(potential_str, potential_parameters)
    new_energy = cluster_potential(coords, pot_func)

    #Example: Folder for Lennard-Jones cluster with epsilon=sigma=1 will be called \LJ\1.0_1.0\
    param_tag = "_".join(str(float(p)) for p in potential_parameters)
    potential_tag = f"{potential_str.upper()}"

    #Avoid users from ending lines by removing \n and \r\n and \r
    extra_info = " ".join(extra_info.splitlines())

    method_str = method_str.lower()
    if method_str not in ["basin_hopping", "basin_hopping_ml", "genetic_algorithm", "parallel_tempering", "simulated_annealing"]:
        raise ValueError(f'The method_str should be one of the following: "basin_hopping", "basin_hopping_ml", "genetic_algorithm", "parallel_tempering" or "simulated_annealing", but got {method_str}')

    #Base folder for this potential and the parameter pair
    param_dir = RECORDS_DIR / potential_tag / param_tag
    methods_dir = param_dir / "methods"

    #File paths for actual npy files
    overall_npy = param_dir / "overall" / f"{n_atoms}.npy"
    method_npy = methods_dir / method_str / f"{n_atoms}.npy"

    #File paths for the log files
    overall_chrono_log = param_dir / "overall.log"
    overall_by_n_log = param_dir / "overall_by_N.log"
    method_chrono_log = methods_dir / f"{method_str}.log"
    method_by_n_log = methods_dir / f"{method_str}_by_N.log"

    prev_overall_energy = float(cluster_potential(np.load(overall_npy), pot_func)) if os.path.exists(overall_npy) else None
    prev_method_energy = float(cluster_potential(np.load(method_npy), pot_func)) if os.path.exists(method_npy) else None

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    first_ever_overall = prev_overall_energy is None
    first_ever_method = prev_method_energy is None
    broke_method = (prev_method_energy is None) or (new_energy < prev_method_energy - tol)
    broke_overall = (prev_overall_energy is None) or (new_energy < prev_overall_energy - tol)

    #If it is the first time running this set of parameters with the potential, all directories need to be made
    if first_ever_overall:
        os.makedirs(os.path.dirname(overall_npy), exist_ok=True)
        os.makedirs(os.path.dirname(method_npy), exist_ok=True)
        np.save(overall_npy, coords)
        np.save(method_npy, coords)

        print(f"We have our first attempt for the current configuration, so kinda a record (N={n_atoms}, {potential_tag}_{param_tag}, {method_str}): E = {new_energy:.10f}")
        line = f"{current_time:^20} | N={n_atoms:4d} | {potential_tag}_{param_tag:<15} | E = {new_energy:<5.10f} | method = {method_str:<20} | {'FIRST ATTEMPT':^22} | {extra_info}"
        append_log(overall_chrono_log, line)
        update_atom_count_log(overall_by_n_log, n_atoms, line)
        append_log(method_chrono_log, line)
        update_atom_count_log(method_by_n_log, n_atoms, line)

    #If it is the first time this method tries these parameters, only the methods directories need to be created
    elif first_ever_method:
        os.makedirs(os.path.dirname(method_npy), exist_ok=True)
        np.save(method_npy, coords)

        print(f"We have our first attempt for the current method, so kinda a method record (N={n_atoms}, {potential_tag}_{param_tag}, {method_str}): E = {new_energy:.10f}")
        line = f"{current_time:^20} | N={n_atoms:4d} | {potential_tag}_{param_tag:<15} | E = {new_energy:<5.10f} | method = {method_str:<20} | {'FIRST ATTEMPT (METHOD)':^22} | {extra_info}"
        append_log(method_chrono_log, line)
        update_atom_count_log(method_by_n_log, n_atoms, line)

        if broke_overall:
            os.makedirs(os.path.dirname(overall_npy), exist_ok=True)
            np.save(overall_npy, coords)
            print(f"We broke our own overall record!! ({method_str}, N={n_atoms}): E = {new_energy:.10f} (prev: {prev_overall_energy:.8f})")
            overall_line = f"{current_time:^20} | N={n_atoms:4d} | {potential_tag}_{param_tag:<15} | E = {new_energy:<5.10f} | method = {method_str:<20} | prev = {prev_overall_energy:<5.10f} | {'OVERALL BEST':^22} | {extra_info}"
            append_log(overall_chrono_log, overall_line)
            update_atom_count_log(overall_by_n_log, n_atoms, overall_line)

    #Otherwise no directories need to be made
    else:
        if broke_method:
            os.makedirs(os.path.dirname(method_npy), exist_ok=True)
            np.save(method_npy, coords)
            print(f"We broke our own record with this method ({method_str}, N={n_atoms}): E = {new_energy:.10f} (prev: {prev_method_energy:.8f})")
            method_line = f"{current_time:^20} | N={n_atoms:4d} | {potential_tag}_{param_tag:<15} | E = {new_energy:<5.10f} | prev = {prev_method_energy:<5.10f} | {'METHOD BEST':^22} | {extra_info}"
            append_log(method_chrono_log, method_line)
            update_atom_count_log(method_by_n_log, n_atoms, method_line)

        if broke_overall:
            os.makedirs(os.path.dirname(overall_npy), exist_ok=True)
            np.save(overall_npy, coords)
            print(f"We broke our own overall record!! ({method_str}, N={n_atoms}): E = {new_energy:.10f} (prev: {prev_overall_energy:.8f})")
            overall_line = f"{current_time:^20} | N={n_atoms:4d} | {potential_tag}_{param_tag:<15} | E = {new_energy:<5.10f} | method = {method_str:<20} | prev = {prev_overall_energy:<5.10f} | {'OVERALL BEST':^22} | {extra_info}"
            append_log(overall_chrono_log, overall_line)
            update_atom_count_log(overall_by_n_log, n_atoms, overall_line)

    if potential_str.upper() == "LJ" and n_atoms <= 150 and potential_parameters == (1, 1):
        wales_coords = read_LJwales_data(n_atoms)
        wales_energy = float(cluster_potential(wales_coords, pot_func))

        if new_energy < wales_energy - tol:
            print(f"We broke the Wales datasets record!!!!!! (Wales LJ N={n_atoms}, {method_str}): E = {new_energy:.10f} < {wales_energy:.10f}")
            wales_line = f"{current_time:^20} | N={n_atoms:4d} | {potential_tag}_{param_tag:<15} | E = {new_energy:<5.10f} | method = {method_str:<20} | BEAT WALES ({wales_energy:<5.10f}) | {extra_info}"
            append_log(overall_chrono_log, wales_line)
            update_atom_count_log(overall_by_n_log, n_atoms, wales_line)
            append_log(method_chrono_log, wales_line)
            update_atom_count_log(method_by_n_log, n_atoms, wales_line)

    return new_energy