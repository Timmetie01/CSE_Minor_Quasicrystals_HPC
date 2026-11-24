from pathlib import Path
import re
import shutil
import sys
import numpy as np
from src.main_code.potentials_calculations import (
    cluster_potential,
    make_lennard_jones_potential,
    make_oscillating_pair_potential,
)


def get_potential_from_path(target_path: str):
    '''
    Infers the potential type and parameters (Lennard-Jones or Oscillating Pair each with 2 parameters) from the inputted file path.
    These file paths are created by teh data_handling.py file in the data folder
    '''
    parts = Path(target_path).parts

    #Looking through all directories of the file path
    for i, part in enumerate(parts):
        #Looking for potential title, i.e. Lennard-Jones or Oscillating-Pair. Also ensure directory does not end, so parts[i+1] is valid
        if part.upper() in ("LJ", "OP") and i + 1 < len(parts):
            #The potential is formatted like \1.0_1.0\ so split that correctly
            params = tuple(float(x) for x in parts[i + 1].split("_"))
            if part.upper() == "LJ":
                return make_lennard_jones_potential(params[0], params[1])
            return make_oscillating_pair_potential(params[0], params[1])
    raise ValueError(f"Could not parse potential/parameters from path: {target_path}")


def merge_npy(ours_path: str, theirs_path: str, target_path: str):
    '''
    From two file paths with saved npy arrays, choose the one with lowest potential and store that at the target_path. Infers the potential type and parameters from the file path
    '''
    pot_func = get_potential_from_path(target_path)
    e_ours = float(cluster_potential(np.load(ours_path), pot_func))
    e_theirs = float(cluster_potential(np.load(theirs_path), pot_func))

    #'ours' will be kept in the end, so if 'theirs' is lower just make 'ours' a copy of 'theirs'
    if e_theirs < e_ours:
        shutil.copyfile(theirs_path, ours_path)

def extract_energy(line: str):
    #Context: \b means word boundary, \s means whitespace, * means >=0 characters, \s* means >=0 spaces, E and = are the literal characters
    #The number (the part in regular brackts): the brackes make it a 'group' so match.group returns something. [] brackets mean one of the characters inside these brackets. Question mark marks previous thing as optional
    #\d means any whole single digit number, + means one or more times (together any 'integer' number). idk what ?: does. \. means literal dot, and the end optionally captures potential scientific notation.
    match = re.search(r"\bE\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
    
    #The only group created was the floating point number. If we can't find it we mark it as infinite potential, effectively discarding the file.
    return float(match.group(1)) if match else float("inf")


def merge_by_n_log(ours_path: str, theirs_path: str):
    ours_lines = Path(ours_path).read_text(encoding="utf-8").splitlines()
    theirs_lines = Path(theirs_path).read_text(encoding="utf-8").splitlines()

    merged = []
    for i in range(max(len(ours_lines), len(theirs_lines))):
        l_ours = ours_lines[i] if i < len(ours_lines) else ""
        l_theirs = theirs_lines[i] if i < len(theirs_lines) else ""

        #Non-empty evaluates to True
        if not l_ours.strip():
            merged.append(l_theirs)
        elif not l_theirs.strip():
            merged.append(l_ours)
        else:
            merged.append(l_theirs if extract_energy(l_theirs) < extract_energy(l_ours) else l_ours)

    #Put enters between the lines and merge them. If no lines exist, give empty string
    Path(ours_path).write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")


if __name__ == "__main__":
    mode, _, ours_file, theirs_file, repo_path = sys.argv[1:6]
    #We have 3 different things to merge:
    #1) Chronological log files: these do not need a python script to merge, but in the .gitattributes we can automatically select keep both changes.
    #2) The actual data in the saved arrays. It gets compared regarding potential energy based on the potential used and its parameters from the file path, lowest is kept.
    if mode == "npy":
        merge_npy(ours_file, theirs_file, repo_path)
    #3) The files saved in order of atom count. It compares the reported energies in each file version as stated on each line and keeps the line with the lowest.
    elif mode == "by_n":
        merge_by_n_log(ours_file, theirs_file)

