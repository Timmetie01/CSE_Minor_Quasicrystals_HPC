import data.data_handling
import numpy as np
import pytest
from data.data_handling import read_LJwales_data, RECORDS_DIR, get_potential_func
from src.main_code.potentials_calculations import cluster_potential
from scipy.spatial.distance import pdist

def test_data_log_validity():
    pass

def test_LJ_wales_correctness():
    assert np.all(read_LJwales_data(1) == np.array([0,0,0]))
    
    assert np.all(read_LJwales_data(10) == np.array([
    [-0.112678561569957,   1.154350068036701,  -0.194840194577019],
    [ 0.455529927957375,  -0.141698933988423,   1.074987039970359],
    [-1.076845089999501,  -0.472850203002737,  -0.000759561321676],
    [ 0.772283646029137,   0.650565133523509,   0.318426126854726],
    [-0.156611774846419,  -0.917108951862921,   0.505406908904027],
    [-0.704034080497556,   0.331528029107990,  -0.717967548406147],
    [ 0.749396644213383,  -0.438279561655135,   0.048302291736783],
    [-0.145155893839899,  -0.631595300103604,  -0.579684879295956],
    [ 0.417694325594403,   0.318313250559209,  -0.692983073948749],
    [-0.199579143040966,   0.146776469285409,   0.239112890083651]]))


def extract_energy(line: str) -> float:
    for part in line.split("|"):
        part = part.strip()
        if part.startswith("E =") or part.startswith("E="):
            return float(part.split("=")[1])
    raise ValueError(f"No 'E =' field found in line: {line}")

def test_records_database_integrity():
    if not RECORDS_DIR.exists():
        pytest.skip("data/records does not exist yet.")

    #No leftover Git conflict markers in any log file
    for log_file in RECORDS_DIR.rglob("*.log"):
        text = log_file.read_text(encoding="utf-8")
        assert "<<<<<<<" not in text and ">>>>>>>" not in text, (
            f"Unresolved Git conflict markers found in {log_file}"
        )

    for potential_dir in RECORDS_DIR.iterdir():
        if not potential_dir.is_dir():
            continue
        assert potential_dir.name in data.data_handling.ALLOWED_POTENTIALS, f"Unexpected potential folder: {potential_dir}"

        for param_dir in potential_dir.iterdir():
            if not param_dir.is_dir():
                continue
            params = tuple(float(p) for p in param_dir.name.split("_"))
            pot_func = get_potential_func(potential_dir.name, params)

            # Check directory naming under methods/
            methods_dir = param_dir / "methods"
            method_dirs = []
            if methods_dir.exists():
                for d in methods_dir.iterdir():
                    if d.is_dir():
                        assert d.name in data.data_handling.ALLOWED_METHODS, f"Unknown method folder: {d}"
                        method_dirs.append(d)

            # Collect all (npy_folder, by_n_log) pairs
            pairs = [(param_dir / "overall", param_dir / "overall_by_N.log")]
            for m_dir in method_dirs:
                pairs.append((m_dir, methods_dir / f"{m_dir.name}_by_N.log"))

            energies_by_folder = {}  # folder_name -> {n_atoms: actual_energy}

            for npy_dir, by_n_log in pairs:
                if not npy_dir.exists() and not by_n_log.exists():
                    continue

                assert by_n_log.exists(), f"Missing {by_n_log} for {npy_dir}"
                lines = by_n_log.read_text(encoding="utf-8").splitlines()
                logged_ns = set()
                folder_energies = {}

                for idx, line in enumerate(lines):
                    n_atoms = idx + 1
                    if not line.strip():
                        continue
                    logged_ns.add(n_atoms)

                    npy_path = npy_dir / f"{n_atoms}.npy"
                    assert npy_path.exists(), f"Line {n_atoms} in {by_n_log} has no matching {npy_path}"

                    coords = np.load(npy_path)
                    #Physical validity checks
                    assert coords.shape == (n_atoms, 3), f"{npy_path}: expected ({n_atoms}, 3), got {coords.shape}"
                    assert np.isfinite(coords).all(), f"{npy_path} contains NaN or Inf!"
                    if n_atoms >= 2:
                        assert pdist(coords).min() > 1e-4, f"{npy_path} has overlapping atoms!"

                    actual_e = float(cluster_potential(coords, pot_func))
                    logged_e = extract_energy(line)
                    assert actual_e == pytest.approx(logged_e, abs=1e-9), (
                        f"{by_n_log} line {n_atoms}: log E={logged_e} != actual E={actual_e}"
                    )
                    folder_energies[n_atoms] = actual_e

                #Ensure no unlogged .npy files exist
                if npy_dir.exists():
                    npy_ns = {int(p.stem) for p in npy_dir.glob("*.npy")}
                    assert npy_ns == logged_ns, f"Orphan .npy or missing log lines in {npy_dir}"

                energies_by_folder[npy_dir.name] = folder_energies

            #Cross-check: overall must be <= every method, and at least one method must equal overall
            overall_e = energies_by_folder.get("overall", {})
            method_energy_dicts = [v for k, v in energies_by_folder.items() if k != "overall"]

            for n_atoms, e_oa in overall_e.items():
                matching_methods = [
                    m_e[n_atoms] for m_e in method_energy_dicts if n_atoms in m_e
                ]
                assert matching_methods, f"overall/{n_atoms}.npy exists, but no method has N={n_atoms}"
                assert e_oa <= min(matching_methods) + 1e-12, (
                    f"overall/{n_atoms}.npy ({e_oa}) is worse than best method ({min(matching_methods)})"
                )
                assert e_oa == pytest.approx(min(matching_methods), abs=1e-9), (
                    f"overall/{n_atoms}.npy ({e_oa}) doesn't match any method's best ({min(matching_methods)})"
                )