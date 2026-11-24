import numpy as np
import pytest
import pyvista as pv
from src.main_code.visualization import find_bonds_at_distance, find_lj_bonds


def test_basic_pair_detection():
    """Finds bonds only between pairs matching bond_dist within tolerance."""
    # Equilateral triangle of side 1.0, plus one far-away atom
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
        [10.0, 10.0, 10.0],
    ])
    lines, bond_points = find_bonds_at_distance(coords, bond_dist=1.0, tol=1e-5)

    # 3 pairs in the triangle should bond; the 4th atom should not
    assert lines.shape == (3, 3)
    assert bond_points.shape == (6, 3)

    segments = bond_points.reshape(-1, 2, 3)
    lengths = np.linalg.norm(segments[:, 0] - segments[:, 1], axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_vtk_connectivity_format():
    """Verifies VTK line format [2, idx1, idx2] and point indexing."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
    ])
    lines, bond_points = find_bonds_at_distance(coords, bond_dist=2.0, tol=0.01)

    expected_lines = np.array([
        [2, 0, 1],
        [2, 2, 3],
    ])
    np.testing.assert_array_equal(lines, expected_lines)
    assert lines.dtype in (np.int32, np.int64)


def test_tolerance_boundary():
    """Checks strict inequality abs(r - bond_dist) < tol."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.04, 0.0, 0.0],  # diff = 0.04 < 0.05 -> included
        [0.0, 1.06, 0.0],  # diff = 0.06 > 0.05 -> excluded
    ])
    lines, bond_points = find_bonds_at_distance(coords, bond_dist=1.0, tol=0.05)

    assert len(lines) == 1
    np.testing.assert_allclose(bond_points[1], [1.04, 0.0, 0.0])


def test_no_bonds_found():
    """Returns empty arrays with valid shapes when no pairs match."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    lines, bond_points = find_bonds_at_distance(coords, bond_dist=1.0, tol=0.05)

    assert lines.shape == (0, 3)
    assert bond_points.shape == (0, 3)


def test_pyvista_polydata():
    """Ensures output plugs directly into pv.PolyData without VTK errors."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    lines, bond_points = find_bonds_at_distance(coords, bond_dist=1.0, tol=0.01)

    mesh = pv.PolyData(bond_points, lines=lines)
    assert mesh.n_points == 4
    assert mesh.n_lines == 2


# Tests for find_lj_bonds

def test_lj_equilibrium_distance():
    """Finds bonds separated by r_eq = 2^(1/6) * sigma."""
    sigma = 1.5
    r_eq = (2 ** (1 / 6)) * sigma

    coords = np.array([
        [0.0, 0.0, 0.0],
        [r_eq, 0.0, 0.0],         # Exact equilibrium -> bonded
        [0.0, sigma, 0.0],        # Zero-crossing distance (not r_eq) -> not bonded
    ])
    lines, bond_points = find_lj_bonds(coords, epsilon=2.5, sigma=sigma, tol=1e-4)

    assert len(lines) == 1
    np.testing.assert_allclose(bond_points[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(bond_points[1], [r_eq, 0.0, 0.0])


def test_lj_default_parameters():
    """Verifies default sigma=1 corresponds to r_eq ≈ 1.12246."""
    r_eq_default = 2 ** (1 / 6)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r_eq_default, 0.0, 0.0],
    ])
    lines, _ = find_lj_bonds(coords, tol=1e-6)
    assert len(lines) == 1