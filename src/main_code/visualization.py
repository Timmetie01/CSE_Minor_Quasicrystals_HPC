import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from src.main_code.potentials_calculations import lennard_jones_potential, lennard_jones_derivative, oscillating_pair_potential, oscillating_pair_derivative, cluster_potential
from src.main_code.local_optimizers.fire import FIRE
from data.data_handling import read_LJwales_data, read_database, ALLOWED_POTENTIALS

def plot_potential(epsilon=1, sigma=1, frequency=1, phase_shift=0):
    '''
    Plots the Lennard-Jones and Oscillating Pair potentials. Yet to implement cutoff-OP.

    :param float epsilon:
        The parameter 'epsilon' for the Lennard-Jones potential, representing well-depth
    :param float sigma:
            The parameter 'sigma' for the Lennard-Jones potential, representing the distance at which the potential is zero
    :param float frequency:
            The parameter 'frequency' for the Oscillating Pair potential, representing at which frequency the cosine term changes over r.
    :param float phase_shift:
            The parameter 'phase_shift' for the Oscillating Pair potential, representing the phase-shift in the cosine term.
    '''
    r = np.arange(0.01, 6, 0.001)
    lj_arr = lennard_jones_potential(r, epsilon, sigma)
    op_arr = oscillating_pair_potential(r, frequency, phase_shift)
    plt.plot(r, lj_arr, color='darkblue', label='Lennard-Jones')
    plt.plot(r, op_arr, color='firebrick', label='Oscillating Pair')
    plt.ylabel("Potential (Dead Racoons)")
    plt.xlabel("Distance (??)")
    plt.grid(ls='--')
    plt.legend()
    plt.title('Potential (r)')
    plt.ylim(np.min(np.array([lj_arr,op_arr])) * 1.2, np.min(np.array([lj_arr,op_arr])) * -1.5)
    plt.show()


def plot_FIRE_convergence(coords_init, potential_func=lennard_jones_potential, potential_derivative=lennard_jones_derivative, showplot=True):
    """
    Plots the convergence of the FIRE optimizer by keeping all intermediate states stored. As such can use a lot of data on large cluste sizes and slow convergence.

    :param np.ndarray, nx3 coords_init:
        The initial coordinates which will be optimmized while logging intermediate steps
    :param Callable potential_func:
        The function that returns the potential as it is called with argument 'r' or array-like 'r'
    :param Callable potential_derivative:
        The derivative of the potneital function w.r.t. the interparticle distance r. Takes as argument only r, either float or array-like.
    :param bool, default=True showplot:
        Chooses if the plot is shown after drawing it. Setting this to fals allows more plotting in the figure before showing it.
    """

    final_coords, step_array = FIRE(coords_init, potential_derivative, True)

    step_count = step_array.shape[0]
    potential_array = np.zeros((step_count))

    for i in range(step_count):
        potential_array[i] = cluster_potential(step_array[i,:,:], potential_func)

    plt.plot(potential_array, color='darkblue')
    if showplot:
        plt.grid(ls='--')
        plt.ylim(np.min(potential_array) * 1.1, 0)
        plt.xlabel('Iteration Number')
        plt.ylabel('Potential Energy')
        plt.title('Potential Energy during FIRE iterations')
        plt.show()


def plot_cluster(coords, potential_str="LJ", potential_params=(1,1), show_bonds=True, 
                 atom_radius=0.4, bond_radius=0.05, atom_color="steelblue", bond_color="lightgray", 
                 sphere_resolution=20, bond_resolution=20, smooth_shading=True):
    """
    3-dimensionally plots a cluster of atoms, optionally also showing bonds. 
    These bonds are approximated by finding the corresponding radius for the potential well depth(s), 
    and if the distance from a particle to another particle is within a certain tolerance to that corresponding radius the bond will be drawn.

    :param np.ndarray, nx3 coords:
        The coordinates of the atoms to be plotted.
    :param str, default="LJ" potential_str:
        The potential type used for the bond distance calculation. Should be in ALLOWED_POTENTIALS in data.datahandling. If show_bonds is false, does not matter.
    :param tuple, default=(1,1) potential_params:
        The parameters of the corresponding potential. Either (epsilon, sigma) or (frequency, phase_shift). If show_bonds is false, does not matter.
    :param float atom_radius:
        The radius of the atoms as plotted. Only visually important.
    :param float bond_radius:
        The radius of the cylinders representing the bonds. Only visually important.
    :param str atom_color:
        The color used for the atoms.
    :param str or (str, str, str) bond_color:
        The color used for the bonds. If a tuple is given and the OP-potential is used, the three different wells get the corresponding colors from closest te furthest.
    :param int, default=20 sphere_resolution:
        The amount of vertices to make a sphere. Only has a visual impact.
    :param int, default=20 bond_resolution:
        The amount of vertices to make a bond cylinder. Only has a visual impact.
    :param bool, default=True smooth_shading:
        Enables smooth shading if true.
    """

    cloud = pv.PolyData(coords)
    cloud["radius"] = np.ones(len(coords))*atom_radius

    if type(bond_color) != tuple:
        bond_color = (bond_color, bond_color, bond_color)

    spheres = cloud.glyph(scale="radius", geom=pv.Sphere(theta_resolution=20, phi_resolution=sphere_resolution), orient=False)
    plotter = pv.Plotter()
    plotter.add_mesh(spheres, color=atom_color, smooth_shading=smooth_shading)

    if potential_str == "LJ" and show_bonds:

        lines, bond_points = find_lj_bonds(coords, potential_params[0], potential_params[1])
        bond_lines = pv.PolyData(bond_points, lines=lines)
        cylinders = bond_lines.tube(radius=bond_radius, n_sides=bond_resolution)
        plotter.add_mesh(cylinders, color=bond_color[0], smooth_shading=True)
    elif (potential_str == "OP" or potential_str == "raw_OP") and show_bonds:
        bond_tuple = find_op_bonds()
        cylinders = bond_lines.tube(radius=bond_radius, n_sides=bond_resolution)
        for i in range(3):
            bond_lines = pv.PolyData(bond_tuple[i][1], lines=bond_tuple[i][0])
            plotter.add_mesh(cylinders, color=bond_color[i], smooth_shading=True)  
    elif show_bonds and potential_str not in ALLOWED_POTENTIALS:
        raise ValueError(f"Choose one of the allowed potentials. You chose {potential_str}, but only {ALLOWED_POTENTIALS} are allowed.")

    plotter.show()

def find_bonds_at_distance(coords, bond_dist, tol=0.05):
    '''
    Compares the distance for each possible atom pair. If abs(r - bond_dist) < tol, it will be registered as a 'bond' and returned as a line segment connecting the coordinate pair.

    :param np.ndarray, nx3 coords:
        The coordinate array for which the bonds will be found.
    :param float bond_dist:
        The distance at which two atoms are considered to have bonded.
    :param float, default=0.05 tol:
        The maximum absolute difference between the bond distance and atom distance for atoms to still be considered bonded.
    :returns tuple[np.ndarray, np.ndarray]:
        - **lines** (*np.ndarray, nx3*): VTK-style connectivity array where each row is `[2, idx1, idx2]`, representing a 2-point line segment indexing into `bond_points`.
        
        - **bond_points** (*np.ndarray, nx2x3*): Flattened array of 3D endpoint coordinates for all detected bonds.
    '''
    bonds = np.empty((0,2,3))
    for i, coord1 in enumerate(coords):
            for coord2 in coords[i+1:]:
                if np.abs(np.linalg.norm(coord1 - coord2) - bond_dist) < tol:
                    bonds = np.append(bonds, np.array([[coord1, coord2]]), 0)
    
    bond_points = bonds.reshape(-1, 3)
    num_bonds = len(bonds)
    idx = np.arange(2 * num_bonds).reshape(-1, 2)
    lines = np.column_stack((np.full(num_bonds, 2), idx))
    return lines, bond_points


def find_lj_bonds(coords, epsilon=1, sigma=1, tol=0.05):
    '''
    Finds atom pairs separated by the equilibrium bond distance of a Lennard-Jones potential r_eq = 2^(1/6)*sigma.

    :param np.ndarray, nx3 coords:
        The coordinate array for which the bonds will be found.
    :param float, default=1 epsilon:
        Depth of the potential well (unused in equilibrium distance calculation, kept for potential parameter consistency).
    :param float, default=1 sigma:
        Distance at which the inter-particle potential is zero.
    :param float, default=0.05 tol:
        The maximum absolute difference between the equilibrium bond distance and atom distance for atoms to still be considered bonded.
    :returns tuple[np.ndarray, np.ndarray]:
        - **lines** (*np.ndarray, nx3*): VTK-style connectivity array where each row is [2, idx1, idx2], representing a 2-point line segment indexing into bond_points.
        
        - **bond_points** (*np.ndarray, nx2x3*): Flattened array of 3D endpoint coordinates for all detected bonds.
    '''
    bond_dist = 2 ** (1/6) * sigma
    return find_bonds_at_distance(coords, bond_dist, tol)


def find_op_bonds(coords, frequency=1, phase_shift=0, tol=0.05):
    '''
    Finds atom pairs separated by the equilibrium bond distances of an oscillating pair potential.

    .. warning::
        This function is not yet implemented and will raise a NotImplementedError.

    :param np.ndarray, nx3 coords:
        The coordinate array for which the bonds will be found.
    :param float, default=1 frequency:
        Spatial oscillation frequency of the pair potential.
    :param float, default=0 phase_shift:
        Phase shift of the oscillating potential.
    :param float, default=0.05 tol:
        The maximum absolute difference between a potential well distance and atom distance for atoms to still be considered bonded.
    :raises NotImplementedError:
        Always raised as the oscillating pair potential bond detection is not yet implemented.
    '''
    raise NotImplementedError
    
    return (find_bonds_at_distance(coords, bond_dist[0], tol), find_bonds_at_distance(coords, bond_dist[1], tol), find_bonds_at_distance(coords, bond_dist[2], tol))

def plot_wales_cluster(N_atoms: int, show_bonds=True, 
                 atom_radius=0.4, bond_radius=0.05, atom_color="steelblue", bond_color="lightgray", 
                 sphere_resolution=20, bond_resolution=20, smooth_shading=True):
    """
    3-dimensionally plots a cluster of atoms from the Lennard-Jones-optimized Wales dataset, optionally also showing bonds. 
    These bonds are approximated by finding the corresponding radius for the LJ-potential well depth(s), 
    and if the distance from a particle to another particle is within a certain tolerance to that corresponding radius the bond will be drawn.

    :param int N_atoms:
        The amount of atoms for which the data should be read and to be plotted.
    :param float atom_radius:
        The radius of the atoms as plotted. Only visually important.
    :param float bond_radius:
        The radius of the cylinders representing the bonds. Only visually important.
    :param str atom_color:
        The color used for the atoms.
    :param str or (str, str, str) bond_color:
        The color used for the bonds. If a tuple is given and the OP-potential is used, the three different wells get the corresponding colors from closest te furthest.
    :param int, default=20 sphere_resolution:
        The amount of vertices to make a sphere. Only has a visual impact.
    :param int, default=20 bond_resolution:
        The amount of vertices to make a bond cylinder. Only has a visual impact.
    :param bool, default=True smooth_shading:
        Enables smooth shading if true.
    """
    coords = read_LJwales_data(N_atoms)
    plot_cluster(coords, "LJ", (1, 1), show_bonds, atom_radius, bond_radius, atom_color, bond_color, sphere_resolution, bond_resolution, smooth_shading)


def plot_from_database(N_atoms: int, potential_str="LJ", potential_params=(1,1), method_str=None, show_bonds=True, 
                 atom_radius=0.4, bond_radius=0.05, atom_color="steelblue", bond_color="lightgray", 
                 sphere_resolution=20, bond_resolution=20, smooth_shading=True):
    """
    3-dimensionally plots a cluster of atoms, optionally also showing bonds. 
    These bonds are approximated by finding the corresponding radius for the potential well depth(s), 
    and if the distance from a particle to another particle is within a certain tolerance to that corresponding radius the bond will be drawn.

    :param int N_atoms:
        The amount of atoms for which the data should be read and to be plotted.
    :param str, default="LJ" potential_str:
        The potential type used for the bond distance calculation. Should be in ALLOWED_POTENTIALS in data.datahandling. If show_bonds is false, does not matter.
    :param tuple, default=(1,1) potential_params:
        The parameters of the corresponding potential. Either (epsilon, sigma) or (frequency, phase_shift). If show_bonds is false, does not matter.
    :param float atom_radius:
        The radius of the atoms as plotted. Only visually important.
    :param float bond_radius:
        The radius of the cylinders representing the bonds. Only visually important.
    :param str atom_color:
        The color used for the atoms.
    :param str or (str, str, str) bond_color:
        The color used for the bonds. If a tuple is given and the OP-potential is used, the three different wells get the corresponding colors from closest te furthest.
    :param int, default=20 sphere_resolution:
        The amount of vertices to make a sphere. Only has a visual impact.
    :param int, default=20 bond_resolution:
        The amount of vertices to make a bond cylinder. Only has a visual impact.
    :param bool, default=True smooth_shading:
        Enables smooth shading if true.
    """
    coords = read_database(N_atoms, potential_str, potential_params, method_str)
    plot_cluster(coords, potential_str, potential_params, show_bonds, atom_radius, bond_radius, atom_color, bond_color, sphere_resolution, bond_resolution, smooth_shading)


#plot_potential(1/4, 1, 4, 1)

#coords = np.random.random((13,3)) * 3
#plot_FIRE_convergence(coords)

# min = 0
# for i in range(50):
#     coords = np.random.random((13,3)) * 3
#     init_coords = coords
#     coords = FIRE(coords, lennard_jones_derivative)
#     pot = cluster_potential(coords)
#     if pot < min:
#         best_coords = coords
#         best_init_coords = init_coords
#         min = pot

#plot_FIRE_convergence(best_init_coords)
#plot_cluster(FIRE(best_coords, lennard_jones_derivative))


if __name__ == "__main__":
    plot_wales_cluster(5)
    plot_wales_cluster(13)
    plot_wales_cluster(55)
    plot_wales_cluster(147)

    plot_from_database(5)
    plot_from_database(13)
    plot_from_database(50)
