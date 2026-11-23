# CSE_Minor_Quasicrystals_HPC

Python code for simulating quasicrystal-clusters, implementing parallelism and built to operate on the Delftblue Supercomputer.

Developed for the Computational Science & Engineering minor at the TU Delft, course TW3725TU - Final Minor Project-part B.


## Code Structure
Following the PYPA standard: https://www.pypa.io/en/latest/specifications/ , src-layout

### src/main_code/
(Yet to rename to 'optimizers')
contains:

_potentials_calculations.py:_

Functions used for calculating the potential and its derivative as a function of the radius between atoms.
Also has functions used for calculating the potential of a cluster or the gradient of the potential of a cluster.

_atom_class.py:_

Contains the main AtomCluster class, integrating all parts of the code like global/local optimization. Also contains complementary classes like Potential. To be called by users of the package

_local_optimizers/fire.py:_

Implements the Fast Inertial Relaxation Engine algorithm, locally optimizing the potential of any atom cluster.
https://link.aps.org/doi/10.1103/PhysRevLett.97.170201

_local_optimizers/bfgs.py:_

Implements the (Low-memory and original) Broyden–Fletcher–Goldfarb–Shanno algorithm, locally optimizing the potential of any atom cluster.
https://doi.org/10.1093/imamat/6.3.222, https://doi.org/10.1093/comjnl/13.3.317, https://doi.org/10.1090/s0025-5718-1970-0258249-6, https://doi.org/10.1090/S0025-5718-1970-0274029-X

_global_optimizers/basin_hopping.py:_



_global_optimizers/genetic_algorithm.py:_



_global_optimizers/parallel_tempering.py:_



_global_optimizers/simulated_annealing.py:_



### src/benchmarking/



### tests/

All tests are collected in this folder. Functions in main_code should have tests written for them, and before opening a pull request tests should pass.

### data/






