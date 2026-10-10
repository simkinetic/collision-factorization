[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20403094.svg)](https://doi.org/10.5281/zenodo.20403094)


# Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator

This repository contains the official MATLAB implementation of the numerical framework presented in the paper *Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator* (monatomic gases, this branch; releases v1.x). The polyatomic extension is released separately as v2.0.0 on the `polyatomic` branch.

The codebase provides a complete suite to compute, compress, and evaluate the nonlinear Boltzmann collision operator using a basis of associated Laguerre polynomials and spherical harmonics. It features the singular 5D quadrature engine, the sparse Coordinate (COO) geometric routing architecture, and OpenMP-accelerated tensor contraction algorithms.

## Repository Structure

* **`src/`**: Core library containing the object-oriented MATLAB framework (`SpectralBasis`, `GeneralCollisionTensor`, `ScatteringKernel`) and the `SHL/` sub-library for spherical harmonic and Wigner 3-$j$ evaluations.
* **`src/mex/`**: C++ source files for the heavily optimized, parallelized tensor contraction algorithms and quadrature evaluations.
* **`src/precalc/`**: Precomputed serialized tensors (`.mat`) used in the paper: hard spheres ($\gamma=1$) at $K_{\max}=4$, $L_{\max}=2,\dots,13$ and at $K_{\max}=6,8$, $L_{\max}=12$; Maxwell molecules ($\gamma=0$) at $K_{\max}=4$, $L_{\max}=2,4,6$. Other tensors are generated with `precompute_collision_operator.m` (see the assembly costs below).
* **`revision/`**: Scripts added for the revised version of the paper (see section 4 below).
* **`tests/`**: Automated unit tests validating the spectral basis properties, spherical harmonics, quadrature reproduction, and analytical Wang Chang-Uhlenbeck (WCU) eigenvalues.
* **Root Directory**: High-level tutorials and benchmark scripts designed to directly reproduce the figures and tables presented in the manuscript.

---

## Compilation Instructions (MEX & OpenMP)

To achieve the execution speeds reported in the manuscript, the computational bottlenecks are implemented in C++ and compiled as MATLAB MEX functions. These functions rely heavily on **OpenMP** for multi-threading.

You must compile the `.cpp` files located in the `src/mex/` directory before running the benchmarks.

### macOS (Homebrew)

Apple's default Clang compiler does not natively bundle OpenMP. If you are using macOS, install `libomp` via Homebrew:

```bash
brew install libomp

```

Next, ensure your environment variables are configured to link the Homebrew OpenMP libraries. Add the following to your `~/.zprofile` or `~/.zshrc`:

```bash
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CXXFLAGS="-I/opt/homebrew/opt/libomp/include"

```

Finally, open MATLAB, navigate to the `src/mex/` directory, and compile each C++ file using the following `mex` command:

```matlab
% Compile the quadrature sum-factorization engine
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" compute_rtensor_sumfac_mex.cpp

% Compile the tensor contraction algorithms
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" angular_first_collision_kernel_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" radial_first_collision_kernel_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" dense_tensor_kernel_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" naive_collision_kernel_mex.cpp

```

### Linux / Windows

On systems where GCC or MSVC natively supports OpenMP, you can generally compile the files directly in MATLAB by passing the respective OpenMP flags:

```matlab
% Linux (GCC) example
mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" compute_rtensor_sumfac_mex.cpp

```

---

## Reproducing Paper Results

The root directory contains scripts to reproduce the core results, validations, and benchmarks from the manuscript.

### 1. Physical Validation & Tutorials

These scripts validate the operator against analytical kinetic theory limits:

* `tutorial_collisions_bwu_solution.m`: Simulates the transient relaxation of a Maxwell gas, validating the operator against the analytical Bobylev-Krook-Wu (BKW) solution and confirming exact conservation of invariants.
* `tutorial_collisions_spectral_properties.m`: Computes the Jacobian of the collision operator to extract the Wang Chang-Uhlenbeck eigenvalue spectrum.
* `tutorial_chapman_enscog_decay_rates.m`: Extracts the infinite-order Chapman-Enskog viscosity limits for Hard Sphere gases by block-diagonalizing the linearized operator.
* `tutorial_stress_relaxation.m`: Simulates the nonlinear anisotropic stress relaxation of a Hard Sphere gas.

### 2. Computational Benchmarks

These scripts quantify the algorithmic performance of the factorization:

* `benchmark_quadrature_error.m`: Evaluates the spectral convergence of the 2D Duffy singular quadrature scheme as a function of grid padding.
* `benchmark_memory_compression.m`: Storage of the factorized operator (impact tensor + COO Gaunt tensor) versus the dense Cartesian tensor for $L_{\max}=2,\dots,16$; needs no precomputed tensors (Figure 8).
* `benchmark_collision_contraction_scaling.m`: Measures the execution time of the contraction orders (term-by-term, radial-first, angular-first) against the dense Cartesian tensor for $K_{\max}=4$, $L_{\max}=4,\dots,13$ (Figure 7 of the paper).

### 3. Generating Tensors

* `precompute_collision_operator.m`: Generates the continuous Wigner-Eckart factorized tensors for a specified spectral resolution and collision kernel, saving the resulting data structures to the `src/precalc/` directory.

Assembly of the hard-sphere impact tensor with padding 16 takes, on a 12-core Apple M2 Pro: 192 s ($K_{\max}=6$, $L_{\max}=8$), 563 s (4, 10), 695 s (6, 10), 898 s (8, 10), 2067 s (6, 12); $K_{\max}=4$, $L_{\max}=13$ with padding 20 takes about 80 minutes (Table 17 of the paper).

### 4. Scripts of the Revised Version (JCP revision, 2026)

Each script prints its result and writes a CSV; the table or figure of the paper it reproduces is given in brackets.

* `revision_maxwell_degree_rule.m`: Maxwell-molecule degree selection rule $2k_1+l_1 = 2k_2+l_2+2k_3+l_3$ on the assembled tensor, against quadrature padding (Table 6).
* `revision_coefficient_verification.m`: linearized hard-sphere operator: Chapman-Cowling bracket matrices and the Sonine sequences of the viscosity and heat-conductivity corrections (Table 9).
* `revision_afp62_eigenvalues.m`: generalized eigenvalues of the linearized hard-sphere blocks against Alterman, Frankowski & Pekeris (1962) (Table 10).
* `revision_galilean_hardsphere.m`: Galilean-invariance residuals for Maxwell molecules and hard spheres (Table 13).
* `revision_initial_data_projection.m`: projection errors of the bi-Gaussian and discontinuous initial data of Wang & Cai (2019) (Table 11).
* `revision_initial_data_dynamics.m`: relaxation of these initial data with the angular-first kernel in `ode45` (Figure 6, Table 12).
* `revision_timing_kscan.m`: execution time of the contraction orders for $K_{\max}=4,6,8$ at $L_{\max}=12$ (Table 14).
* `revision_export_impact_tensor.m` and `revision/compare_hanke_torrilhon.py`: entry-by-entry comparison of the Maxwell-molecule impact tensor with that of Hanke & Torrilhon (2023); the latter is read from their Zenodo file `SmatrixAMX1099887766554433221100A` (10.5281/zenodo.7242835), directory given by the environment variable `HT_DIR` (Table 7).
* `revision/opcount.py`: structural counts and operation counts of the contraction orders (Tables 3-5).
* `revision/memory_ladder.py`: memory at $L_{\max}=6,\dots,16$ and at the total-degree truncation $2k+l \le 20$ (Tables 15-16).

The tutorials `tutorial_collisions_bwu_solution.m` and `tutorial_stress_relaxation.m` produce Figures 4 and 5; `tutorial_collisions_spectral_properties.m` produces Figure 3 and Table 8.

---

## Running the Unit Tests

To verify that the environment is configured correctly and the underlying mathematical libraries (e.g., spherical harmonics, basis exactness) are functioning as intended, run the master test script from the root directory:

```matlab
run_tests

```