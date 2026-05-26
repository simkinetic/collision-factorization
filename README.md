[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20403094.svg)](https://doi.org/10.5281/zenodo.20403094)


# Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator

This repository contains the official MATLAB implementation of the numerical framework presented in the paper *Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator*.

The codebase provides a complete suite to compute, compress, and evaluate the nonlinear Boltzmann collision operator using a basis of associated Laguerre polynomials and spherical harmonics. It features the singular 5D quadrature engine, the sparse Coordinate (COO) geometric routing architecture, and OpenMP-accelerated tensor contraction algorithms.

## Repository Structure

* **`src/`**: Core library containing the object-oriented MATLAB framework (`SpectralBasis`, `GeneralCollisionTensor`, `ScatteringKernel`) and the `SHL/` sub-library for spherical harmonic and Wigner 3-$j$ evaluations.
* **`src/mex/`**: C++ source files for the heavily optimized, parallelized tensor contraction algorithms and quadrature evaluations.
* **`src/precalc/`**: Precomputed serialized tensors (`.mat`) for various angular resolutions ($L_{\max}$) and collision potentials (Maxwell molecules $\gamma=0$, Hard Spheres $\gamma=1$).
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
* `benchmark_memory_compression.m`: Calculates the storage footprint reduction achieved by the COO geometric routing versus the dense Cartesian baseline.
* `benchmark_collision_contraction_scaling.m`: Measures the wall-clock execution time of the different tensor contraction strategies (Naive, Radial-First, Angular-First) to demonstrate the reported hardware accelerations.

### 3. Generating Tensors

* `precompute_collision_operator.m`: Generates the continuous Wigner-Eckart factorized tensors for a specified spectral resolution and collision kernel, saving the resulting data structures to the `src/precalc/` directory.

---

## Running the Unit Tests

To verify that the environment is configured correctly and the underlying mathematical libraries (e.g., spherical harmonics, basis exactness) are functioning as intended, run the master test script from the root directory:

```matlab
run_tests

```