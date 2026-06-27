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

## DSMC Polyatomic Collision Kernel (Djordjić et al. 2023)

The `ScatteringKernel` class supports the variable-hard-sphere Borgnakke–Larsen (VHS–BL)
collision-kernel model used by the DSMC community
([Djordjić, Oblapenko, Pavić-Čolić & Torrilhon, *Continuum Mech. Thermodyn.* **35** (2023) 103–119](https://doi.org/10.1007/s00161-022-01167-8)),
both the base DSMC model (eq. 54) and its internal-energy-modulated **extended** model (eq. 43).

### Construction

```matlab
% Base DSMC convex-split kernel: B = |u|^zeta [ omega*K_delta*R^(zeta/2)
%                                             + (1-omega)*delta(r-r')delta(R-R')*e_tr^(zeta/2) ]
K = ScatteringKernel('DSMC', struct('zeta', 0.533, 'delta', 2.01, 'omega', 0.3));
%   zeta  = 2*(1 - s_visc)   relative-speed exponent (or pass 's_visc' instead)
%   delta = internal DOF     (nu = delta/2 - 1 is the internal-Laguerre parameter)
%   omega = p_int in [0,1]   inelastic (non-frozen) collision probability

% Extended model (eq. 43): add internal-energy modulation. Defaults are 0 -> base DSMC.
K = ScatteringKernel('DSMC', struct('zeta',0.53,'delta',2.01,'omega',0.54, ...
        'eta_hat',-0.453, 'zeta_hat',0.965, ...     % non-frozen factor 1 + eta_hat(...)^(zeta_hat/2)
        'eta_hat_f',0.570, 'zeta_hat_f',0.965));     % frozen     factor 1 + eta_hat_f(i^zeta_hat_f + ...)
%   eta_hat, eta_hat_f >= -1/2  (kernel positivity);  zeta_hat, zeta_hat_f >= 0
```

The kernel is then assembled exactly as any other model:

```matlab
Basis = SpectralBasis(K_max, L_max, I_max, K.nu);
T = GeneralCollisionTensor(Basis, K);
T.generate_R_tensor_sumfac(radial_pad, tangential_pad, internal_pad);  % padding beyond exactness
C = T.assemble_full_tensor();
```

The frozen and non-frozen channels are computed by `compute_rtensor_polyatomic_sumfac_mex`
(`kernel_model = 1` non-frozen, `2` frozen) and blended in `GeneralCollisionTensor` as
`omega*C_vhs*R_nf + (1-omega)*C_vhs_frozen*R_fr`. Recompile that MEX after pulling these
changes (same `mex -R2018a … -fopenmp` recipe as above).

### Implementation notes

* **Frozen channel.** Elastic (`I'=I`, `I*'=I*`, `|u'|=|u|`), integrated directly in
  internal-energy space; basis orthonormality carries the Kronecker structure. There is *no*
  `(R,r)`-partition measure in this channel (a common pitfall — adding the Borgnakke–Larsen
  `H_delta` weight double-counts a `nu`-dependent factor and corrupts the internal-heat-flux
  rate).
* **Extended model.** The non-frozen `(r(1-R))^(zeta_hat/2)` weight separates across the
  sum-factorization cascade; the `(I/E)^(zeta_hat/2)` factors are per-quadrature-point scalars.
  Gain and loss share an identical weighting, so conservation (`#null = 5`) is preserved.
  Because of the non-integer internal exponents the extended channels converge *algebraically*
  (padding-controlled), unlike the spectral base channels.

### DSMC validation / transport benchmarks

* `benchmark_dsmc_transport.m`: base DSMC model. Builds the operator for the calorically-perfect
  gases of Table 1 (N₂/O₂/NO/CO/H₂), extracts the shear/bulk/heat-flux production terms, and
  reports the Prandtl number (eq. 39), the bulk/shear ratio ν/μ (eq. 58), and the reachable Pr
  window over ω ∈ [0,1] against the Eucken/frozen/measured values (Tables 2–3).
* `benchmark_dsmc_extended.m`: extended model. Reports Pr and ν/μ at the paper's Table-4 fit
  parameters, then **re-fits** (ω, η̂_f, and ζ̂_f if needed) so that the experimental Prandtl
  number and bulk/shear ratio are reached *simultaneously* in this operator.
* `paper/make_figures.m`: regenerates the validation figures (frozen-Pr accuracy vs.
  Monte-Carlo, convergence, transport reachability) into `paper/figures/`.
* `paper/main.tex`: the accompanying report (theory, methods, validation tables and figures).

---

## Running the Unit Tests

To verify that the environment is configured correctly and the underlying mathematical libraries (e.g., spherical harmonics, basis exactness) are functioning as intended, run the master test script from the root directory:

```matlab
run_tests

```