[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20403094.svg)](https://doi.org/10.5281/zenodo.20403094)


# Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator

This repository contains the official MATLAB implementation of the numerical framework presented in two companion papers:

* **Monatomic** — *Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator*, Hiemstra, Keßler & Abdelmalik, [arXiv:2605.28475](https://doi.org/10.48550/arXiv.2605.28475).
* **Polyatomic** — *Wigner-Eckart Factorization of the Polyatomic Boltzmann Collision Operator*, which extends the framework with an internal-energy axis (internal truncation $I_{\max}$) and the DSMC Borgnakke–Larsen kernel.

The codebase provides a complete suite to compute, compress, and evaluate the nonlinear Boltzmann collision operator using a basis of associated Laguerre polynomials and spherical harmonics. It features the singular 5D (monatomic) / 9D (polyatomic) quadrature engine, the sparse Coordinate (COO) geometric routing architecture, and OpenMP-accelerated tensor contraction algorithms.

Scripts belonging to the two studies are kept side by side. Anything named `*_polyatomic_*` belongs to the polyatomic paper; the remaining root-level `benchmark_*` and `tutorial_*` scripts are the monatomic artifacts and are retained unchanged.

## Repository Structure

* **`src/`**: Core library containing the object-oriented MATLAB framework (`SpectralBasis`, `GeneralCollisionTensor`, `ScatteringKernel`) and the `SHL/` sub-library for spherical harmonic and Wigner 3-$j$ evaluations.
* **`src/mex/`**: C++ source files for the heavily optimized, parallelized tensor contraction algorithms and quadrature evaluations.
* **`src/precalc/`**: Working directory for the serialized collision tensors (`.mat`). **It ships empty.** Every cache in it is a build product, regenerated on demand — see [Regenerating the precomputed tensors](#regenerating-the-precomputed-tensors).
* **`tests/`**: Automated unit tests validating the spectral basis properties, spherical harmonics, quadrature reproduction, the analytical Wang Chang-Uhlenbeck (WCU) eigenvalues, the polyatomic conservation embedding, and the $I_{\max} = 0$ monatomic collapse.
* **`results/`**: Every script writes here (`*.mat`, `*.csv`, generated `*.tex` tables and figures), and the figure renderers read back out. Its contents are untracked, with one exception: `fig_transport_fits_data.mat` ships, because `plot_transport_fits_paper.m` reads it as input and regenerating it needs the cached operators.
* **Root Directory**: High-level tutorials and benchmark scripts designed to directly reproduce the figures and tables presented in the two manuscripts.

## Requirements

* **MATLAB R2020b or newer.** Developed and reported on **R2023b**; the `-R2018a` MEX interleaved-complex API is required, which is R2018a and later. No toolboxes beyond base MATLAB are needed to run the benchmarks; the test suite uses the **MATLAB unit-test framework** (`matlab.unittest`), which is part of base MATLAB.
* **A C++ compiler that MATLAB recognises**, with **OpenMP**. Verify with `mex -setup C++`.
  * **macOS** — Xcode command-line tools plus Homebrew `libomp` (Apple Clang does not bundle OpenMP). The reported timings are from Apple Clang + `libomp` on a 12-core Apple M2 Pro, 32 GB, macOS 14.4.1.
  * **Linux** — GCC 9 or newer (`-fopenmp` is built in).
  * **Windows** — MSVC 2019 or newer (`/openmp`), or MinGW-w64 GCC.
* **Disk.** A few hundred MB for the polyatomic caches. The optional monatomic legacy sweeps want tens of GB — see their section below.

---

## Compilation Instructions (MEX & OpenMP)

The computational bottlenecks are implemented in C++ and compiled as MATLAB MEX functions, parallelized with **OpenMP**.

**Compiled binaries are deliberately not distributed** — they are platform-specific, and `.gitignore` excludes `*.mex*`. **Nothing in this repository runs until you have compiled all seven sources in `src/mex/`.** They are:

| Source in `src/mex/` | What it does |
| --- | --- |
| `compute_rtensor_sumfac_mex.cpp` | Monatomic 5D singular quadrature, sum-factorized. |
| `compute_rtensor_polyatomic_sumfac_mex.cpp` | Polyatomic 9D singular quadrature, sum-factorized; `kernel_model = 1` non-frozen, `2` frozen. |
| `compute_rtensor_polyatomic_aux_mex.cpp` | Polyatomic 9D quadrature on the auxiliary-Laplace (spectral) path — the production path for the extended kernel. |
| `angular_first_collision_kernel_mex.cpp` | Angular-first contraction ordering (the fastest; the paper's 40.6× result). |
| `radial_first_collision_kernel_mex.cpp` | Radial-first contraction ordering. |
| `naive_collision_kernel_mex.cpp` | Naive streaming contraction. |
| `dense_tensor_kernel_mex.cpp` | Dense Cartesian baseline contraction. |

All seven are needed: the first three to assemble any operator, the last four because the Section 5.6 performance benchmark times all four orderings against each other. `run_tests` will fail with `Undefined function` if any is missing.

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
% Compile the quadrature sum-factorization engines
% (monatomic 5D, and the two polyatomic 9D paths: sum-factorized and auxiliary-Laplace)
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" compute_rtensor_sumfac_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" compute_rtensor_polyatomic_sumfac_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" compute_rtensor_polyatomic_aux_mex.cpp

% Compile the tensor contraction algorithms
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" angular_first_collision_kernel_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" radial_first_collision_kernel_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" dense_tensor_kernel_mex.cpp
mex -R2018a CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp" naive_collision_kernel_mex.cpp

```

### Linux / Windows

GCC and MSVC support OpenMP natively, so no extra library is needed. From `src/mex/`:

```matlab
% Linux (GCC) -- compiles all seven
srcs = dir('*.cpp');
for k = 1:numel(srcs)
    mex('-R2018a', 'CXXFLAGS=$CXXFLAGS -fopenmp', 'LDFLAGS=$LDFLAGS -fopenmp', srcs(k).name);
end
```

```matlab
% Windows (MSVC) -- compiles all seven
srcs = dir('*.cpp');
for k = 1:numel(srcs)
    mex('-R2018a', 'COMPFLAGS=$COMPFLAGS /openmp', srcs(k).name);
end
```

### Verifying the build

```matlab
cd <repo root>
addpath('src', 'src/mex', genpath('src/SHL'));
which compute_rtensor_polyatomic_aux_mex   % must resolve to a .mex* file
run_tests                                   % the real check
```

Recompile **all** of `src/mex/` after pulling changes: the frozen/non-frozen channel treatment is split across `compute_rtensor_polyatomic_sumfac_mex` and `compute_rtensor_polyatomic_aux_mex`, and the two paths must agree to machine precision for base kernels.

---

## Reproducing Paper Results

The root directory contains scripts to reproduce the core results, validations, and benchmarks of both manuscripts.

### 1. Physical Validation & Tutorials (monatomic)

These scripts validate the operator against analytical kinetic theory limits:

* `tutorial_collisions_bwu_solution.m`: Simulates the transient relaxation of a Maxwell gas, validating the operator against the analytical Bobylev-Krook-Wu (BKW) solution and confirming exact conservation of invariants. Builds its tensor in memory; runs on a clean checkout.
* `tutorial_collisions_spectral_properties.m`: Computes the Jacobian of the collision operator to extract the Wang Chang-Uhlenbeck eigenvalue spectrum. Builds its tensor in memory; runs on a clean checkout.
* `tutorial_chapman_enscog_decay_rates.m`: Extracts the infinite-order Chapman-Enskog viscosity limits for Hard Sphere gases by block-diagonalizing the linearized operator. **Requires a monatomic cache that is not shipped** (`collisiontensor_k4_l2_gamma1.00.mat`); regenerate it first with `precompute_collision_operator.m`.
* `tutorial_stress_relaxation.m`: Simulates the nonlinear anisotropic stress relaxation of a Hard Sphere gas. **Requires a monatomic cache that is not shipped** (`collisiontensor_k4_l4_gamma1.00.mat`); regenerate it first with `precompute_collision_operator.m`.

### 2. Computational Benchmarks

* `benchmark_quadrature_error.m`: Evaluates the spectral convergence of the 2D Duffy singular quadrature scheme as a function of grid padding (monatomic). Builds in memory; runs on a clean checkout.
* `benchmark_polyatomic_performance.m`: **This is the script that reproduces the polyatomic paper's storage and contraction results** — Section 5.6 in its entirety: the storage counts and compression (Figure 9, Table 5), the geometry and slice counts $N_G$, $N_T$, $N_S$, and the contraction timing sweep over $(L_{\max}, I_{\max})$ (Figure 10). It runs at the paper's $K_{\max} = 2$ and sweeps $I_{\max} = 0,\ldots,4$, uses the paper's storage model, eq. (30), and builds everything in memory, so it needs no cache and runs on a clean checkout. It reproduces Table 5 digit for digit; the timings are machine-dependent measurements and will land near, not on, the published speedups.

### 3. Monatomic legacy benchmarks (caches not shipped)

The following two scripts are the performance artifacts of the **monatomic** predecessor paper ([arXiv:2605.28475](https://doi.org/10.48550/arXiv.2605.28475)), whose $37.2\times$ contraction speedup the polyatomic paper cites. They are retained unchanged for that reason. **They cannot reproduce any number in the polyatomic paper's Section 5.6**: both are monatomic ($K_{\max} = 4$, no $I_{\max}$ axis), and `benchmark_memory_compression.m` additionally uses a 20-byte-per-nonzero storage model rather than the polyatomic paper's 40-byte-per-transition model of eq. (30). Use `benchmark_polyatomic_performance.m` for Section 5.6.

* `benchmark_memory_compression.m`: Storage footprint of the COO geometric routing versus the dense Cartesian baseline, monatomic, $K_{\max} = 4$, hard spheres.
* `benchmark_collision_contraction_scaling.m`: Wall-clock execution time of the contraction strategies (Dense, Naive, Radial-First, Angular-First), monatomic, $K_{\max} = 4$, hard spheres.

Both load `collisiontensor_k4_l*_gamma1.00.mat` from `src/precalc/`, and **those caches are not distributed**. On a clean checkout each `L_max` prints `[SKIP] File missing`, every curve stays `NaN`, and the first figure then fails in `ylim()` with `MATLAB:rulerFunctions:InvalidNumericLimits`. This is expected. To run them, regenerate the caches first with `precompute_collision_operator.m`, edited to the settings these scripts expect:

```matlab
K_max      = 4;                        % already its default
L_max_list = [2, 4, 6, 8, 10, 12];     % it ships as [2, 4, 6]
gamma      = 1.0;                      % it ships as 0.0 (Maxwell molecules)
```

Budget hours of quadrature and tens of GB on disk; the upper `L_max` entries at the script's `pad = 20` dominate both. The same regeneration (at the relevant single `L_max`) is what the two cache-dependent tutorials in Section 1 need.

---

## Regenerating the precomputed tensors

`src/precalc/` ships **empty**. The two generators are:

* `build_or_load_dsmc_tensor.m` — the **polyatomic** cache layer. Given a configuration struct it builds the operator, or loads it if a matching cache already exists, and persists it to `src/precalc/` with the whole configuration encoded in the filename:
  `collisiontensor_dsmc_k{K}_l{L}_i{I}_z{zeta}_d{delta}_w{omega}_p{rad}-{tan}-{int}[_eh…][_lapNs{Ns}][_pfshift][_etr43|_etr43spec|_noetr].mat`
* `precompute_collision_operator.m` — the **monatomic** batch generator (`collisiontensor_k{K}_l{L}_gamma{γ}.mat`), for the legacy monatomic benchmarks and tutorials.

## Running the Unit Tests

This is the first thing to run after compiling the MEX files. It verifies both that the environment is configured correctly and that the mathematical core behaves as claimed. From the repository root:

```matlab
run_tests
```

`run_tests` adds `src/`, `src/SHL/`, `src/mex/` and `tests/` to the path, discovers every test class in `tests/`, runs them, and prints a results table. It needs no cache — every operator is built in memory at small truncations. Expect **2–5 minutes**.

| Test class | What it checks |
| --- | --- |
| `TestSpectralBasis` | Orthogonality, analytic normalization, and sub-basis extraction of the Laguerre × spherical-harmonic basis. |
| `TestSphericalHarmonics` | Gaunt-coefficient labels and values, and the real/complex harmonic transform. |
| `TestQuadraturePolynomialReproduction` | Exact reproduction of polynomials by the Legendre, Lobatto, and Jacobi rules. |
| `TestWCUEigenvalues` | The Maxwell operator's spectrum against the analytic Wang Chang-Uhlenbeck eigenvalues, to $10^{-10}$. |
| `TestPolyatomicConservation` | Exact enforcement of the five collision invariants, and survival of the Landau-Teller exchange mode, over $\delta \in \{2, 3, 5\}$ — the static half of the paper's Section 5.3. |
| `TestMonatomicModulationInertness` | That the $I_{\max} = 0$ Dirac collapse reproduces the monatomic operator exactly for every kernel branch and on both build paths — the Section 5.2 collapse-point claim, and a regression guard on the Laplace-path correction terms. |

A failure in `TestWCUEigenvalues` or `TestPolyatomicConservation` almost always means a MEX file is stale: recompile all seven and rerun.