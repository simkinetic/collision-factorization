[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20403094.svg)](https://doi.org/10.5281/zenodo.20403094)


# Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator

This repository contains the official MATLAB implementation of the numerical framework presented in two companion papers:

* **Monatomic** — *Wigner-Eckart Factorization of the Spectral Boltzmann Collision Operator*, Hiemstra, Keßler & Abdelmalik, [arXiv:2605.28475](https://doi.org/10.48550/arXiv.2605.28475).
* **Polyatomic** — *Wigner-Eckart Factorization of the Polyatomic Boltzmann Collision Operator*, which extends the framework with an internal-energy axis (internal truncation $I_{\max}$) and the DSMC Borgnakke–Larsen kernel.

The codebase provides a complete suite to compute, compress, and evaluate the nonlinear Boltzmann collision operator using a basis of associated Laguerre polynomials and spherical harmonics. It features the singular 5D (monatomic) / 9D (polyatomic) quadrature engine, the sparse Coordinate (COO) geometric routing architecture, and OpenMP-accelerated tensor contraction algorithms.

Scripts belonging to the two studies are kept side by side. Anything named `*_polyatomic_*`, or listed in the [polyatomic script-to-artifact map](#polyatomic-paper-script-to-artifact-map) below, belongs to the polyatomic paper; the remaining root-level `benchmark_*` and `tutorial_*` scripts are the monatomic artifacts and are retained unchanged.

## Repository Structure

* **`src/`**: Core library containing the object-oriented MATLAB framework (`SpectralBasis`, `GeneralCollisionTensor`, `ScatteringKernel`) and the `SHL/` sub-library for spherical harmonic and Wigner 3-$j$ evaluations.
* **`src/mex/`**: C++ source files for the heavily optimized, parallelized tensor contraction algorithms and quadrature evaluations.
* **`src/precalc/`**: Working directory for the serialized collision tensors (`.mat`). **It ships empty.** Every cache in it is a build product, regenerated on demand — see [Regenerating the precomputed tensors](#regenerating-the-precomputed-tensors).
* **`tests/`**: Automated unit tests validating the spectral basis properties, spherical harmonics, quadrature reproduction, the analytical Wang Chang-Uhlenbeck (WCU) eigenvalues, the polyatomic conservation embedding, and the $I_{\max} = 0$ monatomic collapse.
* **`results/`**: Generated output, **not tracked in git**. The benchmark scripts create this directory and write their results (`*.mat`, `*.csv`, and generated `*.tex` tables) into it; the figure renderers `plot_wcu_limit_paper.m`, `plot_polyrelax_conservation_paper.m`, `plot_polyrelax_sweeps_paper.m` and `plot_polyce_convergence.m` read them back out. A fresh clone has no `results/`: run the producing benchmark first, then its renderer. The [script-to-artifact map](#polyatomic-paper-script-to-artifact-map) gives the pairing and the runtime for each.
* **Root Directory**: High-level tutorials and benchmark scripts designed to directly reproduce the figures and tables presented in the two manuscripts. See the [polyatomic script-to-artifact map](#polyatomic-paper-script-to-artifact-map) for which script produces which polyatomic artifact.

### What is deliberately not in the repository

Three classes of file are excluded by `.gitignore` and must be produced locally:

| Not shipped | How to get it |
| --- | --- |
| Compiled MEX binaries (`*.mexmaci64`, `*.mexa64`, `*.mexw64`) | Compile the seven C++ sources — [Compilation instructions](#compilation-instructions-mex--openmp). Nothing runs before this. |
| Precomputed collision tensors (`src/precalc/*.mat`) | Regenerate — [Regenerating the precomputed tensors](#regenerating-the-precomputed-tensors). |
| All PDFs, including generated figures | Rerun the producing script from the [script-to-artifact map](#polyatomic-paper-script-to-artifact-map). |

---

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

The root directory contains scripts to reproduce the core results, validations, and benchmarks of both manuscripts. For the polyatomic paper, the [script-to-artifact map](#polyatomic-paper-script-to-artifact-map) below is the authoritative list of which script produces which figure and table.

### 1. Physical Validation & Tutorials (monatomic)

These scripts validate the operator against analytical kinetic theory limits:

* `tutorial_collisions_bwu_solution.m`: Simulates the transient relaxation of a Maxwell gas, validating the operator against the analytical Bobylev-Krook-Wu (BKW) solution and confirming exact conservation of invariants. Builds its tensor in memory; runs on a clean checkout.
* `tutorial_collisions_spectral_properties.m`: Computes the Jacobian of the collision operator to extract the Wang Chang-Uhlenbeck eigenvalue spectrum. Builds its tensor in memory; runs on a clean checkout.
* `tutorial_chapman_enscog_decay_rates.m`: Extracts the infinite-order Chapman-Enskog viscosity limits for Hard Sphere gases by block-diagonalizing the linearized operator. **Requires a monatomic cache that is not shipped** (`collisiontensor_k4_l2_gamma1.00.mat`); regenerate it first — see [Monatomic tensors](#monatomic-tensors).
* `tutorial_stress_relaxation.m`: Simulates the nonlinear anisotropic stress relaxation of a Hard Sphere gas. **Requires a monatomic cache that is not shipped** (`collisiontensor_k4_l4_gamma1.00.mat`); regenerate it first — see [Monatomic tensors](#monatomic-tensors).

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

### Which benchmarks need a cache, and which do not

**Cheap — no cache, runs on a clean checkout.** These build everything in memory. Start here.

| Script | Runtime |
| --- | --- |
| `run_tests` | ~2–5 min |
| `benchmark_polyatomic_performance.m` (Table 5, Figs. 9–10) | ~10 min |
| `benchmark_quadrature_error.m` (monatomic) | ~1 min |
| `benchmark_polyatomic_quadrature_error.m` | ~5 min |
| `tutorial_collisions_bwu_solution.m`, `tutorial_collisions_spectral_properties.m`, `tutorial_polyatomic_collisions_wcu_spectra.m`, `tutorial_polyatomic_transient_relaxation.m` | minutes each |

**Self-caching — builds on first run, reuses afterwards.** No manual step: they call `build_or_load_dsmc_tensor.m`, which writes the cache it just built.

| Script | First-run cost |
| --- | --- |
| `benchmark_polyatomic_wcu_limit.m` (Fig. 4) | ~30 min — 7 builds over the $\delta$ sweep |
| `benchmark_polyatomic_shear_rate.m` | ~4 min — 21 builds |
| `benchmark_polyatomic_temperature_relaxation.m` (Figs. 5–7) | ~20 min |
| `benchmark_polyatomic_chapman_enskog.m` (Table 3) | ~10 min algebraic path, ~11 min at `use_laplace = true`, `K_top_ext = 2` |
| `benchmark_polyatomic_spatial_quadrature_convergence.m` (Fig. 3 data) | **~1 h** — the pad-32 reference builds dominate. The 1 kB result, `spatial_quadrature_convergence_data.mat`, **is shipped**, so `plot_spatial_quadrature_convergence_paper.m` renders Figure 3 immediately and you only need this run to reproduce the data itself. |
| `benchmark_dsmc_transport.m`, `benchmark_dsmc_extended.m` | ~10–20 min each |

**Requires caches built in advance — will error on a clean checkout.** The three Section 5.5 transport scripts load four operators per gas by literal filename (`load(fullfile(pc, …))`), so a missing cache is a hard `Unable to read file` error rather than a rebuild:

* `benchmark_polyatomic_transport_fits.m` (Table 4, Fig. 8 data)
* `benchmark_polyatomic_closure_attribution.m`
* `benchmark_polyatomic_quadrature_path_difference.m`

Build their twelve operators (three gases × four channel branches) first. This takes on the order of **1–2 hours** and about 200 MB:

```matlab
addpath(pwd);
gases = { ...
  struct('name','N2', 'zeta',0.534, 'delta',2.01, 'zhf',0.3),   ...
  struct('name','CO', 'zeta',0.53,  'delta',2.01, 'zhf',0.965), ...
  struct('name','H2', 'zeta',0.608, 'delta',1.94, 'zhf',0.965)};

for g = gases
    G = g{1};
    base = struct('K_max',2,'L_max',2,'I_max',2, 'zeta',G.zeta, 'delta',G.delta, ...
                  'rad_pad',16,'tan_pad',16,'int_pad',8, 'conserve',true);

    % 1. non-frozen base (algebraic path)
    c = base; c.omega = 1;                    build_or_load_dsmc_tensor(c);
    % 2. non-frozen modulation amplitude (spectral Laplace path, N_lambda = 24)
    c = base; c.omega = 1; c.eta_hat = 1; c.zeta_hat = 0.965;
              c.laplace = true; c.laplace_Ns = 24;
                                              build_or_load_dsmc_tensor(c);
    % 3. frozen base (spectral)
    c = base; c.omega = 0; c.laplace = true; c.laplace_Ns = 24;
                                              build_or_load_dsmc_tensor(c);
    % 4. frozen modulation amplitude (spectral)
    c = base; c.omega = 0; c.eta_hat_f = 1; c.zeta_hat_f = G.zhf;
              c.laplace = true; c.laplace_Ns = 24;
                                              build_or_load_dsmc_tensor(c);
    % 5. frozen base, algebraic path -- needed only by the path-difference script
    c = base; c.omega = 0;                    build_or_load_dsmc_tensor(c);
end
```

> **Filename caveat.** On the spectral (Laplace) path the internal-sector axes are exact at their Table-1 node counts, so `GeneralCollisionTensor` clamps the requested internal padding to zero and `build_or_load_dsmc_tensor` records the *effective* padding in the name: the three builds above with `laplace = true` are written as `…_p16-16-0_…`, while the three Section 5.5 scripts ask for the historical `…_p16-16-8_…`. The two are the same tensor (measured agreement $\le 2 \times 10^{-14}$ on all four spectral branches), so after regenerating, copy each spectral file to the `-8` name:
>
> ```bash
> cd src/precalc
> for f in *_p16-16-0_*.mat; do cp "$f" "${f/_p16-16-0_/_p16-16-8_}"; done
> ```
>
> `build_or_load_dsmc_tensor` itself handles both names (it has a legacy read fallback); only the three scripts that bypass it need this.

> **`benchmark_polyatomic_transport_fits.m` overwrites its own input.** It reads `results/fig_transport_fits_data.mat` to reuse the shipped $(\omega, \hat\eta_f)$ sampling grids and to print old-vs-new deltas, and then writes the same file back. That file is tracked, so `git checkout results/fig_transport_fits_data.mat` restores the shipped copy. `plot_transport_fits_paper.m` is the render-only pass: it reads that `.mat` and writes no data, so use it whenever only the figure needs redrawing.

### Monatomic tensors

`precompute_collision_operator.m` generates `collisiontensor_k{K}_l{L}_gamma{γ}.mat`. Its shipped defaults are `K_max = 4`, `L_max_list = [2, 4, 6]`, `gamma = 0.0` (Maxwell molecules). Edit those three lines to whatever the consuming script wants — see the [monatomic legacy benchmarks](#3-monatomic-legacy-benchmarks-caches-not-shipped) below for the settings those two need, and the tutorial list for the single-`L_max` cases. A single `(K_max, L_max) = (4, 2)` hard-sphere tensor at the script's `pad = 20` is minutes; the full `L_max` list to 12 is **hours of quadrature and tens of GB on disk**.

---

## Polyatomic paper: script-to-artifact map

Every numbered artifact of the polyatomic paper and the script that produces it. Figure and table numbers refer to that paper. Naming convention: `benchmark_polyatomic_*` computes and saves; `plot_*_paper` renders a saved result into the manuscript's `figures/` directory and computes nothing.

> **Run the `benchmark_polyatomic_*` scripts from the repository root.** They resolve their
> output directory as the relative path `results/`, so launching one from anywhere else
> scatters a fresh `results/` beside the working directory and the renderers will not find it.
> The `plot_*` renderers are immune — they locate `results/` from their own file location — but
> they still need the benchmark to have written there first.

| Paper artifact | Script(s) | Cache / cost |
| --- | --- | --- |
| Fig. 3, Sec. 5.1 — spatial quadrature convergence | `benchmark_polyatomic_spatial_quadrature_convergence.m` → `plot_spatial_quadrature_convergence_paper.m` | data shipped (`spatial_quadrature_convergence_data.mat`); regeneration ~1 h |
| Sec. 5.1 — internal-axis quadrature convergence | `benchmark_polyatomic_quadrature_error.m` | none, ~5 min |
| Fig. 4, Sec. 5.2 — WCU monatomic limit | `benchmark_polyatomic_wcu_limit.m` → `plot_wcu_limit_paper.m` | self-caching, ~30 min; writes `results/wcu_limit_results.mat` |
| Sec. 5.2 — absolute shear / energy-exchange rates against a measured $\nu_0$ | `benchmark_polyatomic_shear_rate.m` | self-caching, ~4 min |
| Fig. 5, Sec. 5.3 — invariant drift over the relaxation transient | `benchmark_polyatomic_temperature_relaxation.m` → `plot_polyrelax_conservation_paper.m` | self-caching, ~20 min; writes `results/polyrelax_results.mat` |
| Sec. 5.3 — static conservation enforcement (the twelve parameterized configurations: $\delta \in \{2,3,5\}$ × four checks) | `tests/TestPolyatomicConservation.m` (via `run_tests`) | none |
| Figs. 6–7, Sec. 5.4 — two-temperature (Landau-Teller) relaxation sweeps | `benchmark_polyatomic_temperature_relaxation.m` → `plot_polyrelax_sweeps_paper.m` | same run as Fig. 5 |
| Table 3, Sec. 5.5.6 — monatomic Chapman-Enskog viscosity factor $f_\mu$ | `benchmark_polyatomic_chapman_enskog.m` (Phase 1b; Phase 1a gives the Maxwell $f_\mu \equiv 1$) | self-caching, ~10 min |
| Sec. 5.5.2 — transport coefficients at the published parameters | `benchmark_polyatomic_transport_fits.m` (the `table4` operating point) | **needs prebuilt caches** |
| Table 4 + Fig. 8, Sec. 5.5.3 — polyatomic transport coefficients and fits | `benchmark_polyatomic_transport_fits.m` (data) → `plot_transport_fits_paper.m` (figure) | **needs prebuilt caches**; overwrites its own input, see the warning above |
| Sec. 5.5.4 — Chapman-Enskog order sensitivity, N₂ | `benchmark_polyatomic_chapman_enskog.m` (Phase 2) | self-caching |
| Sec. 5.5.4 — resolved vs first-order $\mu_b/\mu$ and its $I_{\max}$ convergence, all three gases (stdout + CSV) | `benchmark_polyatomic_internal_truncation.m` | self-caching, hours — the $I_{\max} = 3, 4$ rungs are the expensive ones |
| Sec. 5.5 — closure-attribution cross-check behind Table 4 (stdout only) | `benchmark_polyatomic_closure_attribution.m` | **needs prebuilt caches** |
| Sec. 5.5.5 — spectral-vs-algebraic quadrature-path difference (stdout only) | `benchmark_polyatomic_quadrature_path_difference.m` | **needs prebuilt caches** |
| Table 5 + Figs. 9–10, Sec. 5.6 — storage, geometry, contraction timings | `benchmark_polyatomic_performance.m` | none, ~10 min |

### Where artifacts are written

Every benchmark and plotting script writes to `<repo>/results`, resolved through
`paper_output_dir.m` and created on first use. Nothing is written outside the
repository, and nothing needs configuring on a fresh clone.

`results/` is untracked with one exception: `results/fig_transport_fits_data.mat`
(7.4 MB) ships with the repository. It is the *input* of
`plot_transport_fits_paper.m`, not only its output, and regenerating it from
scratch requires the cached operators — so Figure 8 renders immediately on a
clean checkout. `benchmark_polyatomic_transport_fits.m` overwrites it in place;
`git checkout results/fig_transport_fits_data.mat` restores the shipped copy.

**Known gaps.** Two statements in Section 5.5 have no script in this repository that reproduces them directly, and are reported here rather than left for the reader to discover:

* **Sec. 5.5.4, resolved re-fits.** `benchmark_polyatomic_internal_truncation.m` supplies the resolved and first-order $\mu_b/\mu$ for all three gases, but the resolved re-fits $(\omega, \hat\eta_f)$ quoted alongside them are not produced by any shipped script.
* **Sec. 5.5.5, truncation robustness.** The sweep over $(K_{\max}, I_{\max}) \in \{(1,2), (2,2), (3,2), (2,1), (2,3)\}$ is not scripted, though the caches it used are reproducible with `build_or_load_dsmc_tensor.m` at those truncations.

Supporting drivers (they feed the Section 5.5 discussion but are not themselves artifact generators): `benchmark_dsmc_transport.m` and `benchmark_dsmc_extended.m`, the transport benchmarks described below; `plot_polyce_convergence.m`, which re-renders the Chapman-Enskog convergence figure from `results/polyce_results.mat`; `build_or_load_dsmc_tensor.m`, the tensor cache layer they share.

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
(`kernel_model = 1` non-frozen, `2` frozen) — or by `compute_rtensor_polyatomic_aux_mex` on
the default spectral Laplace path (`laplace_extended = true`) — and blended in
`GeneralCollisionTensor` as `omega*C_vhs*R_nf + (1-omega)*C_vhs_frozen*R_fr`. Recompile
**both** MEX files after pulling these changes (same `mex -R2018a … -fopenmp` recipe as
above); the two paths must agree to machine precision for base kernels.

### Implementation notes

* **Frozen channel.** Elastic (`I'=I`, `I*'=I*`, `|u'|=|u|`), integrated directly in
  internal-energy space; basis orthonormality carries the Kronecker structure. There is *no*
  `(R,r)`-partition measure in this channel (a common pitfall — adding the Borgnakke–Larsen
  `H_delta` weight double-counts a `nu`-dependent factor and corrupts the internal-heat-flux
  rate). The deltas do, however, leave the eq.-(43) weight `e_tr^(zeta/2)` behind, evaluated
  at the collapsed point `e_tr = R' = R = (1/2)|u|^2 / E`, `E = (1/2)|u|^2 + I + I*` — so
  eq. (43) carries the *same* `R^(zeta/2)` weight on both channels (folded into the Jacobi
  `R`-weight on the non-frozen one). It is a per-`(I,I*,|u|)` scalar multiplying gain and
  loss alike, so conservation is untouched, and it is identically 1 at `zeta = 0` and at
  `I_max = 0`. Set `GeneralCollisionTensor.frozen_no_etr = true` to drop it and recover the
  eq.-(53) DSMC-comparison form of the paper's Sect. 8 (diagnostic only — omitting it
  over-weights the frozen channel by `1/<R^(zeta/2)>`).
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
The internal report that preceded the manuscript (its LaTeX sources and its `make_figures.m`
driver) has been removed as superseded. Its three validation figures — frozen-Pr accuracy vs.
Monte-Carlo, auxiliary-integral convergence, transport reachability — have no shipped
regeneration script. The data behind the middle one survives as `results/saxis_data.mat`
(fields `Ns_list`, `errF`, `errN`: the auxiliary-Laplace $s$-integral error vs. the
Gauss-Jacobi node count, frozen and non-frozen channels). That file is the **only** record of
that study in this repository; neither `benchmark_polyatomic_quadrature_error.m` (internal
$I,J,r,R$ axes) nor `benchmark_polyatomic_spatial_quadrature_convergence.m` (spatial axes)
measures the $s$-integral.

---

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