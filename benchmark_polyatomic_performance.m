% benchmark_polyatomic_performance.m
% =========================================================================
% Computational performance of the factorized POLYATOMIC collision operator:
% storage compression and contraction timings along BOTH truncation axes,
% the angular truncation L_max and the internal truncation I_max.
%
% PAPER ARTIFACTS PRODUCED
%   Section 5.6 (Computational Performance) in its entirety. The published
%   numbers stand as printed; this script is the missing means of
%   regenerating them, not a revision of them. Of what it produces, the
%   EXACT COUNTS (storage, geometry) are shipped as data files in the paper
%   repo because they are machine-independent and reproduce Table 5 to the
%   printed digit; the TIMINGS are written only into this repo, because they
%   are machine-dependent measurements (see TIMING PROTOCOL).
%     * Figure 9  (fig_perf_storage)      -- absolute footprint and the
%                 compression M_dense/M_fact vs L_max, for I_max = 0..4.
%     * Table 5   (tab:perf_storage)      -- DOF, N_G, N_T, dense [GB],
%                 factorized [MB], compression, at I_max = 2 and 4.
%     * Figure 10 (fig_perf_contraction)  -- speedup of the three factorized
%                 orderings over the dense Cartesian baseline at I_max = 2.
%     * "Contraction Timings" paragraph   -- the dense/angular, dense/naive
%                 and dense/radial speedups at (L_max, I_max) = (10, 2), and
%                 the ordering margin t_radial/t_angular over the whole grid
%                 (quoted at L_max = 6 and 8 across I_max = 0..4, and at
%                 (12, 2) where the dense baseline no longer fits).
%     * the slice counts N_S (quoted as 35 at L_max = 2 rising to 9100 at
%                 L_max = 12, i.e. 0.42 N_G falling to 0.06 N_G).
%     * "Feasibility Boundaries" paragraph -- the dense footprints at
%                 (12, 2), (8, 4) and (10, 4), read off the storage counts.
%
%   WHY THIS SCRIPT EXISTS. The two legacy performance benchmarks,
%   benchmark_collision_contraction_scaling.m and
%   benchmark_memory_compression.m, are MONATOMIC: they run at K_max = 4,
%   I_max is absent, and they read the gamma-named precomputed caches
%   collisiontensor_k%d_l%d_gamma%.2f.mat. They cannot produce any number in
%   Section 5.6, which is at K_max = 2 and sweeps I_max. They also use a
%   different storage model (see the STORAGE MODEL note below). This script
%   is the polyatomic replacement and is the sole source of the Section 5.6
%   artifacts.
%
% TIMING PROTOCOL (this is the protocol the paper states, sections/section5.tex:179)
%   * Single core. All four contraction kernels (dense_tensor_kernel_mex,
%     naive_collision_kernel_mex, radial_first_collision_kernel_mex,
%     angular_first_collision_kernel_mex) are serial nested loops with no
%     OpenMP and no branch on a coefficient value, so the timings are fixed
%     by structure alone: N_G, N_T, N_Q and N_KI = (K_max+1)(I_max+1).
%   * The timed quantity is ONE nonlinear operator evaluation
%     Q = C : (f (x) f), INCLUDING allocation of the output buffer -- the
%     zeros(...) call sits inside the timed anonymous function.
%   * Reported as the MEDIAN over repeated calls: timeit() already returns a
%     robust median over an internally calibrated number of calls, and we
%     wrap it in n_rep outer repeats with a FRESH random draw of R and f each
%     time, reporting the median of those and carrying min/max as the spread.
%   * SYNTHETIC radial--internal blocks with TRUE Gaunt geometry. The
%     geometry (gaunt_labels, gaunt_vals, ic_map) comes from the generator --
%     GeneralCollisionTensor's constructor calls generate_geometry() and
%     nothing else -- while R_tensor is filled with randn of the correct
%     shape. The nine-dimensional quadrature is NOT run. This is legitimate
%     precisely because the kernels never branch on a value, and it makes
%     truncations reachable that a converged build could not afford. NO
%     PHYSICAL CLAIM is made from these runs.
%   * Only RELATIVE performance is reported. All four strategies run in the
%     same session and environment, so the ratios are the reportable
%     quantity. The ABSOLUTE times are machine-dependent and will differ
%     between machines, between MATLAB releases, and between runs on the
%     same machine: the dense baseline in particular is bandwidth-bound on a
%     multi-gigabyte array, so a few per cent of run-to-run drift in it moves
%     every dense/X speedup by the same few per cent. Do not expect this
%     script to reproduce the paper's printed speedups digit for digit on
%     other hardware; expect the ordering, the trends and the complexity
%     slopes.
%   * Reference environment: single core of a 12-core Apple M2 Pro (32 GB,
%     macOS 14.4.1) under MATLAB R2023b -- the environment of the published
%     Section 5.6 numbers, which stand as printed.
%
% STORAGE MODEL -- follows the PAPER, not the legacy benchmark
%   The paper's eq. (30) (\label{eq:storage_model}) is a pure count in bytes,
%
%       M_dense = 8 * DOF^3,
%       M_fact  = 8 * ( N_T * N_KI^3  +  5 * N_G ),      DOF = N_Q * N_KI,
%
%   with N_Q = (L_max+1)^2 and N_KI = (K_max+1)(I_max+1). The factorized
%   count carries the dense channel blocks of R together with the
%   coordinate-format routing list of G at FIVE DOUBLES = 40 BYTES per Gaunt
%   transition (three index labels, the Gaunt weight, the channel tag).
%
%   This deliberately differs from benchmark_memory_compression.m:72, which
%   charges 20 bytes per nonzero (3 x uint32 + 1 x double, and no channel
%   tag). We follow the PAPER's 40-byte model here so that Table 5, Figure 9
%   and the text agree with the equation printed in the paper. The two models
%   differ by a factor 2 on the routing term only; at I_max = 0 that term is
%   98% of the factorized total, so the choice is not cosmetic.
%
%   The script also breaks out the two terms separately (R_bytes and
%   gaunt_bytes), which is what substantiates the text's statement that at
%   L_max = 12 the routing list is 98% of the factorized total at I_max = 0
%   but only 27% at I_max = 4.
%
% SLICE COUNT
%   N_S is the number of distinct (channel t, q1) pairs in the sorted Gaunt
%   list -- the outer-loop count of the angular-first ordering, whose cost is
%   O(N_G N_KI^2) + O(N_S N_KI^3) against the radial-first O(N_G N_KI^3).
%   Replacing N_G by N_S in the cubic term is what predicts the ordering
%   margin; the measured margin exceeds it by the cache-locality factor.
%
% BUILD / RESOLUTION
%   K_max = 2 throughout (the paper's setting). L_max in {2,4,6,8,10,12},
%   I_max in {0,...,4}. Geometry is built once per L_max and is independent
%   of I_max. Nothing is read from or written to src/precalc -- everything is
%   in memory, so the numbers are reproducible from a clean checkout.
%
%   The dense Cartesian baseline is assembled only on the I_max = dense_I
%   column (the paper's Figure 10) and only where its footprint fits under
%   dense_max_GB. At (L_max, I_max) = (10, 2) the dense tensor is 9.6 GB, so
%   this script needs a machine with headroom; at (12, 2) it is 26.2 GB and
%   is deliberately not run, which is the paper's own feasibility boundary.
%   The dense array is passed to the MEX kernel as-is (the kernel takes a
%   bare pointer and its own N), so no C(:) copy is made -- that copy would
%   double the peak footprint to 19 GB at (10, 2).
%
% CORRECTNESS CROSS-CHECK
%   At every grid point the three factorized orderings are asserted to agree
%   to 1e-10 relative, and where the dense baseline is built it is asserted
%   to agree with them as well. A timing comparison between kernels that do
%   not compute the same thing would be meaningless.
%
% WRITES
%   Into the PAPER repo figures/ (set write_csv = false to skip) -- EXACT
%   COUNTS only. These are integer counts and a byte model, not measurements:
%   they are identical on any machine and reproduce Table 5 to the printed
%   digit, so they are legitimate shipped data artifacts.
%     fig_perf_storage_data.csv   Figure 9 + Table 5: the byte counts and the
%                                 compression on the full (L_max, I_max) grid.
%     fig_perf_geometry_data.csv  N_Q, N_G, N_T, N_S, N_S/N_G and the
%                                 geometric ceiling N_Q^3/N_T per L_max.
%
%   Into THIS repo (never the paper repo):
%     perf_contraction_timings.csv  the measured timing table. Timings are
%                                 machine-dependent (see TIMING PROTOCOL), so
%                                 this file is a local measurement log, not a
%                                 paper artifact. The paper's printed
%                                 speedups are the ones measured on the
%                                 reference machine and stand as published;
%                                 a re-run here will land near them, not on
%                                 them. A copy IS shipped alongside the
%                                 paper, as figures/fig_perf_contraction_data.csv,
%                                 carrying a header comment recording the
%                                 machine and stating that the ratios, not the
%                                 absolute times, are the reportable quantity.
%
% RUNTIME
%   ~5 min for the factorized grid at n_rep = 3, plus ~5 min for the dense
%   column (dominated by assembling and streaming the 9.6 GB tensor at
%   L_max = 10). Set run_dense = false for the storage/ordering data alone.
% =========================================================================

proj = fileparts(mfilename('fullpath'));
addpath(fullfile(proj,'src'));
addpath(genpath(fullfile(proj,'src','SHL')));
addpath(fullfile(proj,'src','mex'));

write_csv  = true;
paper_figs = '/Users/ekke/dev/simkinetic/papers/polyatomic-collision-factorization/figures';

% ---- resolution grid (Section 5.6) --------------------------------------
K_max    = 2;
L_list   = [2 4 6 8 10 12];
I_list   = 0:4;

% ---- timing controls ----------------------------------------------------
n_rep        = 3;      % outer repeats, fresh random R and f each time
run_dense    = true;   % assemble the dense Cartesian baseline (Figure 10)
dense_I      = 2;      % internal truncation of the dense column
dense_max_GB = 12;     % skip the dense baseline above this footprint
tol_kernels  = 1e-10;  % cross-check tolerance between contraction orderings

rng(20260829);   % only fixes the synthetic R and f; the geometry is exact

% At the smallest truncations one operator evaluation is a few microseconds,
% i.e. within a small multiple of timeit's own measurement overhead, and
% timeit says so on every call. The warning is expected and is recorded once
% here rather than repeated ~40 times; those rows are indicative only and no
% paper number is taken from them (the paper quotes L_max >= 6).
ws = warning('off','MATLAB:timeit:HighOverhead');
cleanupW = onCleanup(@() warning(ws));
fprintf(['NOTE: timeit''s HighOverhead warning is suppressed. Rows with an\n' ...
         '      evaluation time below ~1e-5 s sit near the timer floor.\n']);

fprintf('\n=========================================================================\n');
fprintf(' Polyatomic collision operator: storage and contraction performance\n');
fprintf(' (paper Section 5.6 -- Figures 9 and 10, Table 5)\n');
fprintf(' K_max = %d, L_max in [%s], I_max in [%s]\n', K_max, ...
    num2str(L_list), num2str(I_list));
fprintf('=========================================================================\n');

n_L = numel(L_list);
geo_rows   = {};
store_rows = {};
time_rows  = {};

t_all = tic;

for li = 1:n_L
    L = L_list(li);

    % ---- geometry: generator-exact, independent of I_max ----------------
    % GeneralCollisionTensor's constructor runs generate_geometry() only; the
    % nine-dimensional quadrature (generate_R_tensor*) is never called.
    Basis  = SpectralBasis(K_max, L, max(I_list), 0.5);
    Kernel = ScatteringKernel('Polyatomic', 0.0);
    T      = GeneralCollisionTensor(Basis, Kernel);

    g_labels = T.gaunt_labels;
    g_vals   = T.gaunt_vals;
    ic_map   = T.ic_map;

    N_Q = (L+1)^2;
    N_G = size(g_labels,1);
    N_T = max(ic_map);
    % N_S: distinct (channel, q1) slices -- the angular-first outer count.
    N_S = size(unique([ic_map, g_labels(:,1)], 'rows'), 1);

    % The two factorized orderings require the transition list blocked by
    % channel t and, within a channel, by q1. Sort once per L_max.
    [~, sidx] = sortrows([ic_map, g_labels(:,1)]);
    ic_s = ic_map(sidx);  lb_s = g_labels(sidx,:);  vs_s = g_vals(sidx);

    geo_rows(end+1,:) = {L, N_Q, N_G, N_T, N_S}; %#ok<SAGROW>

    fprintf('\n--- L_max = %2d :  N_Q = %4d   N_G = %7d   N_T = %4d   N_S = %6d  (N_S/N_G = %.4f)\n', ...
        L, N_Q, N_G, N_T, N_S, N_S/N_G);

    for ii = 1:numel(I_list)
        I    = I_list(ii);
        N_KI = (K_max+1)*(I+1);
        DOF  = N_Q * N_KI;

        % ---- storage: paper eq. (30), a pure byte count -----------------
        R_bytes     = 8 * N_T * N_KI^3;   % dense channel blocks of R
        gaunt_bytes = 8 * 5 * N_G;        % routing list, 5 doubles/transition
        fact_bytes  = R_bytes + gaunt_bytes;
        dense_bytes = 8 * DOF^3;

        store_rows(end+1,:) = {L, I, K_max, N_KI, DOF, N_Q, N_G, N_T, N_S, ...
            dense_bytes, fact_bytes, R_bytes, gaunt_bytes, ...
            dense_bytes/2^30, fact_bytes/2^20, dense_bytes/fact_bytes, ...
            gaunt_bytes/fact_bytes}; %#ok<SAGROW>

        % ---- timings: n_rep repeats, fresh synthetic blocks each time ----
        tn = zeros(1,n_rep); tr = zeros(1,n_rep); ta = zeros(1,n_rep);
        for r = 1:n_rep
            R = randn(N_KI, N_KI, N_KI, N_T);
            f = randn(N_Q, N_KI);

            tn(r) = timeit(@() call_mex_out(@naive_collision_kernel_mex, ...
                        zeros(N_Q,N_KI), f, g_labels, g_vals, ic_map, R, N_Q, N_KI));
            tr(r) = timeit(@() call_mex_out(@radial_first_collision_kernel_mex, ...
                        zeros(N_Q,N_KI), f, lb_s, vs_s, ic_s, R, N_Q, N_KI));
            ta(r) = timeit(@() call_mex_out(@angular_first_collision_kernel_mex, ...
                        zeros(N_Q,N_KI), f, lb_s, vs_s, ic_s, R, N_Q, N_KI));

            if r == 1
                % correctness cross-check on this draw
                Qn = zeros(N_Q,N_KI); naive_collision_kernel_mex(Qn, f, g_labels, g_vals, ic_map, R, N_Q, N_KI);
                Qr = zeros(N_Q,N_KI); radial_first_collision_kernel_mex(Qr, f, lb_s, vs_s, ic_s, R, N_Q, N_KI);
                Qa = zeros(N_Q,N_KI); angular_first_collision_kernel_mex(Qa, f, lb_s, vs_s, ic_s, R, N_Q, N_KI);
                sc = max(abs(Qa(:)));
                e_rn = max(abs(Qr(:)-Qa(:)))/sc;
                e_nn = max(abs(Qn(:)-Qa(:)))/sc;
                assert(e_rn < tol_kernels && e_nn < tol_kernels, ...
                    'Contraction orderings disagree at (L,I)=(%d,%d): %.2e, %.2e', L, I, e_rn, e_nn);
            end
        end
        t_naive = median(tn); t_radial = median(tr); t_angular = median(ta);

        % ---- dense Cartesian baseline (Figure 10 column) ----------------
        t_dense = NaN; e_dn = NaN;
        dense_GB = dense_bytes/2^30;
        if run_dense && I == dense_I && dense_GB <= dense_max_GB
            td = zeros(1,n_rep);
            for r = 1:n_rep
                R = randn(N_KI, N_KI, N_KI, N_T);
                f = randn(N_Q, N_KI);
                f_flat = f(:);

                % Scatter the factorized operator into the dense Cartesian
                % tensor. The (q1,q2,q3) triples are unique in the Gaunt list,
                % so plain assignment (not accumulation) is correct.
                C = zeros(DOF, DOF, DOF);
                off = (0:N_KI-1)*N_Q;
                for z = 1:N_G
                    C(g_labels(z,1)+off, g_labels(z,2)+off, g_labels(z,3)+off) = ...
                        R(:,:,:,ic_map(z)) * g_vals(z);
                end

                % C is passed 3-D: dense_tensor_kernel_mex takes a bare
                % pointer plus DOF, so no C(:) copy is made.
                td(r) = timeit(@() call_mex_out(@dense_tensor_kernel_mex, ...
                            zeros(DOF,1), f_flat, C, DOF));

                if r == 1
                    Qd = zeros(DOF,1); dense_tensor_kernel_mex(Qd, f_flat, C, DOF);
                    Qa = zeros(N_Q,N_KI); angular_first_collision_kernel_mex(Qa, f, lb_s, vs_s, ic_s, R, N_Q, N_KI);
                    e_dn = max(abs(Qd - Qa(:)))/max(abs(Qa(:)));
                    assert(e_dn < tol_kernels, ...
                        'Dense baseline disagrees with the factorized operator at (L,I)=(%d,%d): %.2e', L, I, e_dn);
                    clear Qd Qa;
                end
                clear C;
            end
            t_dense = median(td);
        end

        time_rows(end+1,:) = {L, I, K_max, N_KI, DOF, N_Q, N_G, N_T, N_S, n_rep, ...
            t_dense, t_naive, t_radial, t_angular, ...
            min(tn), max(tn), min(tr), max(tr), min(ta), max(ta), ...
            t_radial/t_angular, t_dense/t_angular, t_dense/t_naive, t_dense/t_radial, ...
            dense_GB, fact_bytes/2^20}; %#ok<SAGROW>

        fprintf(['    I_max=%d  N_KI=%2d  DOF=%5d | dense %9.5f  naive %9.6f  ' ...
                 'radial %9.6f  angular %9.6f  s | t_r/t_a %6.2f | d/a %6.2f  d/n %5.2f  d/r %5.2f | ' ...
                 'dense %8.3f GB  fact %7.3f MB  compression %7.0fx\n'], ...
            I, N_KI, DOF, t_dense, t_naive, t_radial, t_angular, ...
            t_radial/t_angular, t_dense/t_angular, t_dense/t_naive, t_dense/t_radial, ...
            dense_GB, fact_bytes/2^20, dense_bytes/fact_bytes);

        clear R f;
    end
    clear T Basis Kernel g_labels g_vals ic_map lb_s vs_s ic_s;
end

fprintf('\nTotal wall time: %.1f s\n', toc(t_all));

%% ========================================================================
%  TABLES
%  ========================================================================
Tstore = cell2table(store_rows, 'VariableNames', ...
    {'L_max','I_max','K_max','N_KI','DOF','N_Q','N_G','N_T','N_S', ...
     'dense_bytes','fact_bytes','R_bytes','gaunt_bytes', ...
     'dense_GB','fact_MB','compression','gaunt_fraction_of_fact'});

Ttime = cell2table(time_rows, 'VariableNames', ...
    {'L_max','I_max','K_max','N_KI','DOF','N_Q','N_G','N_T','N_S','n_rep', ...
     't_dense_s','t_naive_s','t_radial_s','t_angular_s', ...
     't_naive_min_s','t_naive_max_s','t_radial_min_s','t_radial_max_s', ...
     't_angular_min_s','t_angular_max_s', ...
     'ratio_radial_over_angular','speedup_dense_over_angular', ...
     'speedup_dense_over_naive','speedup_dense_over_radial', ...
     'dense_GB','fact_MB'});

Tgeo = cell2table(geo_rows, 'VariableNames', {'L_max','N_Q','N_G','N_T','N_S'});
Tgeo.N_S_over_N_G = Tgeo.N_S ./ Tgeo.N_G;
Tgeo.geometric_ceiling = (Tgeo.N_Q.^3) ./ Tgeo.N_T;   % N_Q^3/N_T, the compression ceiling

% ---- Table 5 as printed (I_max = 2 and 4) --------------------------------
fprintf('\n== Table 5: storage of the polyatomic collision operator (K_max = %d) ==\n', K_max);
for I = [2 4]
    fprintf('\n  Internal truncation I_max = %d\n', I);
    fprintf('  %-6s %-8s %-9s %-6s %-12s %-14s %-12s\n', ...
        'L_max','DOF','N_G','N_T','Dense [GB]','Factorized [MB]','Compression');
    s = Tstore(Tstore.I_max == I, :);
    for r = 1:height(s)
        fprintf('  %-6d %-8d %-9d %-6d %-12.4f %-14.3f %-12.0fx\n', ...
            s.L_max(r), s.DOF(r), s.N_G(r), s.N_T(r), s.dense_GB(r), s.fact_MB(r), s.compression(r));
    end
end

fprintf('\n== Geometry and slice counts ==\n');
fprintf('  %-6s %-6s %-9s %-6s %-8s %-10s %-12s\n','L_max','N_Q','N_G','N_T','N_S','N_S/N_G','N_Q^3/N_T');
for r = 1:height(Tgeo)
    fprintf('  %-6d %-6d %-9d %-6d %-8d %-10.4f %-12.0f\n', ...
        Tgeo.L_max(r), Tgeo.N_Q(r), Tgeo.N_G(r), Tgeo.N_T(r), Tgeo.N_S(r), ...
        Tgeo.N_S_over_N_G(r), Tgeo.geometric_ceiling(r));
end

% ---- the headline numbers of the "Contraction Timings" paragraph ---------
fprintf('\n== Headline contraction numbers ==\n');
hl = Ttime(Ttime.L_max == 10 & Ttime.I_max == 2, :);
if height(hl) == 1 && ~isnan(hl.t_dense_s)
    fprintf('  (L_max,I_max) = (10,2):  dense/angular %.1fx   dense/naive %.1fx   dense/radial %.1fx\n', ...
        hl.speedup_dense_over_angular, hl.speedup_dense_over_naive, hl.speedup_dense_over_radial);
end
h12 = Ttime(Ttime.L_max == 12 & Ttime.I_max == 2, :);
if height(h12) == 1
    fprintf('  (L_max,I_max) = (12,2):  t_radial/t_angular %.2f\n', h12.ratio_radial_over_angular);
end
for L = [6 8]
    s = Ttime(Ttime.L_max == L, :);
    fprintf('  L_max = %2d: t_radial/t_angular over I_max = 0..4: %s\n', ...
        L, strjoin(compose('%.2f', s.ratio_radial_over_angular'), '  '));
end
s12 = Tstore(Tstore.L_max == 12, :);
fprintf('  L_max = 12: routing list as a fraction of the factorized total, I_max = 0..4: %s\n', ...
    strjoin(compose('%.0f%%', 100*s12.gaunt_fraction_of_fact'), '  '));

%% ========================================================================
%  WRITE
%  ========================================================================
% The timing table is a LOCAL measurement log and stays in this repo. Timings
% are machine-dependent, and the paper's Section 5.6 speedups are the values
% measured on the reference machine; shipping a re-run's timings alongside
% them would put two slightly different sets of numbers in circulation for
% the same quantity. Only the exact counts go to the paper.
p0 = fullfile(proj, 'perf_contraction_timings.csv');
writetable(Ttime, p0);
fprintf('\nWrote %s   (local measurement log -- NOT a paper artifact)\n', p0);

if write_csv
    if ~isfolder(paper_figs)
        warning('Paper figures directory not found: %s -- skipping CSV write.', paper_figs);
    else
        p1 = fullfile(paper_figs,'fig_perf_storage_data.csv');
        p3 = fullfile(paper_figs,'fig_perf_geometry_data.csv');
        writetable(Tstore, p1);
        writetable(Tgeo,   p3);
        fprintf('Wrote %s\n', p1);
        fprintf('Wrote %s\n', p3);
    end
end

fprintf('\nDone.\n');

%% ========================================================================
%  HELPERS
%  ========================================================================
function out = call_mex_out(f, out, varargin)
    % The kernels write in place into a caller-allocated buffer. Allocating
    % that buffer inside the timed function is deliberate: the paper's timed
    % quantity includes the output allocation.
    f(out, varargin{:});
end
