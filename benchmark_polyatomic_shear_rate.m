% benchmark_polyatomic_shear_rate.m
% =========================================================================
% Absolute (non-circular) verification of the shear and energy-exchange
% relaxation rates against an INDEPENDENTLY MEASURED equilibrium collision
% frequency nu_0.
%
% PAPER CLAIM SUBSTANTIATED
%   Section 5.2 (WCU limit):
%     "Measured against the equilibrium collision frequency nu_0 of the loss
%      operator of the same build, the shear eigenvalue is lambda_shear =
%      -nu_0/2 at every d_i, matching the analytic value (eq. lambda_shear) of
%      Appendix B to below 1e-13. ... Over the same sweep the energy-exchange
%      eigenvalue satisfies lambda_exch = -nu_0 (3 + d_i)/(3 + 2 d_i)."
%   Appendix B, eqs. (lambda_shear) and (lambda_bulk).
%
% WHY THIS SCRIPT EXISTS (the circularity it removes)
%   benchmark_polyatomic_wcu_limit.m and
%   benchmark_polyatomic_temperature_relaxation.m both DEFINE the collision
%   frequency from the shear eigenvalue itself (nu_0 := 2|lambda_shear|).
%   That is a perfectly good normalizer -- it is scale-free and independent of
%   the exchange rate it is used to normalize -- but it makes
%   "lambda_shear = -nu_0/2" an identity, and therefore proves nothing about
%   lambda_shear in absolute terms.
%
%   Here nu_0 is instead measured from the LOSS OPERATOR, built by zeroing the
%   gain kernels (GeneralCollisionTensor.loss_only, a diagnostic build flag
%   that is inert by default and does not touch the production path). At
%   zeta = 0 the loss term is exactly nu_0 * f, so the linearized loss-only
%   operator is -nu_0 * Identity on every non-density mode: nu_0 is read off
%   its diagonal, with no reference to any gain-side eigenvalue. The shear
%   eigenvalue is then measured from the FULL operator of the same build and
%   compared to -nu_0/2 in absolute terms.
%
%   The measured nu_0 is additionally checked against its closed form
%       nu_0 = 4*pi*n,   n = sqrt(pi^(3/2) * Gamma(d_i/2)),
%   where n is the background number density implied by the code's
%   linearisation point c_1 = 1 (the operator is linearized about the
%   equilibrium whose density-mode coefficient is unity). Two independent
%   ingredients:
%     * 4*pi -- at zeta = 0 the DSMC non-frozen kernel is B = K_delta, and
%       K_delta is fixed (ScatteringKernel, eq. 40) so that K_delta times the
%       Borgnakke-Larsen partition-measure mass equals 1; the remaining solid
%       -angle integral contributes 4*pi.
%     * n = 1/Psi_1 -- the reciprocal of the constant density basis function
%       of the shipped SpectralBasis. The script verifies the closed form
%       against SpectralBasis.evaluate directly, so the formula cannot drift
%       away from the normalization the operator was actually built with.
%
% BUILD
%   Non-frozen Maxwell kernel (zeta = 0, omega = 1) at K=L=I=2, spatial padding
%   (16,16), internal axes at their exactness-bound Table-1 node counts (the
%   spectral / auxiliary-Laplace path clamps internal padding to zero), i.e.
%   the Section 5.2 production build. d_i is swept geometrically over
%   {2, 1, 1/2, ..., 1/32}. Everything is built in memory -- no cache is read
%   or written, so the numbers below are reproducible from a clean checkout.
%
%   The loss-only builds run with conserve_invariants = false: the loss
%   operator alone is not conservative, and the invariant projection would
%   rotate exactly the rows being measured. The full operators run with the
%   production conservation enforcement on.
%
%   A second loss-only build per d_i on the ALGEBRAIC (laplace_extended =
%   false) path cross-checks that nu_0 is a property of the operator and not
%   of the quadrature path. For a pure base kernel the two paths agree to
%   machine precision.
%
%   Runtime ~4 min for the whole sweep (21 builds, measured on an M-series
%   Mac). The spectral builds are ~1.6 s each; essentially all the time is the
%   algebraic cross-check (~27 s each), which is the slow pointwise path.
%   Note the MEX contractions use a nondeterministic parallel reduction, so
%   repeated builds of the same configuration differ at ~1e-13 relative; that
%   is the floor for every number in the tables below.
%
% WRITES
%   <paper>/figures/fig_wcu_limit_shear_rate_data.csv
%   -- one row per d_i, the data artifact behind the Section 5.2 shear claim.
%   Set write_csv = false below to run without writing.
%
% PASS CRITERIA (all asserted, and summarized at the end)
%   |lambda_shear/nu_0 + 1/2|                        < 1e-13   at every d_i
%   |lambda_exch/nu_0 + (3+d_i)/(3+2 d_i)|           < 1e-13   at every d_i
%   |nu_0/(4*pi*n) - 1|                              < 1e-12   at every d_i
%   e_(k=0,i=0,l=2,m=0) is an eigenvector of the full J (residual < 1e-12),
%     i.e. the shear mode does not couple to the internal sector -- the
%     assumption the Appendix B derivation rests on.
%   the full operator has exactly 5 null modes (the collision invariants).
% =========================================================================

proj = fileparts(mfilename('fullpath'));
addpath(fullfile(proj,'src'));
addpath(genpath(fullfile(proj,'src','SHL')));
addpath(fullfile(proj,'src','mex'));

write_csv  = true;
paper_figs = '/Users/ekke/dev/simkinetic/papers/polyatomic-collision-factorization/figures';
csv_name   = 'fig_wcu_limit_shear_rate_data.csv';

% ---- build configuration (Section 5.2) ----------------------------------
di_list = [2, 1, 1/2, 1/4, 1/8, 1/16, 1/32];
K_max = 2; L_max = 2; I_max = 2;
rad_pad = 16; tan_pad = 16; int_pad = 8;   % internal request is clamped to 0

tol_rate = 1e-13;   % lambda/nu_0 vs analytic
tol_nu0  = 1e-12;   % nu_0 vs 4*pi*n
tol_vec  = 1e-12;   % shear eigenvector residual

n_d = numel(di_list);
res = struct('delta',cell(1,n_d));

idx = @(B,k,i,l,m) (k*B.N_I + i)*B.N_Q + (l^2 + l + m) + 1;

fprintf('\n=========================================================================\n');
fprintf(' Absolute shear / exchange rates vs an independently measured nu_0\n');
fprintf(' (paper Section 5.2; loss-only operator, gain kernels zeroed)\n');
fprintf('=========================================================================\n');

t_all = tic;
for d = 1:n_d
    delta = di_list(d);
    fprintf('\n--- d_i = %g -----------------------------------------------------\n', delta);

    Kernel = ScatteringKernel('DSMC', struct('zeta',0,'delta',delta,'omega',1));
    Basis  = SpectralBasis(K_max, L_max, I_max, Kernel.nu);

    % ---- analytic background number density, and its code-level check ----
    n_analytic = sqrt(pi^1.5 * gamma(delta/2));
    % Psi_1 is the constant (k=0,i=0,l=0,m=0) density mode of the shipped basis;
    % n = 1/Psi_1. Evaluated off-origin to avoid the r=0 branch of evaluate().
    Psi = Basis.evaluate([0 0 1e-8], 0);
    n_basis = 1 / Psi(idx(Basis,0,0,0,0));

    % ---- loss-only operator, spectral (production) path ------------------
    tb = tic;
    Tl = GeneralCollisionTensor(Basis, Kernel);
    Tl.laplace_extended    = true;    % Section 5.2 production path
    Tl.conserve_invariants = false;   % the loss operator is not conservative
    Tl.loss_only           = true;    % <-- gain kernels zeroed
    Tl.generate_R_tensor_sumfac(rad_pad, tan_pad, int_pad);
    Cl = Tl.assemble_full_tensor();
    t_loss_spec = toc(tb);
    Jl = squeeze(Cl(:,:,1)) + squeeze(Cl(:,1,:));

    dg   = diag(Jl);
    offd = Jl - diag(dg);
    % nu_0 off the loss diagonal of a NON-density mode (the density mode
    % carries the doubled linearisation and is excluded).
    nu0_spec   = -dg(idx(Basis,0,0,2,0));
    loss_flat  = max(abs(-dg(2:end) - nu0_spec));   % spread over non-density modes
    loss_offd  = max(abs(offd(:)));

    % ---- loss-only operator, algebraic path (path-independence check) ----
    tb = tic;
    Ta = GeneralCollisionTensor(Basis, Kernel);
    Ta.laplace_extended    = false;   % pointwise / algebraic build
    Ta.conserve_invariants = false;
    Ta.loss_only           = true;
    Ta.generate_R_tensor_sumfac(rad_pad, tan_pad, int_pad);
    Ca = Ta.assemble_full_tensor();
    t_loss_alg = toc(tb);
    Ja = squeeze(Ca(:,:,1)) + squeeze(Ca(:,1,:));
    nu0_alg = -Ja(idx(Basis,0,0,2,0), idx(Basis,0,0,2,0));

    % ---- full operator, spectral (production) path -----------------------
    tb = tic;
    Tf = GeneralCollisionTensor(Basis, Kernel);
    Tf.laplace_extended    = true;
    Tf.conserve_invariants = true;    % production conservation enforcement
    Tf.loss_only           = false;   % <-- default production build
    Tf.generate_R_tensor_sumfac(rad_pad, tan_pad, int_pad);
    Cf = Tf.assemble_full_tensor();
    t_full = toc(tb);
    J = squeeze(Cf(:,:,1)) + squeeze(Cf(:,1,:));

    % shear: least-negative eigenvalue of the i=0, l=2, m=0 sub-block
    cols_shear = arrayfun(@(k) idx(Basis,k,0,2,0), 0:K_max);
    ev2        = sort(real(eig(J(cols_shear, cols_shear))), 'descend');
    lam_shear  = ev2(1);

    % is e_(0,0,2,0) an exact eigenvector of the FULL operator? (Appendix B
    % assumes the shear mode carries no internal energy and does not couple.)
    e = zeros(size(J,1),1); e(idx(Basis,0,0,2,0)) = 1;
    Je = J*e;  lam_ray = Je(idx(Basis,0,0,2,0));
    vec_resid = norm(Je - lam_ray*e) / abs(lam_ray);

    % energy exchange: nonzero eigenvalue of the 2x2 block
    % {(k,i,l)=(1,0,0), (0,1,0)}; the other eigenvalue is the energy invariant.
    blk  = [idx(Basis,1,0,0,0), idx(Basis,0,1,0,0)];
    evx  = sort(real(eig(J(blk,blk))));
    lam_exch      = evx(1);
    lam_exch_null = evx(2);

    % conservation: 5 null modes, invariant rows identically zero
    n_null = sum(abs(eig(J)) < 1e-9);
    a = sqrt(3/2); b = sqrt(Kernel.nu + 1); nn = hypot(a,b);
    inv_rows = max([norm(J(idx(Basis,0,0,0, 0),:)), ...
                    norm(J(idx(Basis,0,0,1,-1),:)), ...
                    norm(J(idx(Basis,0,0,1, 0),:)), ...
                    norm(J(idx(Basis,0,0,1, 1),:)), ...
                    norm((a*J(idx(Basis,1,0,0,0),:) + b*J(idx(Basis,0,1,0,0),:))/nn)]);

    % ---- ratios vs analytic ---------------------------------------------
    r_shear     = lam_shear / nu0_spec;
    r_exch      = lam_exch  / nu0_spec;
    exch_target = -(3 + delta) / (3 + 2*delta);

    res(d).delta        = delta;
    res(d).nu_alpha     = Kernel.nu;
    res(d).n_analytic   = n_analytic;
    res(d).n_basis      = n_basis;
    res(d).nu0_spec     = nu0_spec;
    res(d).nu0_alg      = nu0_alg;
    res(d).nu0_ratio    = nu0_spec / (4*pi*n_analytic);
    res(d).loss_flat    = loss_flat;
    res(d).loss_offd    = loss_offd;
    res(d).lam_shear    = lam_shear;
    res(d).r_shear      = r_shear;
    res(d).err_shear    = abs(r_shear + 0.5);
    res(d).lam_exch     = lam_exch;
    res(d).r_exch       = r_exch;
    res(d).exch_target  = exch_target;
    res(d).err_exch     = abs(r_exch - exch_target);
    res(d).vec_resid    = vec_resid;
    res(d).exch_null    = abs(lam_exch_null);
    res(d).n_null       = n_null;
    res(d).inv_rows     = inv_rows;
    res(d).t_loss_spec  = t_loss_spec;
    res(d).t_loss_alg   = t_loss_alg;
    res(d).t_full       = t_full;

    fprintf('  nu_0 (loss-only) = %.15f   4*pi*n = %.15f   ratio-1 = %.2e\n', ...
        nu0_spec, 4*pi*n_analytic, res(d).nu0_ratio - 1);
    fprintf('  lambda_shear/nu_0 = %.18f   (err vs -1/2 = %.2e)\n', r_shear, res(d).err_shear);
    fprintf('  lambda_exch /nu_0 = %.18f   (err vs %.12f = %.2e)\n', ...
        r_exch, exch_target, res(d).err_exch);
    fprintf('  builds: loss/spectral %.1f s, loss/algebraic %.1f s, full %.1f s\n', ...
        t_loss_spec, t_loss_alg, t_full);
end
t_total = toc(t_all);

% =========================================================================
%  Report
% =========================================================================
fprintf('\n\n============ nu_0 MEASURED FROM THE LOSS-ONLY OPERATOR ============\n');
fprintf('%8s | %20s | %20s | %11s | %14s | %14s | %10s\n', ...
    'd_i','nu_0 (spectral)','nu_0 (algebraic)','|spec-alg|','n = 1/Psi_1','4*pi*n','nu_0/4pi n');
for d = 1:n_d
    fprintf('%8.5g | %20.14f | %20.14f | %11.2e | %14.10f | %14.10f | %10.2e\n', ...
        res(d).delta, res(d).nu0_spec, res(d).nu0_alg, ...
        abs(res(d).nu0_spec - res(d).nu0_alg)/abs(res(d).nu0_spec), ...
        res(d).n_basis, 4*pi*res(d).n_analytic, res(d).nu0_ratio - 1);
end
fprintf(['\n  (the loss operator is -nu_0*I on the non-density modes: ' ...
         'max diagonal spread %.2e, max off-diagonal %.2e)\n'], ...
        max([res.loss_flat]), max([res.loss_offd]));
fprintf('  (n closed form sqrt(pi^{3/2} Gamma(d_i/2)) vs 1/Psi_1 of the shipped basis:');
fprintf(' max rel diff %.2e)\n', max(abs([res.n_analytic]-[res.n_basis])./[res.n_analytic]));

fprintf('\n============ ABSOLUTE RATES vs APPENDIX B ============\n');
fprintf('%8s | %22s | %22s | %10s | %22s | %22s | %10s\n', ...
    'd_i','lambda_shear','lambda_shear/nu_0','err(-1/2)','lambda_exch','lambda_exch/nu_0','err');
for d = 1:n_d
    fprintf('%8.5g | %22.15f | %22.18f | %10.2e | %22.15f | %22.18f | %10.2e\n', ...
        res(d).delta, res(d).lam_shear, res(d).r_shear, res(d).err_shear, ...
        res(d).lam_exch, res(d).r_exch, res(d).err_exch);
end

fprintf('\n============ STRUCTURAL CHECKS ============\n');
fprintf('%8s | %14s | %14s | %8s | %14s | %14s\n', ...
    'd_i','shear eigvec','exch null ev','#null','max|inv row|','-(3+d)/(3+2d)');
for d = 1:n_d
    fprintf('%8.5g | %14.2e | %14.2e | %8d | %14.2e | %14.10f\n', ...
        res(d).delta, res(d).vec_resid, res(d).exch_null, res(d).n_null, ...
        res(d).inv_rows, res(d).exch_target);
end

% ---- verdict -------------------------------------------------------------
ok_shear = max([res.err_shear])  < tol_rate;
ok_exch  = max([res.err_exch])   < tol_rate;
ok_nu0   = max(abs([res.nu0_ratio] - 1)) < tol_nu0;
ok_vec   = max([res.vec_resid])  < tol_vec;
ok_null  = all([res.n_null] == 5);
% delta-independence: the ratio is the same number at every d_i
spread_shear = max([res.r_shear]) - min([res.r_shear]);

fprintf('\n============ VERDICT ============\n');
fprintf('  lambda_shear/nu_0 = -1/2                   max err %.2e   %s\n', ...
    max([res.err_shear]), tf(ok_shear));
fprintf('  ... and is d_i-INDEPENDENT                 spread  %.2e\n', spread_shear);
fprintf('  lambda_exch/nu_0 = -(3+d_i)/(3+2 d_i)      max err %.2e   %s\n', ...
    max([res.err_exch]), tf(ok_exch));
fprintf('  nu_0 = 4*pi*n (independent of any lambda)  max err %.2e   %s\n', ...
    max(abs([res.nu0_ratio]-1)), tf(ok_nu0));
fprintf('  shear mode is an exact eigenvector         max res %.2e   %s\n', ...
    max([res.vec_resid]), tf(ok_vec));
fprintf('  full operator has exactly 5 null modes                       %s\n', tf(ok_null));
fprintf('\n  total wall time %.1f s (%d builds)\n', t_total, 3*n_d);

if ~(ok_shear && ok_exch && ok_nu0 && ok_vec && ok_null)
    error('benchmark_polyatomic_shear_rate:Failed', ...
          'One or more absolute-rate checks failed; see the table above.');
end

% =========================================================================
%  CSV export -- data artifact for the Section 5.2 shear claim
% =========================================================================
if write_csv
    if ~exist(paper_figs, 'dir')
        warning('Paper figures directory not found, skipping CSV: %s', paper_figs);
    else
        f = fullfile(paper_figs, csv_name);
        fid = fopen(f, 'w');
        fprintf(fid, ['d_i,alpha,n,n_basis,nu0_loss,nu0_loss_algebraic,' ...
                      'four_pi_n,nu0_rel_err,lambda_shear,lambda_shear_over_nu0,' ...
                      'shear_err,lambda_exch,lambda_exch_over_nu0,exch_analytic,' ...
                      'exch_err,shear_eigvec_resid,n_null,max_invariant_row\n']);
        for d = 1:n_d
            fprintf(fid, ['%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,' ...
                          '%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n'], ...
                res(d).delta, res(d).nu_alpha, res(d).n_analytic, res(d).n_basis, ...
                res(d).nu0_spec, res(d).nu0_alg, 4*pi*res(d).n_analytic, ...
                res(d).nu0_ratio - 1, res(d).lam_shear, res(d).r_shear, ...
                res(d).err_shear, res(d).lam_exch, res(d).r_exch, ...
                res(d).exch_target, res(d).err_exch, res(d).vec_resid, ...
                res(d).n_null, res(d).inv_rows);
        end
        fclose(fid);
        fprintf('\n  Wrote %s\n', f);
    end
end

fprintf('\nDONE\n');

function s = tf(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end
