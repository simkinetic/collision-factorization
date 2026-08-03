% benchmark_polyatomic_chapman_enskog.m
% =========================================================================
% Polyatomic infinite-order Chapman-Enskog transport-coefficient convergence
% for the (extended) DSMC Borgnakke-Larsen kernel, evaluated with the
% Wigner-Eckart spectral operator.
%
% This is the polyatomic counterpart of the monatomic section 5.3.1
% "Infinite-Order Chapman-Enskog Inversion" (tutorial_chapman_enscog_decay_rates.m).
% We linearize the collision operator about absolute equilibrium,
%
%     J = dQ/dc|_{c_eq} = C(:,:,1) + C(:,1,:),          (only c_000 = 1)
%
% and, following classical Chapman-Enskog / Sonine theory, read the transport
% relaxation rates off the angular sub-blocks of J:
%
%   * Shear   (L=2, no invariant)          -> shear-viscosity correction f_mu
%   * Bulk    (L=0, mass+energy invariant) -> bulk relaxation rate P_Pi, nu/mu
%   * Heat    (L=1, momentum invariant)    -> Prandtl number Pr
%
% The "infinite-order" part: at radial (Sonine) truncation order kk we invert
% the truncated block and Schur-reduce it onto the first-order macroscopic
% moment(s). Increasing kk, each coefficient converges to its resolved value.
% The first-order truncation reproduces the fixed-block helpers of
% benchmark_dsmc_transport.m / benchmark_dsmc_extended.m exactly.
%
% Because the basis functions R_{k,l} do not depend on K_max, the operator
% entries for k <= kk are identical whether built at K_max = kk or at the top
% resolution. We therefore build ONE operator per model configuration (at the
% top order) and truncate -- the monatomic tutorial's strategy.
%
% PHASES:
%   1a  base DSMC, Maxwell (zeta=0, omega=1)     -> f_mu = 1 at all kk (WCU).
%   1b  base DSMC, hard spheres (zeta=1), frozen -> f_mu -> 205/202 (2x2) and
%       (omega=0)                                    -> 1.016034 (Pekeris limit).
%       These two are machine-precision checks of the convergence machinery.
%   2   extended-model N2 (Table-4 params, Laplace spectral path) -> the result:
%       f_mu, P_Pi, nu/mu, Pr converging in kk.
% =========================================================================

proj = fileparts(mfilename('fullpath'));
addpath(fullfile(proj,'src'));
addpath(genpath(fullfile(proj,'src','SHL')));
addpath(fullfile(proj,'src','mex'));

%% --- Global parameters (all tunable) ------------------------------------
% Flags are honored if pre-set in the workspace (so a -batch run can override,
% e.g. `use_laplace=true; export_to_pdf_figure=false; benchmark_..._enskog`).
if ~exist('export_to_pdf_figure','var'), export_to_pdf_figure = true; end
if ~exist('quick_test','var'),           quick_test = false;          end

L_max = 2;            % required for the L=2 shear mode
I_max = 2;            % internal resolution (held fixed while sweeping radial kk)
K_top = 3;            % top radial order; sweep kk = 0..K_top
pad   = [16 16 4];    % [radial tangential internal] quadrature padding

% Extended-model (Phase 2) evaluation path:
%   use_laplace = false -> algebraic build, ONE 9D quadrature pass per channel
%                          (the path benchmark_dsmc_extended.m uses; ~10 min total).
%   use_laplace = true  -> spectral Laplace path, 2+4*Ns passes: far more accurate
%                          for the non-integer internal exponents but MUCH slower
%                          (~11 min at K_top_ext=2, Ns=4). Cost explodes with K, so
%                          the Laplace run uses its own smaller K_top_ext.
if ~exist('use_laplace','var'), use_laplace = false; end
if ~exist('Ns','var'),          Ns = 4;    end   % Gauss-Jacobi s-nodes (Laplace)
if ~exist('K_top_ext','var'), K_top_ext = K_top; end  % extended-phase radial order
if use_laplace && ~exist('K_top_ext_set','var'), K_top_ext = 2; end
if quick_test,  K_top = 2; K_top_ext = 2; end

% Monatomic reference limits for the shear correction factor f_mu:
f_mu_ref_2 = 205/202;      % 2nd-order (kk=1) hard-sphere fraction
f_mu_ref_inf = 1.016034;   % Pekeris infinite-order hard-sphere limit

fprintf('\n==============================================================\n');
fprintf('  BENCHMARK: Polyatomic Chapman-Enskog transport convergence\n');
fprintf('==============================================================\n');
fprintf('L_max=%d I_max=%d  K_top=%d  padding=(%d,%d,%d)\n', ...
        L_max,I_max,K_top,pad(1),pad(2),pad(3));

%% ========================= PHASE 1: base-model validation ===============
fprintf('\n--- Phase 1a: base DSMC, Maxwell (zeta=0, omega=1) ---\n');
delta_mx = 2.0;  nu_mx = delta_mx/2 - 1;
Bmx = SpectralBasis(K_top, L_max, I_max, nu_mx);
Jmx = buildJ(Bmx, struct('zeta',0.0,'delta',delta_mx,'omega',1.0), pad, false, Ns);
Smx = ce_sweep(Jmx, Bmx, delta_mx, K_top);
print_anchor('Maxwell (zeta=0, omega=1)', ce_anchor(Jmx, Bmx, delta_mx));
print_sweep('Maxwell (zeta=0, omega=1)', Smx);
max_dev_mx = max(abs(Smx.f_mu - 1));
pass_mx = max_dev_mx < 1e-8;
fprintf('  CHECK Maxwell f_mu == 1 : max|f_mu-1| = %.2e  -> %s\n', ...
        max_dev_mx, tern(pass_mx,'PASS','FAIL'));

fprintf('\n--- Phase 1b: base DSMC, hard spheres (zeta=1), frozen (omega=0) ---\n');
delta_hs = 2.0;  nu_hs = delta_hs/2 - 1;
Bhs = SpectralBasis(K_top, L_max, I_max, nu_hs);
Jhs = buildJ(Bhs, struct('zeta',1.0,'delta',delta_hs,'omega',0.0), pad, false, Ns);
Shs = ce_sweep(Jhs, Bhs, delta_hs, K_top);
print_anchor('Hard spheres frozen (zeta=1, omega=0)', ce_anchor(Jhs, Bhs, delta_hs));
print_sweep('Hard spheres frozen (zeta=1, omega=0)', Shs);
err2   = abs(Shs.f_mu(2) - f_mu_ref_2);        % kk=1 -> 2x2 fraction
errinf = abs(Shs.f_mu(end) - f_mu_ref_inf);    % top kk -> Pekeris limit
pass_hs = err2 < 1e-4 && errinf < 1e-4;
fprintf('  CHECK frozen shear -> monatomic:  |f_mu(2x2) - 205/202| = %.2e,  |f_mu(inf) - %.6f| = %.2e  -> %s\n', ...
        err2, f_mu_ref_inf, errinf, tern(pass_hs,'PASS','FAIL'));

%% ========================= PHASE 2: extended-model N2 ===================
fprintf('\n--- Phase 2: extended model N2 (Table-4 params, Laplace path) ---\n');
% N2 Table-4 (benchmark_dsmc_extended.m): delta,zeta,omega,eta_hat,zeta_hat,eta_hat_f,zeta_hat_f
p_n2 = struct('zeta',0.534,'delta',2.01,'omega',0.312052, ...
              'eta_hat',-0.3,'zeta_hat',0.965,'eta_hat_f',-0.207793,'zeta_hat_f',0.3);
fprintf('  path: %s,  K_top_ext=%d%s\n', ...
        tern(use_laplace,'Laplace spectral','algebraic'), K_top_ext, ...
        tern(use_laplace, sprintf(' (Ns=%d)',Ns), ''));
Bn2 = SpectralBasis(K_top_ext, L_max, I_max, p_n2.delta/2-1);
Jn2 = buildJ(Bn2, p_n2, pad, use_laplace, Ns);
An2 = ce_anchor(Jn2, Bn2, p_n2.delta);        % Djordjic 17-moment (1st-order) closure
Sn2 = ce_sweep(Jn2, Bn2, p_n2.delta, K_top_ext);
print_anchor('Extended N2', An2);
print_sweep('Extended N2', Sn2);
print_cauchy('Extended N2', Sn2);

% Experimental N2 targets (Djordjic Tables 2-3); the params were FIT so the
% 1st-order (17-moment) closure -- NOT the infinite-order model -- hits these.
Pr_exp = 0.717;  numu_exp = 0.73;
fprintf('\n  N2: Djordjic 1st-order closure  vs  our infinite-order (SAME operator)\n');
fprintf('    experiment (fit target):  Pr=%.4f   nu/mu=%.4f\n', Pr_exp, numu_exp);
fprintf('    1st-order (anchor):        Pr=%.4f   nu/mu=%.4f   f_mu=1 (no Sonine corr.)\n', An2.Pr, An2.numu);
fprintf('    infinite-order (resolved): Pr=%.4f   nu/mu=%.4f   f_mu=%.6f\n', Sn2.Pr(end), Sn2.numu(end), Sn2.f_mu(end));
fprintf('    higher-order shift:        dPr=%+.4f  d(nu/mu)=%+.4f  d(f_mu)=%+.6f\n', ...
        Sn2.Pr(end)-An2.Pr, Sn2.numu(end)-An2.numu, Sn2.f_mu(end)-1);

%% ========================= Save results + figure =======================
% Persist everything needed to re-plot (no recompute) and to drop into the paper.
R = struct();
R.kk = Sn2.kk;  R.n2_f_mu = Sn2.f_mu;  R.n2_numu = Sn2.numu;  R.n2_Pr = Sn2.Pr;
R.hs_kk = Shs.kk;  R.hs_f_mu = Shs.f_mu;
R.mx_kk = Smx.kk;  R.mx_f_mu = Smx.f_mu;
R.anchor_f_mu = An2.f_mu;  R.anchor_numu = An2.numu;  R.anchor_Pr = An2.Pr;
R.f_mu_pekeris = f_mu_ref_inf;  R.Pr_exp = Pr_exp;  R.numu_exp = numu_exp;
R.params = p_n2;  R.pad = pad;  R.I_max = I_max;  R.L_max = L_max;
R.K_top = K_top;  R.K_top_ext = K_top_ext;  R.use_laplace = use_laplace;

datadir = fullfile(proj,'paper','data');
if ~isfolder(datadir), mkdir(datadir); end
matfile = fullfile(datadir,'polyce_results.mat');
save(matfile,'R');
fprintf('\n  results saved: %s\n', matfile);
write_n2_table(fullfile(datadir,'polyce_n2_convergence.tex'), Sn2, An2, Pr_exp, numu_exp);
fprintf('  paper table  : %s\n', fullfile(datadir,'polyce_n2_convergence.tex'));

if export_to_pdf_figure
    outpdf = fullfile(proj,'paper','figures','fig_polyce_convergence.pdf');
    plot_polyce_convergence(R, outpdf);
end

fprintf('\nDone.\n');

% ========================= core sweep ====================================
function S = ce_sweep(J, Basis, delta, K_top)
% Extract shear f_mu, bulk P_Pi & nu/mu, and Prandtl Pr at each radial
% truncation kk = 0..K_top by block-inversion / Schur reduction of J.
    N_I = Basis.I_max + 1;  N_Q = (Basis.L_max + 1)^2;  Imax = Basis.I_max;
    idx = @(k,i,l,m) (k*N_I+i)*N_Q + (l^2+l+m) + 1;
    rho = sqrt(5/(4*delta));
    kks = 0:K_top;  n = numel(kks);
    [f_mu, lam_sh, P_Pi, numu, Pr] = deal(nan(1,n));
    nullc = zeros(n,3);                     % [L=2  L=0  L=1] near-zero eigencounts

    lam0 = -J(idx(0,0,2,0), idx(0,0,2,0));  % first-order shear rate (constant)

    for a = 1:n
        kk = kks(a);

        % ---- Shear (L=2): lead (0,0,2,0), Schur-reduce onto it (1x1) -----
        B2 = mk_block(idx, 2, [0 0], zeros(0,2), kk, Imax);
        M2 = -J(B2,B2);  M2inv = inv(M2);
        lam_sh(a) = 1 / M2inv(1,1);         % effective shear relaxation rate
        f_mu(a)   = lam0 / lam_sh(a);       % viscosity correction factor
        nullc(a,1) = count_null(M2);

        % ---- Bulk (L=0): lead {(1,0,0,0),(0,1,0,0)}, mass excluded -------
        %      Energy is the single remaining null; bulk rate = slowest
        %      non-zero eigenvalue (generalizes bulkshear's evb(2)).
        if kk >= 1
            B0 = mk_block(idx, 0, [1 0; 0 1], [0 0], kk, Imax);
            M0 = -J(B0,B0);
            ev = sort(real(eig(M0)), 'ascend');
            thr = 1e-8 * max(1, norm(M0,'fro'));
            nullc(a,2) = sum(abs(ev) < thr);
            pos = ev(ev > thr);
            P_Pi(a) = min(pos);
            numu(a) = (2*delta/(3*(3+delta))) * lam_sh(a) / P_Pi(a);
        end

        % ---- Heat (L=1): lead {(1,0,1,0),(0,1,1,0)}, momentum excluded ---
        %      Schur-reduce onto the 2x2, then the prandtl() combination.
        if kk >= 1
            B1 = mk_block(idx, 1, [1 0; 0 1], [0 0], kk, Imax);
            M1 = -J(B1,B1);  M1inv = inv(M1);
            JL1 = -inv(M1inv(1:2,1:2));      % effective J on {q0, s0}
            Pq0 = -JL1(1,1);  Ps0 = -JL1(2,2);
            Pq1 = -JL1(1,2)*rho;  Ps1 = -JL1(2,1)/rho;
            Pr(a) = (5+delta)/2 * 2*(Pq0*Ps0 - Pq1*Ps1) / ...
                    (lam_sh(a)*(5*(Ps0-Ps1) + delta*(Pq0-Pq1)));
            nullc(a,3) = count_null(M1);
        end
    end
    S = struct('kk',kks,'f_mu',f_mu,'lam_sh',lam_sh,'P_Pi',P_Pi, ...
               'numu',numu,'Pr',Pr,'nullc',nullc);
end

% ========================= helpers =======================================
function J = buildJ(Basis, p, pad, laplace, Ns)
    Kd = ScatteringKernel('DSMC', p);
    T  = GeneralCollisionTensor(Basis, Kd);
    if laplace
        T.laplace_extended = true;
        T.laplace_Ns = Ns;
    end
    T.generate_R_tensor_sumfac(pad(1), pad(2), pad(3));
    C = T.assemble_full_tensor();
    J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
end

function B = mk_block(idx, l, lead_ki, excl_ki, kk, Imax)
% Linear indices of the (m=0) modes of angular degree l: the first-order
% "lead" moments first, then all radial-degree<=kk correction modes (all
% internal i), excluding the lead and any explicitly conserved modes.
    B = zeros(1, size(lead_ki,1));
    for r = 1:size(lead_ki,1)
        B(r) = idx(lead_ki(r,1), lead_ki(r,2), l, 0);
    end
    corr = [];
    for k = 0:kk
        for i = 0:Imax
            ki = [k i];
            if row_in(lead_ki, ki) || row_in(excl_ki, ki), continue; end
            corr(end+1) = idx(k, i, l, 0); %#ok<AGROW>
        end
    end
    B = [B, corr];
end

function t = row_in(M, r)
    t = ~isempty(M) && any(all(M == r, 2));
end

function c = count_null(M)
    c = sum(abs(eig(M)) < 1e-8 * max(1, norm(M,'fro')));
end

function print_sweep(name, S)
    fprintf('  %s:\n', name);
    fprintf('   kk |   f_mu    |  lam_shear |    P_Pi    |   nu/mu   |    Pr     | nulls(L2,L0,L1)\n');
    for a = 1:numel(S.kk)
        fprintf('   %2d | %9.6f | %10.5f | %10s | %9s | %9s |   (%d,%d,%d)\n', ...
            S.kk(a), S.f_mu(a), S.lam_sh(a), ...
            fnum(S.P_Pi(a),'%10.5f'), fnum(S.numu(a),'%9.4f'), fnum(S.Pr(a),'%9.5f'), ...
            S.nullc(a,1), S.nullc(a,2), S.nullc(a,3));
    end
end

function print_cauchy(name, S)
    fprintf('  %s -- Cauchy differences to top order (kk=%d):\n', name, S.kk(end));
    fprintf('   kk |  |df_mu|  |  |dnu/mu| |   |dPr|\n');
    for a = 1:numel(S.kk)
        fprintf('   %2d | %9.2e | %9s | %9s\n', S.kk(a), ...
            abs(S.f_mu(a)-S.f_mu(end)), ...
            fnum(abs(S.numu(a)-S.numu(end)),'%9.2e'), ...
            fnum(abs(S.Pr(a)-S.Pr(end)),'%9.2e'));
    end
end

function write_n2_table(fname, Sn2, An2, Pr_exp, numu_exp)
% Emit a booktabs table of the N2 transport-coefficient convergence for the
% paper: reference rows (experiment, our 1st-order) then the K_max sweep.
    fid = fopen(fname,'w');
    fprintf(fid,'%% Auto-generated by benchmark_polyatomic_chapman_enskog.m\n');
    fprintf(fid,'\\begin{tabular}{lccc}\n\\toprule\n');
    fprintf(fid,' & $f_\\mu$ & $\\nu/\\mu$ & $\\mathrm{Pr}$ \\\\\n\\midrule\n');
    fprintf(fid,'Experiment (Djordji\\''c) & --- & %.3f & %.3f \\\\\n', numu_exp, Pr_exp);
    fprintf(fid,'1st-order (prod. coeffs) & %.4f & %.4f & %.4f \\\\\n', An2.f_mu, An2.numu, An2.Pr);
    fprintf(fid,'\\midrule\n');
    for a = 1:numel(Sn2.kk)
        if isnan(Sn2.numu(a))
            fprintf(fid,'$K_{\\max}=%d$ & %.4f & --- & --- \\\\\n', Sn2.kk(a), Sn2.f_mu(a));
        else
            fprintf(fid,'$K_{\\max}=%d$ & %.4f & %.4f & %.4f \\\\\n', ...
                Sn2.kk(a), Sn2.f_mu(a), Sn2.numu(a), Sn2.Pr(a));
        end
    end
    fprintf(fid,'\\bottomrule\n\\end{tabular}\n');
    fclose(fid);
end

% ---- Djordjic 17-moment (first-order / lowest-Sonine) transport closure ----
% Exactly the fixed-block formulas of benchmark_dsmc_transport.m == paper eqs
% (36)-(39): shear mu = p/P_sigma^(0) is a pure 1x1 (no Sonine correction, so
% f_mu = 1); bulk uses the L=0 n=1 2x2; heat uses the L=1 2x2. These are the
% quantities the N2 parameters were fit against.
function A = ce_anchor(J, Basis, delta)
    N_I = Basis.I_max + 1;  N_Q = (Basis.L_max + 1)^2;
    A.f_mu = 1.0;
    A.numu = bulkshear(J, delta, N_I, N_Q);
    A.Pr   = prandtl(J, delta, sqrt(5/(4*delta)), N_I, N_Q);
end

function Pr = prandtl(J, delta, rho, N_I, N_Q)
    idx = @(k,i,l,m) (k*N_I+i)*N_Q + (l^2+l+m) + 1;
    P_sigma = -J(idx(0,0,2,0), idx(0,0,2,0));
    iq = idx(1,0,1,0); is = idx(0,1,1,0); JL1 = J([iq is],[iq is]);
    Pq0 = -JL1(1,1); Ps0 = -JL1(2,2);
    Pq1 = -JL1(1,2)*rho; Ps1 = -JL1(2,1)/rho;
    Pr = (5+delta)/2 * 2*(Pq0*Ps0 - Pq1*Ps1) / ...
         (P_sigma*(5*(Ps0-Ps1) + delta*(Pq0-Pq1)));
end

function r = bulkshear(J, delta, N_I, N_Q)
    idx = @(k,i,l,m) (k*N_I+i)*N_Q + (l^2+l+m) + 1;
    P_sigma = -J(idx(0,0,2,0), idx(0,0,2,0));
    bidx = [idx(1,0,0,0), idx(0,1,0,0)];
    evb = sort(real(eig(J(bidx,bidx))),'descend');
    P_Pi = -evb(2);                       % 1st is the ~0 energy invariant
    r = (2*delta/(3*(3+delta))) * (P_sigma/P_Pi);
end

function print_anchor(name, A)
    fprintf('  %s -- 1st-order (Djordjic 17-moment): f_mu=%.6f  nu/mu=%.4f  Pr=%.5f\n', ...
            name, A.f_mu, A.numu, A.Pr);
end

function s = fnum(x, fmt)
    if isnan(x), s = '    --   '; else, s = sprintf(fmt, x); end
end
function s = tern(c,a,b), if c, s=a; else, s=b; end, end
