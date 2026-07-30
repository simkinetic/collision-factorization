%% BENCHMARK: Polyatomic Two-Temperature (Landau-Teller) Relaxation
% Spatially-homogeneous (0D) relaxation of a genuine product-of-Maxwellians
% initial condition  M_v(T_v0) * M_I(T_I0)  under the DSMC Borgnakke-Larsen
% collision operator. The translational temperature T_v and the internal
% temperature T_I equilibrate to a common T_eq while the TOTAL energy is
% conserved to machine precision.
%
% Two sweeps:
%   A) internal DOF d_i in {2,4,6} (fixed omega): same (T_v0,T_I0) relaxes to
%      DIFFERENT equilibria  T_eq = (3 T_v0 + d_i T_I0)/(3 + d_i).
%   B) DSMC split omega in {0.25,0.5,0.75,1.0} (fixed d_i=2): SAME T_eq (energy
%      conservation is omega-independent) but DIFFERENT rates. The frozen
%      channel is elastic and leaves both energy modes invariant, so the
%      (T_v - T_I) decay rate is exactly proportional to omega.
%
% Reference style: fig:bkw_relax / fig:bkw_cons of the monatomic paper
% (bib/wigner-eckart-monatomic/section5.tex, tutorial_collisions_bwu_solution.m).

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL');

fprintf('==============================================================\n');
fprintf('  BENCHMARK: Polyatomic Two-Temperature Relaxation (DSMC)\n');
fprintf('==============================================================\n\n');

export_to_pdf_figure = true;

%% --- Global parameters (all tunable) ------------------------------------
K_max = 2; L_max = 2; I_max = 2;
rad_pad = 10; tan_pad = 10; int_pad = 10;
zeta = 0.5;                 % s_visc = 0.75  ->  |u|^zeta
% Initial temperatures. Keep the gap modest so the product-Maxwellian projects
% cleanly at low K (Sonine truncation ~ ((T-1)/T)^K); T_v0=1.4 gives ~2% at K=2.
T_v0 = 1.4;   T_I0 = 1.0;

Teq_of = @(di) (3*T_v0 + di*T_I0) / (3 + di);   % predicted equilibrium temperature

fprintf('Resolution K=%d L=%d I=%d  padding=(%d,%d,%d)  zeta=%.3f\n', ...
    K_max, L_max, I_max, rad_pad, tan_pad, int_pad, zeta);
fprintf('Initial temperatures: T_v0=%.3f  T_I0=%.3f\n\n', T_v0, T_I0);

%% ========================================================================
%  SWEEP A: internal degrees of freedom d_i  (different equilibria)
%  ========================================================================
di_list = [2, 3, 4];
sweepA = struct('di',{},'t',{},'Tv',{},'Ti',{},'Teq_num',{},'rate',{});

fprintf('--- Sweep A: internal DOF d_i (omega = 1) ---\n');
for a = 1:numel(di_list)
    d_i = di_list(a);   delta = d_i;   nu = delta/2 - 1;
    fprintf('  Building DSMC operator: d_i=%d (nu=%.2f)...\n', d_i, nu);

    Basis = SpectralBasis(K_max, L_max, I_max, nu);
    C = build_C(Basis, delta, zeta, 1.0, rad_pad, tan_pad, int_pad);
    calib = calibrate_temperature(Basis);

    tg = make_time_grid(C, Basis);
    tgrid = 0 : tg.dt : tg.T_end;
    R = relax_two_temp(C, Basis, calib, T_v0, T_I0, tgrid);

    sweepA(a).di = d_i;   sweepA(a).t = R.t;
    sweepA(a).Tv = R.Tv;  sweepA(a).Ti = R.Ti;
    sweepA(a).Teq_num = 0.5*(R.Tv(end) + R.Ti(end));
    sweepA(a).rate = fit_rate(R.t, R.Tv, R.Ti);
    sweepA(a).rate_ex = tg.rate_ex;
    fprintf('    T_eq: pred=%.4f num=%.4f | rate: fit=%.4f eig=%.4f | max|l>=1|=%.1e\n', ...
        Teq_of(d_i), sweepA(a).Teq_num, sweepA(a).rate, sweepA(a).rate_ex, R.aniso);
end
fprintf('\n');

%% ========================================================================
%  SWEEP B: DSMC split omega at d_i = 2  (same equilibrium, different rates)
%  ========================================================================
delta_B = 2.0;  nu_B = delta_B/2 - 1;   % d_i = 2, nu = 0
omega_list = [0.25, 0.5, 0.75, 1.0];
sweepB = struct('omega',{},'t',{},'Tv',{},'Ti',{},'Teq_num',{},'rate',{});

fprintf('--- Sweep B: DSMC omega at d_i=%g (build 2 channels once, blend) ---\n', delta_B);
Basis_B = SpectralBasis(K_max, L_max, I_max, nu_B);
calib_B = calibrate_temperature(Basis_B);
fprintf('  Building non-frozen (omega=1) and frozen (omega=0) channels...\n');
C_nf = build_C(Basis_B, delta_B, zeta, 1.0, rad_pad, tan_pad, int_pad);
C_fr = build_C(Basis_B, delta_B, zeta, 0.0, rad_pad, tan_pad, int_pad);

% Common time grid: slowest omega sets T_end, fastest sets dt.
tg_slow = make_time_grid(omega_list(1)*C_nf + (1-omega_list(1))*C_fr, Basis_B);
tg_fast = make_time_grid(omega_list(end)*C_nf + (1-omega_list(end))*C_fr, Basis_B);
tgrid_B = 0 : tg_fast.dt : tg_slow.T_end;

for b = 1:numel(omega_list)
    w = omega_list(b);
    C = w*C_nf + (1-w)*C_fr;                 % assembled tensor is linear in omega
    tgw = make_time_grid(C, Basis_B);        % eigenvalue rate for reference
    R = relax_two_temp(C, Basis_B, calib_B, T_v0, T_I0, tgrid_B);

    sweepB(b).omega = w;   sweepB(b).t = R.t;
    sweepB(b).Tv = R.Tv;   sweepB(b).Ti = R.Ti;
    sweepB(b).Teq_num = 0.5*(R.Tv(end) + R.Ti(end));
    sweepB(b).rate = fit_rate(R.t, R.Tv, R.Ti);
    sweepB(b).rate_ex = tgw.rate_ex;
    fprintf('    omega=%.2f: T_eq_num=%.4f | rate: fit=%.4f eig=%.4f\n', ...
        w, sweepB(b).Teq_num, sweepB(b).rate, sweepB(b).rate_ex);
end

% Conservation figure (d_i=2, omega=1). Two operators on the SAME grid/integrator:
%   black -- the experiment's model (zeta=%.2f): |u|^zeta needs quadrature, so any
%            drift here mixes RK2 error with quadrature error.
%   red   -- a Maxwell control (zeta=0): the base-DSMC R-tensor is polynomial-exact,
%            so this conserves to pure machine/RK2 precision. If black drifts ABOVE
%            red, the excess is the |u|^zeta quadrature -- ruling out a quadrature bug.
C_mx    = build_C(Basis_B, delta_B, 0.0, 1.0, rad_pad, tan_pad, int_pad);   % Maxwell control
Rcons   = relax_two_temp(C_nf, Basis_B, calib_B, T_v0, T_I0, tgrid_B);      % black: zeta=0.5 model
Rcons_mx= relax_two_temp(C_mx, Basis_B, calib_B, T_v0, T_I0, tgrid_B);      % red:  zeta=0 exact
fprintf('  Conservation (d_i=%g, omega=1):\n', delta_B);
fprintf('    model   (zeta=%.2f): max|dMass|=%.2e  max|dE_total|=%.2e\n', ...
    zeta, max(abs(Rcons.mass - Rcons.mass(1))), max(abs(Rcons.E - Rcons.E(1))));
fprintf('    Maxwell (zeta=0.00): max|dMass|=%.2e  max|dE_total|=%.2e  [quadrature-exact control]\n', ...
    max(abs(Rcons_mx.mass - Rcons_mx.mass(1))), max(abs(Rcons_mx.E - Rcons_mx.E(1))));
fprintf('\n');

%% ========================================================================
%  CONSOLE SUMMARY TABLES
%  ========================================================================
fprintf('=== Sweep A: equilibrium temperature vs d_i ===\n');
fprintf('  %4s | %10s | %10s\n', 'd_i', 'T_eq_pred', 'T_eq_num');
for a = 1:numel(sweepA)
    fprintf('  %4d | %10.4f | %10.4f\n', sweepA(a).di, Teq_of(sweepA(a).di), sweepA(a).Teq_num);
end
fprintf('\n=== Sweep B: decay rate vs omega (d_i=%g) -- rate proportional to omega ===\n', delta_B);
fprintf('  (frozen channel is elastic -> leaves both energy modes invariant -> rate/omega is constant)\n');
fprintf('  %6s | %10s | %12s | %12s | %12s | %12s\n', ...
    'omega', 'T_eq_num', 'rate_fit', 'rate_eig', 'rate_fit/w', 'rate_eig/w');
for b = 1:numel(sweepB)
    fprintf('  %6.2f | %10.4f | %12.5f | %12.5f | %12.5f | %12.5f\n', ...
        sweepB(b).omega, sweepB(b).Teq_num, sweepB(b).rate, sweepB(b).rate_ex, ...
        sweepB(b).rate/sweepB(b).omega, sweepB(b).rate_ex/sweepB(b).omega);
end
fprintf('\n');

%% ========================================================================
%  FIGURES
%  ========================================================================
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
FS_labels = 15; FS_ticks = 12; FS_legend = 11; FS_title = 15;
colA = lines(numel(di_list));
colB = lines(numel(omega_list));

outdir = fullfile('paper','figures');
if ~exist(outdir,'dir'), mkdir(outdir); end

% ---- Figure 1: Sweep A (internal DOF) -----------------------------------
fig1 = figure('Name','Sweep A: internal DOF','Position',[80 80 1000 420],'Color','w');

subplot(1,2,1); hold on; grid on;
for a = 1:numel(sweepA)
    c = colA(a,:);
    plot(sweepA(a).t, sweepA(a).Tv, '-',  'Color', c, 'LineWidth', 2);
    plot(sweepA(a).t, sweepA(a).Ti, '--', 'Color', c, 'LineWidth', 2);
    yline(Teq_of(sweepA(a).di), ':', 'Color', c, 'LineWidth', 1.2);
end
xlabel('Time $t$','FontSize',FS_labels);
ylabel('Temperature','FontSize',FS_labels);
title('\textbf{(a) Relaxation to $T_{\mathrm{eq}}(d_i)$}','FontSize',FS_title);
set(gca,'FontSize',FS_ticks,'LineWidth',1.1);
xlim([0 0.1]);                      % zoom on the informative decay window
lg = arrayfun(@(s) sprintf('$d_i=%d$', s.di), sweepA, 'UniformOutput', false);
% one representative pair for the legend semantics
text(0.98,0.05,'solid: $T_v$\quad dashed: $T_I$\quad dotted: $T_{\mathrm{eq}}$', ...
    'Units','normalized','HorizontalAlignment','right','FontSize',FS_legend, ...
    'BackgroundColor',[1 1 1 0.7],'EdgeColor','k','Margin',3);

subplot(1,2,2); hold on; grid on;
set(gca,'YScale','log');
h = gobjects(1,numel(sweepA));
for a = 1:numel(sweepA)
    dT = max(abs(sweepA(a).Tv - sweepA(a).Ti), eps);
    h(a) = plot(sweepA(a).t, dT, '-', 'Color', colA(a,:), 'LineWidth', 2);
end
set(gca,'FontSize',FS_ticks,'LineWidth',1.1);
xlim([0 0.1]); ylim([1e-3 1e0]);
xlabel('Time $t$','FontSize',FS_labels);
ylabel('$|T_v - T_I|$','FontSize',FS_labels);
title('\textbf{(b) Exponential relaxation}','FontSize',FS_title);
legend(h, lg, 'Location','northeast','FontSize',FS_legend);
if export_to_pdf_figure
    exportgraphics(fig1, fullfile(outdir,'fig_polyrelax_sweep_di.pdf'), 'ContentType','vector');
end

% ---- Figure 2: Sweep B (omega) ------------------------------------------
fig2 = figure('Name','Sweep B: DSMC omega','Position',[120 120 1000 420],'Color','w');

subplot(1,2,1); hold on; grid on;
for b = 1:numel(sweepB)
    c = colB(b,:);
    plot(sweepB(b).t, sweepB(b).Tv, '-',  'Color', c, 'LineWidth', 2);
    plot(sweepB(b).t, sweepB(b).Ti, '--', 'Color', c, 'LineWidth', 2);
end
yline(Teq_of(delta_B), 'k:', 'LineWidth', 1.5);
xlabel('Time $t$','FontSize',FS_labels);
ylabel('Temperature','FontSize',FS_labels);
title(sprintf('\\textbf{(a) Same $T_{\\mathrm{eq}}=%.3f$ for all $\\omega$}', Teq_of(delta_B)),'FontSize',FS_title);
set(gca,'FontSize',FS_ticks,'LineWidth',1.1);
xlim([0 0.25]);                     % zoom on the informative decay window
text(0.98,0.05,'solid: $T_v$\quad dashed: $T_I$','Units','normalized', ...
    'HorizontalAlignment','right','FontSize',FS_legend, ...
    'BackgroundColor',[1 1 1 0.7],'EdgeColor','k','Margin',3);

subplot(1,2,2); hold on; grid on;
set(gca,'YScale','log');
h = gobjects(1,numel(sweepB)); lgB = cell(1,numel(sweepB));
for b = 1:numel(sweepB)
    dT = max(abs(sweepB(b).Tv - sweepB(b).Ti), eps);
    h(b) = plot(sweepB(b).t, dT, '-', 'Color', colB(b,:), 'LineWidth', 2);
    lgB{b} = sprintf('$\\omega=%.2f$', sweepB(b).omega);
end
set(gca,'FontSize',FS_ticks,'LineWidth',1.1);
xlim([0 0.25]); ylim([1e-3 1e0]);
xlabel('Time $t$','FontSize',FS_labels);
ylabel('$|T_v - T_I|$','FontSize',FS_labels);
title('\textbf{(b) Slope $\propto \omega$}','FontSize',FS_title);
legend(h, lgB, 'Location','northeast','FontSize',FS_legend);
if export_to_pdf_figure
    exportgraphics(fig2, fullfile(outdir,'fig_polyrelax_sweep_omega.pdf'), 'ContentType','vector');
end

% ---- Figure 3: conservation -- model (black) vs Maxwell control (red) ----
% Same integrator/grid for both; the Maxwell (zeta=0) operator is quadrature-exact,
% so it isolates whether the model's tiny drift is quadrature or the RK2 stepper.
fig3 = figure('Name','Conservation','Position',[160 160 700 440],'Color','w');
hold on; grid on;
sub = round(linspace(1, numel(Rcons.t), 12));
mass_err = max(abs(Rcons.mass    - Rcons.mass(1)),    eps);
E_err    = max(abs(Rcons.E       - Rcons.E(1)),       eps);
mass_mx  = max(abs(Rcons_mx.mass - Rcons_mx.mass(1)), eps);
E_mx     = max(abs(Rcons_mx.E    - Rcons_mx.E(1)),    eps);
semilogy(Rcons.t, mass_err, 'k-o', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName',sprintf('Mass, model ($\\zeta=%.2f$)', zeta));
semilogy(Rcons.t, E_err, 'k-s', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName',sprintf('Energy, model ($\\zeta=%.2f$)', zeta));
semilogy(Rcons_mx.t, mass_mx, 'r-o', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName','Mass, Maxwell ($\zeta=0$)');
semilogy(Rcons_mx.t, E_mx, 'r-s', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName','Energy, Maxwell ($\zeta=0$)');
set(gca,'YScale','log','FontSize',FS_ticks,'LineWidth',1.1);
ylim([1e-18, 1e-8]); yticks(10.^(-18:2:-8));
xlabel('Time $t$','FontSize',FS_labels);
ylabel('Absolute drift','FontSize',FS_labels);
title(sprintf('\\textbf{Conservation ($d_i=%g,\\ \\omega=1$): model vs Maxwell control}', delta_B),'FontSize',FS_title);
legend('Location','east','FontSize',FS_legend);
if export_to_pdf_figure
    exportgraphics(fig3, fullfile(outdir,'fig_polyrelax_conservation.pdf'), 'ContentType','vector');
end

%% ========================================================================
%  SAVE RAW RESULTS (re-plotting + paper inclusion)
%  ========================================================================
params = struct('K_max',K_max,'L_max',L_max,'I_max',I_max, ...
    'rad_pad',rad_pad,'tan_pad',tan_pad,'int_pad',int_pad, ...
    'zeta',zeta,'T_v0',T_v0,'T_I0',T_I0, ...
    'di_list',di_list,'omega_list',omega_list,'delta_B',delta_B);
save(fullfile(outdir,'polyrelax_results.mat'), ...
    'sweepA','sweepB','Rcons','Rcons_mx','params','-v7');

% Sweep A: grids differ per case -> one CSV each
for a = 1:numel(sweepA)
    dT = abs(sweepA(a).Tv - sweepA(a).Ti);
    Tb = table(sweepA(a).t(:), sweepA(a).Tv(:), sweepA(a).Ti(:), dT(:), ...
        'VariableNames', {'t','Tv','Ti','absdT'});
    writetable(Tb, fullfile(outdir, sprintf('polyrelax_A_di%d.csv', sweepA(a).di)));
end

% Sweep B: shared grid -> single wide CSV
Bmat = sweepB(1).t(:);  names = {'t'};
for b = 1:numel(sweepB)
    dT = abs(sweepB(b).Tv - sweepB(b).Ti);
    Bmat = [Bmat, sweepB(b).Tv(:), sweepB(b).Ti(:), dT(:)]; %#ok<AGROW>
    ws = strrep(sprintf('%.2f', sweepB(b).omega), '.', 'p');
    names = [names, {['Tv_w' ws], ['Ti_w' ws], ['absdT_w' ws]}]; %#ok<AGROW>
end
writetable(array2table(Bmat,'VariableNames',names), fullfile(outdir,'polyrelax_B.csv'));

% Conservation: model vs Maxwell drift
Cb = table(Rcons.t(:), abs(Rcons.mass-Rcons.mass(1)), abs(Rcons.E-Rcons.E(1)), ...
    abs(Rcons_mx.mass-Rcons_mx.mass(1)), abs(Rcons_mx.E-Rcons_mx.E(1)), ...
    'VariableNames', {'t','mass_model','E_model','mass_maxwell','E_maxwell'});
writetable(Cb, fullfile(outdir,'polyrelax_conservation.csv'));

% LaTeX summary table for the paper
write_relaxation_table(fullfile('paper','table_temperature_relaxation.tex'), ...
    sweepA, sweepB, Teq_of, delta_B);

fprintf('Saved: %s, per-case CSVs in %s, and paper/table_temperature_relaxation.tex\n', ...
    fullfile(outdir,'polyrelax_results.mat'), outdir);

fprintf('Done.\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function C = build_C(Basis, delta, zeta, omega, rp, tp, ip)
    % Assemble the dense DSMC collision tensor (mirror of benchmark_dsmc_transport build_J).
    Kd = ScatteringKernel('DSMC', struct('zeta',zeta,'delta',delta,'omega',omega));
    T  = GeneralCollisionTensor(Basis, Kd);
    T.generate_R_tensor_sumfac(rp, tp, ip);
    C  = T.assemble_full_tensor();
end

function tg = make_time_grid(C, Basis)
    % Time grid from the linearized operator. The (T_v-T_I) exchange rate is the
    % nonzero eigenvalue of the 2x2 energy-exchange block {(1,0,0),(0,1,0)}; this
    % is the eigenmode the T_v-T_I combination projects onto (and it is exactly
    % proportional to omega, since the frozen elastic channel leaves both energy
    % modes invariant). dt is set from the fastest linear mode (RK2 stability).
    N = Basis.N_terms;
    J = squeeze(C(1:N,1:N,1)) + squeeze(C(1:N,1,1:N));   % linearize at Maxwellian (c_eq(1)=1)
    idx = @(k,i,q) (k*Basis.N_I + i)*Basis.N_Q + q;
    blk = [idx(1,0,1), idx(0,1,1)];                     % {(k,i,l)=(1,0,0),(0,1,0)}
    ev  = sort(real(eig(J(blk,blk))));                  % {~0, -lambda}
    rate_ex = max(-ev(1), 1e-6);                        % energy-exchange rate (linearized)
    mu_max  = max(abs(real(eig(J))));                   % fastest linear mode
    tg.rate_ex = rate_ex;
    tg.T_end = 12.0 / rate_ex;
    tg.dt    = min(0.1/mu_max, tg.T_end/5000);
end

function calib = calibrate_temperature(Basis)
    % Pin the linear moment-functional constants so that
    %   <v^2> = 3*c000 + a1*c100 ,  <I> = (nu+1)*c000 + b1*c010.
    % a0=3, b0=nu+1 exact at equilibrium; a1,b1 from a single perturbed Maxwellian.
    nu = Basis.nu;
    th = 1.2; % artificial perturbation to get a nonzero c100, c010 for calibration
    cv = project_product_maxwellian(th, 1.0, Basis);   % translationally warm
    ci = project_product_maxwellian(1.0, th,  Basis);  % internally warm
    idx = @(k,i,q) (k*Basis.N_I + i)*Basis.N_Q + q;
    c100_v = cv(idx(1,0,1));
    c010_i = ci(idx(0,1,1));
    calib.a0 = 3;         calib.a1 = 3*(th-1)      / c100_v;    % <v^2> = 3 theta
    calib.b0 = nu + 1;    calib.b1 = (nu+1)*(th-1) / c010_i;    % <I>   = (nu+1) theta
    calib.idx = idx;
end

function c = project_product_maxwellian(theta_v, theta_I, Basis)
    % Project M_v(theta_v)*M_I(theta_I) onto the l=0 spectral modes by tensor-product
    % Gauss-Laguerre quadrature (generalizes project_bkw to the internal dimension).
    % Renormalized so the density mode c000 = 1.
    N_rad = 80; N_int = 40;
    qv = Gauss.generalized_laguerre(N_rad, 0.5);        % nodes x = v^2, weight x^{1/2} e^{-x}
    qI = Gauss.generalized_laguerre(N_int, Basis.nu);   % nodes I,     weight I^{nu} e^{-I}
    nu = Basis.nu;

    [XV, XI] = ndgrid(qv.x, qI.x);
    [WV, WI] = ndgrid(qv.w, qI.w);
    vsq = XV(:); Iv = XI(:);
    v_vec = [sqrt(vsq), zeros(numel(vsq), 2)];

    Fv = pi^(-1.5) * theta_v^(-1.5) .* exp(-vsq .* (1/theta_v - 1));          % M_v / e^{-v^2}
    FI = theta_I^(-(nu+1)) / gamma(nu+1) .* exp(-Iv .* (1/theta_I - 1));      % M_I ratio (GL(nu) measure)
    wtot = (2*pi*WV(:)) .* WI(:) .* Fv .* FI;

    Psi = Basis.evaluate(v_vec, Iv);        % [Np x N_terms], includes R*Y*H
    c_all = Psi' * wtot;

    % keep only isotropic (l=0, q=1) modes; other single-point l>0 samples are invalid
    c = zeros(Basis.N_terms, 1);
    for k = 0:Basis.K_max
        for i = 0:Basis.I_max
            idx = (k*Basis.N_I + i)*Basis.N_Q + 1;
            c(idx) = c_all(idx);
        end
    end
    c = c / c(1);   % density mode is index 1 -> renormalize to unit density
end

function R = relax_two_temp(C, Basis, calib, Tv0, Ti0, tgrid)
    % Nonlinear RK2 (Heun) relaxation of the product-Maxwellian IC; returns
    % temperature, total-energy and mass histories.
    N = Basis.N_terms;
    C_flat = reshape(C, size(C,1), N^2);

    c_init = project_product_maxwellian(Tv0, Ti0, Basis);

    t = tgrid(:); nT = numel(t); dt = t(2) - t(1);
    c = zeros(nT, N); c(1,:) = c_init';
    for n = 1:nT-1
        cn = c(n,:)';
        q1 = C_flat * reshape(cn*cn', N^2, 1);   q1 = q1(1:N);
        ct = cn + dt*q1;
        q2 = C_flat * reshape(ct*ct', N^2, 1);   q2 = q2(1:N);
        c(n+1,:) = (cn + 0.5*dt*(q1+q2))';
    end

    idx = calib.idx; nu = Basis.nu;
    c000 = c(:, idx(0,0,1));
    c100 = c(:, idx(1,0,1));
    c010 = c(:, idx(0,1,1));
    v2   = calib.a0.*c000 + calib.a1.*c100;      % <v^2>
    Iint = calib.b0.*c000 + calib.b1.*c010;      % <I>
    R.t  = t;
    R.Tv = v2 ./ (3*c000);
    R.Ti = Iint ./ ((nu+1)*c000);
    R.E  = 0.5*v2 + Iint;                        % total energy (translational + internal)
    R.mass = c000;

    % anisotropy check: largest l>=1 amplitude over the run
    isL0 = false(1, N);
    for k = 0:Basis.K_max, for i = 0:Basis.I_max
        isL0((k*Basis.N_I + i)*Basis.N_Q + 1) = true;
    end, end
    R.aniso = max(max(abs(c(:, ~isL0))));
end

function rate = fit_rate(t, Tv, Ti)
    % Empirical asymptotic decay rate = -slope of log|Tv-Ti| over the clean
    % LATE-TIME exponential tail (skip the early nonlinear/multi-mode transient
    % and stay well above the machine-precision floor). This is the slow
    % Landau-Teller mode and is grid-independent.
    dT = abs(Tv - Ti);
    dT0 = dT(1);
    % Fit the dominant relaxation e-fold: below the initial fast/nonlinear
    % transient (dT < 0.75 dT0) and above the deep residual/floor (dT > 0.10 dT0).
    win = dT < 0.75*dT0 & dT > 0.10*dT0;
    if nnz(win) < 5, rate = NaN; return; end
    p = polyfit(t(win), log(dT(win)), 1);
    rate = -p(1);
end

function write_relaxation_table(fname, sweepA, sweepB, Teq_of, delta_B)
    % Two booktabs tabulars (Sweep A: T_eq vs d_i; Sweep B: rate vs omega) for the
    % paper. Matches the hand-written paper/table_hydrodynamic_quantities.tex style.
    fid = fopen(fname, 'w');
    if fid < 0, warning('write_relaxation_table: cannot open %s', fname); return; end
    fprintf(fid, '%% Auto-generated by benchmark_polyatomic_temperature_relaxation.m -- do not edit by hand.\n');
    fprintf(fid, '%% Requires \\usepackage{booktabs}.  Sweep B fixed d_i = %g.\n', delta_B);
    % Sweep A: equilibrium temperature vs internal DOF
    fprintf(fid, '\\begin{tabular}{cccc}\n\\toprule\n');
    fprintf(fid, '$d_i$ & $T_{\\mathrm{eq}}^{\\mathrm{pred}}$ & $T_{\\mathrm{eq}}^{\\mathrm{num}}$ & rate $r$ \\\\\n\\midrule\n');
    for a = 1:numel(sweepA)
        fprintf(fid, '%d & %.4f & %.4f & %.3f \\\\\n', ...
            sweepA(a).di, Teq_of(sweepA(a).di), sweepA(a).Teq_num, sweepA(a).rate);
    end
    fprintf(fid, '\\bottomrule\n\\end{tabular}\n\n');
    % Sweep B: decay rate vs omega
    fprintf(fid, '\\begin{tabular}{ccccc}\n\\toprule\n');
    fprintf(fid, '$\\omega$ & $T_{\\mathrm{eq}}$ & $r_{\\mathrm{fit}}$ & $r_{\\mathrm{eig}}$ & $r_{\\mathrm{fit}}/\\omega$ \\\\\n\\midrule\n');
    for b = 1:numel(sweepB)
        fprintf(fid, '%.2f & %.4f & %.4f & %.4f & %.4f \\\\\n', ...
            sweepB(b).omega, sweepB(b).Teq_num, sweepB(b).rate, sweepB(b).rate_ex, ...
            sweepB(b).rate/sweepB(b).omega);
    end
    fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
    fclose(fid);
end
