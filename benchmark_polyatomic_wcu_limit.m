% benchmark_polyatomic_wcu_limit.m
% =========================================================================
% Polyatomic -> monatomic Wang-Chang-Uhlenbeck (WCU) limit benchmark.
%
% The polyatomic spectral collision operator carries an internal-energy DOF
% d_i = 2(nu+1), nu = d_i/2 - 1. As d_i -> 0 the internal-energy equilibrium
% background ~ E^nu e^{-E} (mean <E> = nu+1 = d_i/2) collapses onto delta(E):
% no internal energy is available to exchange, and the gas must reduce to a
% monatomic Maxwell gas, whose linearized spectrum is the classical WCU
% spectrum.
%
% This benchmark demonstrates that limit. For a geometric sequence d_i -> 0
% we build the DSMC-Maxwell operator (zeta=0, omega=1, delta=d_i), extract the
% pure-translational (i=0) sub-block of J = dQ/dc, and track each PHYSICAL
% (k,l) mode as its shear-normalized eigenvalue converges onto the analytic
% monatomic WCU value lambda_WCU(k,l). The convergence IS the physics: at
% finite d_i a fraction of collisions leaks translational energy into internal
% modes (i>0), pulling the i=0 rates below the monatomic values; as d_i -> 0
% that leak channel closes, the i=0 block decouples, and each mode -> WCU.
%
% Per-mode extraction (vs. sorting the whole spectrum): the operator is
% rotationally invariant, so J is block-diagonal in the angular index l. We
% eig each l-block's m=0 slice separately -> a clean set {lambda(k,l): k} that
% we pair (by descending order) with {lambda_WCU(k,l): k}. Every plotted curve
% is then ONE physical mode with no cross-mode sorting artifacts.
%
% Notes:
%   * The operator is genuinely SINGULAR at exactly d_i=0 (nu=-1): the internal
%     measure E^{-1}e^{-E} is non-normalizable, so the internal Gauss quadrature
%     zeroth moment Gamma(nu+1)=Gamma(0)=Inf and the i=0 basis norm 1/Gamma(0)=0.
%     Hence d_i=0 is NOT a computed point -- it is anchored by the analytic
%     compute_wcu_spectrum. (The monatomic I_max=0 build is a different,
%     non-singular construction but carries a known energy-invariant regression,
%     mode 5 = -4/7; deliberately not used here.)
%   * DSMC K_delta -> 0 as delta -> 0, so raw rates vanish; we normalize by the
%     shear mode |lambda(k=0,l=2)|, making the comparison scale-free (ratios).
% =========================================================================

proj = fileparts(mfilename('fullpath'));
addpath(fullfile(proj,'src'));
addpath(genpath(fullfile(proj,'src','SHL')));
addpath(fullfile(proj,'src','mex'));

export_to_pdf_figure = true;

% ---- spectral resolution -------------------------------------------------
K_max = 2; L_max = 2; I_max = 2;
rad_pad = 16; tan_pad = 16; int_pad = 8;     % padding beyond exactness

% ---- d_i sweep (geometric, approaching the singular d_i=0 limit) ---------
di_list = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125];

N_Q = (L_max + 1)^2;
N_I = I_max + 1;
idx = @(k,i,l,m) (k*N_I + i)*N_Q + (l^2+l+m) + 1;    % global (k,i,l,m) index

%% ========================================================================
%  PHYSICAL (k,l) MODE LIST + ANALYTIC WCU TARGETS (the d_i=0 limit)
%  ========================================================================
% Within each angular block l, the k modes are paired to WCU by descending
% order, so position p in [1..K_max+1] <-> k = p-1 (WCU eigenvalues are
% monotone in k for fixed l, so the ordering is stable).
shear_wcu = abs(wcu_lambda(0, 2));           % L=2, k=0 shear normalizer

modes_kl = zeros((L_max+1)*(K_max+1), 2);    % rows [k l]
tgt      = zeros(size(modes_kl,1), 1);       % normalized WCU target per mode
mi = 0;
for l = 0:L_max
    kv = 0:K_max;
    wl = sort(arrayfun(@(k) wcu_lambda(k,l), kv), 'descend');  % descending -> k=p-1
    for p = 1:(K_max+1)
        mi = mi + 1;
        modes_kl(mi,:) = [p-1, l];
        tgt(mi)        = wl(p) / shear_wcu;
    end
end
n_modes = size(modes_kl,1);

%% ========================================================================
%  SWEEP d_i -> 0  (per-mode eigenvalue extraction)
%  ========================================================================
fprintf('\n=== Polyatomic -> WCU limit: DSMC-Maxwell (zeta=0, omega=1) ===\n');
fprintf('  K_max=%d L_max=%d I_max=%d | padding (%d,%d,%d)\n\n', ...
        K_max, L_max, I_max, rad_pad, tan_pad, int_pad);

n_d      = numel(di_list);
lam_mode = zeros(n_modes, n_d);          % normalized eigenvalue per (mode, d_i)
nulls    = zeros(n_d, 1);                % full-operator null-space count
shearR   = zeros(n_d, 1);               % raw shear rate (pre-normalization)

cols_l = cell(L_max+1,1);               % m=0, i=0 column set per l
for l = 0:L_max
    cols_l{l+1} = arrayfun(@(k) idx(k,0,l,0), 0:K_max);
end

fprintf('   d_i   |    nu    | #null |   shear_raw\n');
fprintf('  -------+----------+-------+--------------\n');
for d = 1:n_d
    d_i = di_list(d);
    nu  = d_i/2 - 1;

    Basis = SpectralBasis(K_max, L_max, I_max, nu);
    J = build_J(Basis, d_i, 0.0, 1.0, rad_pad, tan_pad, int_pad);

    nulls(d) = sum(abs(eig(J)) < 1e-9);           % conservation: expect 5

    % shear normalizer = least-negative eigenvalue of the l=2 block (mode (0,2))
    ev2   = sort(real(eig(J(cols_l{3}, cols_l{3}))), 'descend');
    shear = abs(ev2(1));
    shearR(d) = shear;

    % per-l block eigenvalues, paired to (k,l) by descending order
    for l = 0:L_max
        evl = sort(real(eig(J(cols_l{l+1}, cols_l{l+1}))), 'descend');
        for p = 1:(K_max+1)
            mrow = (modes_kl(:,1)==p-1) & (modes_kl(:,2)==l);
            lam_mode(mrow, d) = evl(p) / shear;
        end
    end

    fprintf('  %6.4f | %8.4f |   %d   | %12.4e\n', d_i, nu, nulls(d), shear);
end

err_mode = abs(lam_mode - tgt);                    % per-mode convergence error
conserved = max(abs(lam_mode), [], 2) < 1e-9;      % mass + momentum (identically 0)

fprintf('\n  Anchor (d_i=0): analytic monatomic WCU spectrum (shear-normalized).\n');
if all(nulls == 5)
    fprintf('  Conservation: #null == 5 at every d_i (PASS).\n');
else
    fprintf('  Conservation: WARNING -- #null != 5 for some d_i.\n');
end
% per-mode monotone convergence check (each dynamic mode)
mono = true;
for m = 1:n_modes
    if conserved(m), continue; end
    if ~issorted(err_mode(m,:), 'descend'), mono = false; end
end
fprintf('  Convergence: every dynamic mode monotone-decreasing to WCU: %s.\n', ...
        ternary(mono, 'PASS', 'inspect'));

% console: per-mode targets + finite-d_i deviation at the two extremes
fprintf('\n   (k,l) | WCU limit | err(d_i=%.4g) | err(d_i=%.4g)\n', di_list(1), di_list(end));
fprintf('  -------+-----------+---------------+---------------\n');
for m = 1:n_modes
    tag = '';
    if conserved(m),                          tag = ' mass/mom';
    elseif modes_kl(m,1)==1 && modes_kl(m,2)==0, tag = ' energy-leak';
    elseif modes_kl(m,1)==0 && modes_kl(m,2)==2, tag = ' shear';
    end
    fprintf('  (%d,%d)  | %9.4f | %13.3e | %13.3e%s\n', ...
        modes_kl(m,1), modes_kl(m,2), tgt(m), err_mode(m,1), err_mode(m,end), tag);
end

%% ========================================================================
%  FIGURE
%  ========================================================================
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
FS_labels = 14; FS_title = 15; FS_ticks = 12; FS_legend = 9.5;

% palette for the (up to 6) dynamic non-energy-leak modes
pal = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19; ...
       0.49 0.18 0.56; 0.30 0.75 0.93; 0.64 0.08 0.18];
mcolor = zeros(n_modes,3); ci = 0;
for m = 1:n_modes
    if conserved(m)
        mcolor(m,:) = [0.6 0.6 0.6];
    elseif modes_kl(m,1)==1 && modes_kl(m,2)==0
        mcolor(m,:) = [0.85 0.10 0.10];        % energy-leak: red
    else
        ci = ci + 1; mcolor(m,:) = pal(min(ci,end),:);
    end
end
mstyle = {'o','s','^','d','v','>','<','p','h'};   % marker per mode

xa = [min(di_list)*0.7, max(di_list)*1.15];       % x-range

outdir = 'results';
if ~exist(outdir,'dir'), mkdir(outdir); end

fig = figure('Position', [100 100 1450 480], 'Color', 'w');

% --- (a) each physical (k,l) mode -> its WCU value -----------------------
ax_a = axes('Position', [0.05 0.145 0.335 0.78]); hold on; box on;
% WCU target dashed lines = the analytic d_i=0 limit (one per distinct value)
utgt = unique(round(tgt,10));
for u = 1:numel(utgt)
    plot(xa, utgt(u)*[1 1], '--', 'Color', [0.35 0.35 0.35], ...
         'LineWidth', 0.9, 'HandleVisibility','off');
end
h = gobjects(0); lbl = {};
shown_conserved = false;
for m = 1:n_modes
    k = modes_kl(m,1); l = modes_kl(m,2);
    if conserved(m)
        hc = plot(di_list, lam_mode(m,:), '-', 'Color', mcolor(m,:), 'LineWidth', 1.2);
        if ~shown_conserved
            h(end+1) = hc; lbl{end+1} = 'mass, mom.\ (conserved) $\equiv 0$'; %#ok<*SAGROW>
            shown_conserved = true;
        else
            set(hc,'HandleVisibility','off');
        end
        continue;
    end
    hm = plot(di_list, lam_mode(m,:), ['-' mstyle{mod(m-1,numel(mstyle))+1}], ...
        'Color', mcolor(m,:), 'MarkerFaceColor', mcolor(m,:), ...
        'MarkerSize', 5, 'LineWidth', 1.6);
    if k==0 && l==2
        nm = sprintf('$(%d,%d)$ shear $\\to %.2f$', k, l, tgt(m));
    elseif k==1 && l==0
        nm = sprintf('$(%d,%d)$ energy-leak $\\to %.2f$', k, l, tgt(m));
    else
        nm = sprintf('$(%d,%d)\\to %.2f$', k, l, tgt(m));
    end
    h(end+1) = hm; lbl{end+1} = nm;
end
% proxy handle explaining the dashed lines (= analytic d_i=0 WCU limit)
hd = plot(nan, nan, '--', 'Color',[0.35 0.35 0.35], 'LineWidth',0.9);
h(end+1) = hd; lbl{end+1} = 'analytic WCU limit ($d_i{=}0$)';

set(gca, 'XScale','log', 'FontSize', FS_ticks, 'LineWidth', 1.1);
xlim(xa); ylim([-2 0.2]); yticks([-2 -1.5 -1 -0.5 0]);
xlabel('internal DOF $d_i$', 'FontSize', FS_labels);
ylabel('$\lambda_{(k,l)} / |\lambda_{\mathrm{shear}}|$', 'FontSize', FS_labels);
title('\textbf{(a) each $(k,l)$ mode $\to$ its WCU value}', 'FontSize', FS_title);
lg = legend(ax_a, h, lbl, 'FontSize', FS_legend, 'NumColumns', 1);
lg.Position = [0.39 0.28 0.15 0.46];

% --- (b) per-mode convergence error (a few representative modes) ----------
ax_b = axes('Position', [0.635 0.145 0.34 0.78]); hold on; box on;
sel = [find(modes_kl(:,1)==1 & modes_kl(:,2)==0), ...   % energy-leak (1,0)
       find(modes_kl(:,1)==2 & modes_kl(:,2)==0), ...   % (2,0)
       find(modes_kl(:,1)==1 & modes_kl(:,2)==1), ...   % (1,1)
       find(modes_kl(:,1)==1 & modes_kl(:,2)==2)];       % (1,2)
hb = gobjects(0); lb = {};
for s = sel
    hb(end+1) = plot(di_list, err_mode(s,:), ['-' mstyle{mod(s-1,numel(mstyle))+1}], ...
        'Color', mcolor(s,:), 'MarkerFaceColor', mcolor(s,:), ...
        'MarkerSize', 6, 'LineWidth', 1.8);
    lb{end+1} = sprintf('$(%d,%d)$', modes_kl(s,1), modes_kl(s,2));
end
% O(d_i) reference guide
xg = [min(di_list) max(di_list)];
Aref = err_mode(sel(1),end) / di_list(end);
plot(xg, Aref*xg, ':k', 'LineWidth', 1.2);
lb{end+1} = '$\propto d_i$ (guide)'; hb(end+1) = plot(nan,nan,':k','LineWidth',1.2);
set(gca, 'XScale','log', 'YScale','log', 'FontSize', FS_ticks, 'LineWidth', 1.1);
xlim([min(di_list)*0.8, max(di_list)*1.25]);
xlabel('internal DOF $d_i$', 'FontSize', FS_labels);
ylabel('$|\lambda_{(k,l)} - \lambda^{\mathrm{WCU}}_{(k,l)}|$', 'FontSize', FS_labels);
title('\textbf{(b) per-mode convergence to WCU}', 'FontSize', FS_title);
legend(hb, lb, 'Location','southeast', 'FontSize', FS_legend);

ax_a.Toolbar.Visible = 'off';  ax_b.Toolbar.Visible = 'off';   % clean export
if export_to_pdf_figure
    exportgraphics(fig, fullfile(outdir,'fig_wcu_limit.pdf'), 'ContentType', 'vector');
end

%% ========================================================================
%  SAVE RAW RESULTS (re-plotting + paper inclusion)
%  ========================================================================
params = struct('K_max',K_max,'L_max',L_max,'I_max',I_max, ...
    'rad_pad',rad_pad,'tan_pad',tan_pad,'int_pad',int_pad, ...
    'zeta',0.0,'omega',1.0,'di_list',di_list);
save(fullfile(outdir,'wcu_limit_results.mat'), ...
    'di_list','modes_kl','tgt','lam_mode','err_mode','conserved', ...
    'nulls','shearR','params','-v7');

% tidy long-form CSV: one row per (mode, d_i)
[MI, DI] = ndgrid(1:n_modes, 1:n_d);
Tb = table(modes_kl(MI(:),1), modes_kl(MI(:),2), di_list(DI(:))', ...
           tgt(MI(:)), lam_mode(sub2ind([n_modes n_d], MI(:), DI(:))), ...
           err_mode(sub2ind([n_modes n_d], MI(:), DI(:))), ...
    'VariableNames', {'k','l','d_i','wcu_limit','lambda','err'});
writetable(Tb, fullfile(outdir,'wcu_limit.csv'));

% LaTeX booktabs table for the paper: per-mode WCU limit + convergence error
% at the two sweep extremes (matches the console summary).
texfile = fullfile(outdir,'table_wcu_limit.tex');
write_wcu_table(texfile, modes_kl, tgt, err_mode, conserved, di_list);

fprintf('\nSaved: %s, %s, %s%s\n', fullfile(outdir,'wcu_limit_results.mat'), ...
        fullfile(outdir,'wcu_limit.csv'), texfile, ...
        ternary(export_to_pdf_figure, [', ' fullfile(outdir,'fig_wcu_limit.pdf')], ''));
fprintf('Done.\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function J = build_J(Basis, delta, zeta, omega, rp, tp, ip)
    % Linearized DSMC operator J = dQ/dc at the Maxwellian (mirror of
    % benchmark_dsmc_transport.m build_J).
    Kd = ScatteringKernel('DSMC', struct('zeta',zeta,'delta',delta,'omega',omega));
    T  = GeneralCollisionTensor(Basis, Kd);
    T.generate_R_tensor_sumfac(rp, tp, ip);
    C  = T.assemble_full_tensor();
    J  = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
end

function val = wcu_lambda(k, l)
    % Analytic monatomic Maxwell (WCU) eigenvalue for mode (k,l), isotropic
    % scattering B=1. Adapted from tests/TestWCUEigenvalues.compute_wcu_spectrum.
    qr = Gauss.legendre(200, -1, 1);
    mu = qr.x;  w = qr.w;
    c = sqrt((1 + mu)/2);  s = sqrt((1 - mu)/2);
    Pc = legendre(l, c'); Pc = Pc(1,:)';
    Ps = legendre(l, s'); Ps = Ps(1,:)';
    integrand = (c.^(2*k+l)).*Pc + (s.^(2*k+l)).*Ps - 1;
    val = 2*pi * sum(w .* integrand);
    % clamp the collision invariants (mass, momentum, energy) to exact 0
    if (k==0 && l==0) || (k==0 && l==1) || (k==1 && l==0)
        val = 0;
    end
end

function write_wcu_table(fname, modes_kl, tgt, err_mode, conserved, di_list)
    % Booktabs table: (k,l) | WCU limit | |Delta| at the two d_i sweep extremes.
    fid = fopen(fname, 'w');
    fprintf(fid, '%% Auto-generated by benchmark_polyatomic_wcu_limit.m -- do not edit.\n');
    fprintf(fid, '\\begin{tabular}{ccrr}\n\\toprule\n');
    fprintf(fid, '$(k,l)$ & $\\lambda^{\\mathrm{WCU}}$ & $|\\Delta|_{d_i=%.4g}$ & $|\\Delta|_{d_i=%.4g}$ \\\\\n', ...
            di_list(1), di_list(end));
    fprintf(fid, '\\midrule\n');
    for m = 1:size(modes_kl,1)
        note = '';
        if conserved(m),                              note = ' (mass/mom.)';
        elseif modes_kl(m,1)==1 && modes_kl(m,2)==0,  note = ' (energy)';
        elseif modes_kl(m,1)==0 && modes_kl(m,2)==2,  note = ' (shear)';
        end
        fprintf(fid, '$(%d,%d)$%s & $%.4f$ & $%.2e$ & $%.2e$ \\\\\n', ...
            modes_kl(m,1), modes_kl(m,2), note, tgt(m), err_mode(m,1), err_mode(m,end));
    end
    fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
    fclose(fid);
end

function s = ternary(c,a,b), if c, s=a; else, s=b; end, end
