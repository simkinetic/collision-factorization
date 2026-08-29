%% Temperature-relaxation sweep figures, paper-repo versions (d_i -> delta).
% Derived from Figures 1-2 of benchmark_polyatomic_temperature_relaxation.m
% (Bas Gieling); loads the regenerated results and re-renders with the paper
% notation (internal DOF symbol delta). Bas's script is unmodified.
% Outputs (paper repo figures/):
%   fig_polyrelax_sweep_di.pdf     sweep A: delta in {2,3,4}, Landau-Teller
%   fig_polyrelax_sweep_omega.pdf  sweep B: omega in {0.25,...,1}, slope ~ omega
clear; clc; close all;
root = fileparts(mfilename('fullpath'));
D = load(fullfile(root,'results','polyrelax_results.mat'));
sweepA = D.sweepA; sweepB = D.sweepB; P = D.params;
T_v0 = P.T_v0; T_I0 = P.T_I0; delta_B = P.delta_B;
Teq_of = @(di) (3*T_v0 + di*T_I0) / (3 + di);
Teq_str  = {'6/5', '7/6', '8/7'};
lamLT_str = {'5/7', '2/3', '7/11'};
paperdir = '/Users/ekke/dev/simkinetic/papers/polyatomic-collision-factorization/figures';

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
FS_labels = 15; FS_ticks = 12; FS_legend = 11; FS_title = 15;
colA = lines(numel(sweepA)); colB = lines(numel(sweepB));

% ---- Figure 1: Sweep A (internal DOF delta) -----------------------------
fig1 = figure('Name','Sweep A: internal DOF','Position',[80 80 1000 420],'Color','w');
t_max = 6 / min([sweepA.rate]);

subplot(1,2,1); hold on; grid on;
for a = 1:numel(sweepA)
    c = colA(a,:);
    w = sweepA(a).t <= t_max;
    plot(sweepA(a).t(w), sweepA(a).Tv(w), '-',  'Color', c, 'LineWidth', 2);
    plot(sweepA(a).t(w), sweepA(a).Ti(w), '--', 'Color', c, 'LineWidth', 2);
    yline(Teq_of(sweepA(a).di), ':', 'Color', c, 'LineWidth', 0.8);
end
xlabel('Time $t$','FontSize',FS_labels);
ylabel('Temperature','FontSize',FS_labels);
title('\textbf{(a) Relaxation to $T_{\mathrm{eq}}(\delta)$}','FontSize',FS_title);
set(gca,'FontSize',FS_ticks,'LineWidth',1.1);
xlim([0 t_max*1.18]);
for a = 1:numel(sweepA)
    text(t_max*1.02, Teq_of(sweepA(a).di), ['$', Teq_str{a}, '$'], ...
        'Color', colA(a,:), 'FontSize', FS_legend+1, 'FontWeight','bold', ...
        'HorizontalAlignment','left', 'VerticalAlignment','middle', 'Clipping','off');
end
text(0.84,0.06,'solid: $T_v$\quad dashed: $T_I$\quad dotted: $T_{\mathrm{eq}}$', ...
    'Units','normalized','HorizontalAlignment','right','FontSize',FS_legend, ...
    'BackgroundColor',[1 1 1 0.7],'EdgeColor','k','Margin',3);

subplot(1,2,2); hold on; grid on;
set(gca,'YScale','log');
h = gobjects(1,numel(sweepA)+1); lgA = cell(1,numel(sweepA)+1);
dT0 = T_v0 - T_I0;
for a = 1:numel(sweepA)
    dT = max(abs(sweepA(a).Tv - sweepA(a).Ti), eps);
    h(a) = plot(sweepA(a).t, dT, '-', 'Color', colA(a,:), 'LineWidth', 2.4);
    lgA{a} = sprintf('$\\delta=%d$ \\ ($\\lambda_{\\mathrm{LT}}=%s$)', sweepA(a).di, lamLT_str{a});
end
for a = 1:numel(sweepA)
    hLT = plot(sweepA(a).t, dT0*exp(-sweepA(a).rate_LT*sweepA(a).t), 'k--', 'LineWidth', 1.1);
end
h(end) = hLT; lgA{end} = 'Landau--Teller $\;\frac13 e^{-\lambda_{\mathrm{LT}}t}$';
set(gca,'FontSize',FS_ticks,'LineWidth',1.1);
xlim([0 t_max]); ylim([1e-4 1e0]);
xlabel('Time $t$','FontSize',FS_labels);
ylabel('$|T_v - T_I|$','FontSize',FS_labels);
title('\textbf{(b) Landau--Teller decay ($\nu_0=1$)}','FontSize',FS_title);
legend(h, lgA, 'Location','northeast','FontSize',FS_legend);
% Strip the interactive axes toolbar: exportgraphics bakes the hover icons into
% the vector PDF otherwise -- it was found overlapping panel (a)'s title in the
% shipped fig_polyrelax_sweep_di.pdf, and MATLAB only warns softly about it.
for axh = findall(fig1,'Type','axes')'
    try, axh.Toolbar = []; catch, end
end
set(findall(fig1,'Type','axestoolbar'),'Visible','off');
exportgraphics(fig1, fullfile(paperdir,'fig_polyrelax_sweep_di.pdf'), 'ContentType','vector');
fprintf('Wrote %s\n', fullfile(paperdir,'fig_polyrelax_sweep_di.pdf'));

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
xlim([0 0.8]);   % post-normalization rates are ~3.4x slower than Bas's original window assumed
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
xlim([0 0.8]); ylim([1e-10 1e0]);   % multi-decade decay for every omega
xlabel('Time $t$','FontSize',FS_labels);
ylabel('$|T_v - T_I|$','FontSize',FS_labels);
title('\textbf{(b) Slope $\propto \omega$}','FontSize',FS_title);
legend(h, lgB, 'Location','northeast','FontSize',FS_legend);
% Strip the interactive axes toolbar: exportgraphics bakes the hover icons into
% the vector PDF otherwise -- it was found overlapping panel (a)'s title in the
% shipped fig_polyrelax_sweep_di.pdf, and MATLAB only warns softly about it.
for axh = findall(fig2,'Type','axes')'
    try, axh.Toolbar = []; catch, end
end
set(findall(fig2,'Type','axestoolbar'),'Visible','off');
exportgraphics(fig2, fullfile(paperdir,'fig_polyrelax_sweep_omega.pdf'), 'ContentType','vector');
fprintf('Wrote %s\n', fullfile(paperdir,'fig_polyrelax_sweep_omega.pdf'));

% ---- Data CSVs beside the figures ---------------------------------------
for a = 1:numel(sweepA)
    src = fullfile(root,'results',sprintf('polyrelax_A_di%d.csv', sweepA(a).di));
    copyfile(src, fullfile(paperdir, sprintf('fig_polyrelax_sweep_di_data_delta%d.csv', sweepA(a).di)));
end
copyfile(fullfile(root,'results','polyrelax_B.csv'), ...
         fullfile(paperdir,'fig_polyrelax_sweep_omega_data.csv'));
fprintf('Data CSVs copied.\n');
