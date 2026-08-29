%% Conservation figure, paper-repo version (relabeled zeta -> gamma, d_i -> delta).
% Derived from Figure 3 of benchmark_polyatomic_temperature_relaxation.m (Bas
% Gieling); loads the saved results and re-renders with the paper's notation
% (VHS exponent gamma, internal DOF delta). Bas's script is unmodified.
% Output: paper repo figures/fig_polyrelax_conservation.pdf
clear; clc; close all;
root = fileparts(mfilename('fullpath'));
D = load(fullfile(root,'results','polyrelax_results.mat'));
Rcons = D.Rcons; Rcons_mx = D.Rcons_mx;
zeta = D.params.zeta; delta_B = D.params.delta_B;
paperdir = '/Users/ekke/dev/simkinetic/papers/polyatomic-collision-factorization/figures';

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
FS_labels = 15; FS_ticks = 12; FS_legend = 11; FS_title = 15;

fig3 = figure('Name','Conservation','Position',[160 160 700 440],'Color','w');
hold on; grid on;
sub = round(linspace(1, numel(Rcons.t), 12));
mass_err = max(abs(Rcons.mass    - Rcons.mass(1)),    eps);
E_err    = max(abs(Rcons.E       - Rcons.E(1)),       eps);
mass_mx  = max(abs(Rcons_mx.mass - Rcons_mx.mass(1)), eps);
E_mx     = max(abs(Rcons_mx.E    - Rcons_mx.E(1)),    eps);
semilogy(Rcons.t, mass_err, 'k-o', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName',sprintf('Mass, model ($\\gamma=%.2f$)', zeta));
semilogy(Rcons.t, E_err, 'k-s', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName',sprintf('Energy, model ($\\gamma=%.2f$)', zeta));
semilogy(Rcons_mx.t, mass_mx, 'r-o', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName','Mass, Maxwell ($\gamma=0$)');
semilogy(Rcons_mx.t, E_mx, 'r-s', 'LineWidth',2, 'MarkerSize',8, 'MarkerIndices',sub, ...
    'DisplayName','Energy, Maxwell ($\gamma=0$)');
set(gca,'YScale','log','FontSize',FS_ticks,'LineWidth',1.1);
% Axis tightened from [1e-18, 1e-8]: ten decades for two decades of data read as
% an accident and drew the eye to empty space. This range still sits well below
% the eps ~ 2.2e-16 mass floor and well above the ~7e-15 energy drift, so the
% energy curves are legible instead of compressed onto the bottom axis. The
% caption's "plotted at the 1e-16 floor" describes the floor VALUE (unchanged
% here, set by the max(...,eps) clamp above), not the axis range, so it still holds.
% Lower limit is deliberately OFF a decade (3e-17, not 1e-17): MATLAB does not
% draw a tick label that lands exactly on the log-axis lower limit, which left an
% empty unlabeled decade at the bottom looking like an unfinished axis. At 3e-17
% every tick from 1e-16 up is interior and labelled, and the eps ~ 2.2e-16 mass
% floor still clears the bottom by ~0.9 decade.
ylim([3e-17, 1e-12]); yticks(10.^(-16:1:-12));
xlabel('Time $t$','FontSize',FS_labels);
ylabel('Absolute drift','FontSize',FS_labels);
title(sprintf('\\textbf{Conservation ($\\delta=%g,\\ \\omega=1$): model vs Maxwell control}', delta_B),'FontSize',FS_title);
legend('Location','east','FontSize',FS_legend);
% Strip the interactive axes toolbar: exportgraphics otherwise bakes the hover
% icons into the vector PDF (this had already shipped in fig_polyrelax_sweep_di,
% where it truncated a panel title), and MATLAB only warns softly about it.
for axh = findall(fig3,'Type','axes')'
    try, axh.Toolbar = []; catch, end
end
set(findall(fig3,'Type','axestoolbar'),'Visible','off');
exportgraphics(fig3, fullfile(paperdir,'fig_polyrelax_conservation.pdf'), 'ContentType','vector');
fprintf('Wrote %s\n', fullfile(paperdir,'fig_polyrelax_conservation.pdf'));

% Ship the backing CSV alongside the figure, as plot_polyrelax_sweeps_paper.m
% already does for its two sweeps. Without this the paper repo kept a stale
% conservation CSV while its PDF was regenerated -- the drift digits do move
% between runs (the sweep-B time grid is shared, and the quadrature build is
% only reproducible to OpenMP reduction order), so the two must travel together.
copyfile(fullfile(root,'results','polyrelax_conservation.csv'), ...
         fullfile(paperdir,'fig_polyrelax_conservation_data.csv'));
fprintf('Wrote %s\n', fullfile(paperdir,'fig_polyrelax_conservation_data.csv'));
