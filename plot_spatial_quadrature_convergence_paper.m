%% Polyatomic spatial quadrature convergence -- publication figure (paper Sec. 5.1)
% Spectral convergence of the 9D polyatomic singular quadrature for the DSMC
% Borgnakke-Larsen kernel: relative L_inf error of the dense physical tensor R
% versus the SPATIAL (radial/tangential Duffy) padding, against a pad-32
% reference, with the internal padding held fixed and identical everywhere
% (the internal quadrature error is common-mode and cancels in the difference).
%
% Configuration: K_max = L_max = I_max = 2, delta = 2.01, omega = 1,
% internal pad = 4, reference spatial pad = 32, conservation enforcement off.
% Curves: Maxwell (zeta = 0), hard sphere (zeta = 1), N2 fit (zeta = 0.533).
%
% Data: spatial_quadrature_convergence_data.mat (same directory), produced by
% benchmark_polyatomic_spatial_quadrature_convergence.m. This script only renders.
% Styled after benchmark_quadrature_error.m (the monatomic Figure 2).
clear; clc; close all;

root = fileparts(mfilename('fullpath'));
D = load(fullfile(paper_output_dir(), 'spatial_quadrature_convergence_data.mat'));
paperdir = paper_output_dir();

FS_labels = 28; FS_ticks = 18; FS_legend = 18;
c_maxwell = [0.8500, 0.3250, 0.0980];   % orange-red (matches monatomic Maxwell)
c_hs      = [0.0000, 0.4470, 0.7410];   % blue       (matches monatomic HS)
c_n2      = [0.4660, 0.6740, 0.1880];   % green      (third curve)

fig = figure('Name', 'Polyatomic Spatial Quadrature Convergence', ...
             'Position', [100, 100, 750, 650], 'Color', 'w');
hold on;

h_mx = plot(D.paddings, D.err_maxwell, 's--', 'Color', c_maxwell, ...
    'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_maxwell);
h_hs = plot(D.paddings, D.err_hardsphere, 'o-', 'Color', c_hs, ...
    'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_hs);
h_n2 = plot(D.paddings, D.err_n2, 'd-.', 'Color', c_n2, ...
    'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_n2);

set(gca, 'YScale', 'log', 'XScale', 'linear');
xticks([0 4 8 12 16 20 24]);
xlim([-1, max(D.paddings) + 1]);
ylim([1e-15, 1e-1]); yticks(10.^(-14:2:-2));

grid on;
set(gca, 'YMinorGrid', 'off', 'GridLineStyle', '-', 'GridAlpha', 0.2);

xlabel('\textbf{Spatial Quadrature Padding} ($N_{\mathrm{pad}}$)', ...
    'Interpreter', 'latex', 'FontSize', FS_labels);
ylabel('\textbf{Relative $\ell_\infty$ Error} ($\epsilon$)', ...
    'Interpreter', 'latex', 'FontSize', FS_labels);
legend([h_mx, h_hs, h_n2], ...
    {'\textbf{Maxwell} ($\gamma = 0$)', ...
     '\textbf{Hard sphere} ($\gamma = 1$)', ...
     '\textbf{N$_2$ fit} ($\gamma = 0.533$)'}, ...
    'Interpreter', 'latex', 'FontSize', FS_legend, 'Location', 'northeast');

set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex');

set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
exportgraphics(gcf, fullfile(paperdir, 'fig_quadrature_convergence.pdf'), ...
    'ContentType', 'vector');

% CSV alongside the figure for the paper repo
Tbl = table(D.paddings(:), D.err_maxwell(:), D.err_hardsphere(:), D.err_n2(:), ...
    'VariableNames', {'spatial_pad', 'err_maxwell_z0', 'err_hardsphere_z1', 'err_n2_z0533'});
writetable(Tbl, fullfile(paperdir, 'fig_quadrature_convergence_data.csv'));

fprintf('Wrote %s\n', fullfile(paperdir, 'fig_quadrature_convergence.pdf'));
fprintf('Wrote %s\n', fullfile(paperdir, 'fig_quadrature_convergence_data.csv'));
