%% WCU-limit figure, paper-repo version (relabeled d_i -> delta).
% Derived from the plotting section of benchmark_polyatomic_wcu_limit.m (Bas
% Gieling); loads the saved results and re-renders with the paper's notation
% (internal DOF symbol delta instead of d_i). Bas's script is unmodified.
% Output: paper repo figures/fig_wcu_limit.pdf
clear; clc; close all;
root = fileparts(mfilename('fullpath'));
D = load(fullfile(root,'results','wcu_limit_results.mat'));
di_list = D.di_list; lam_mode = D.lam_mode; err_mode = D.err_mode;
modes_kl = D.modes_kl; conserved = D.conserved; tgt = D.tgt;
n_modes = size(modes_kl,1);
paperdir = '/Users/ekke/dev/simkinetic/papers/polyatomic-collision-factorization/figures';

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
% Font sizes raised for print legibility: at the previous values this figure's
% text was ~60% the size, relative to panel width, of the other paper figures
% (14 pt over a 725 px panel here vs 16 pt over a 520 px panel elsewhere), and
% was the smallest type in the paper. The figure is widened to match so the
% panel-(a) legend still fits beside the axes at the larger size.
FS_labels = 17; FS_title = 17; FS_ticks = 14; FS_legend = 11.5;

pal = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19; ...
       0.49 0.18 0.56; 0.30 0.75 0.93; 0.64 0.08 0.18];
mcolor = zeros(n_modes,3); ci = 0;
for m = 1:n_modes
    if conserved(m)
        mcolor(m,:) = [0.6 0.6 0.6];
    elseif modes_kl(m,1)==1 && modes_kl(m,2)==0
        mcolor(m,:) = [0.85 0.10 0.10];
    else
        ci = ci + 1; mcolor(m,:) = pal(min(ci,end),:);
    end
end
mstyle = {'o','s','^','d','v','>','<','p','h'};
xa = [min(di_list)*0.7, max(di_list)*1.15];

fig = figure('Position', [100 100 1640 520], 'Color', 'w');

% --- (a) each physical (k,l) mode -> its WCU value -----------------------
ax_a = axes('Position', [0.052 0.155 0.315 0.76]); hold on; box on;
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
hd = plot(nan, nan, '--', 'Color',[0.35 0.35 0.35], 'LineWidth',0.9);
h(end+1) = hd; lbl{end+1} = 'analytic WCU limit ($\delta{=}0$)';

set(gca, 'XScale','log', 'FontSize', FS_ticks, 'LineWidth', 1.1);
xlim(xa); ylim([-2 0.2]); yticks([-2 -1.5 -1 -0.5 0]);
xlabel('internal DOF $\delta$', 'FontSize', FS_labels);
ylabel('$\lambda_{(k,l)} / |\lambda_{\mathrm{shear}}|$', 'FontSize', FS_labels);
title('\textbf{(a) each $(k,l)$ mode $\to$ its WCU value}', 'FontSize', FS_title);
lg = legend(ax_a, h, lbl, 'FontSize', FS_legend, 'NumColumns', 1);
lg.Position = [0.376 0.26 0.18 0.50];

% --- (b) per-mode convergence error --------------------------------------
ax_b = axes('Position', [0.645 0.155 0.325 0.76]); hold on; box on;
sel = [find(modes_kl(:,1)==1 & modes_kl(:,2)==0), ...
       find(modes_kl(:,1)==2 & modes_kl(:,2)==0), ...
       find(modes_kl(:,1)==1 & modes_kl(:,2)==1), ...
       find(modes_kl(:,1)==1 & modes_kl(:,2)==2)];
hb = gobjects(0); lb = {};
for s = sel
    hb(end+1) = plot(di_list, err_mode(s,:), ['-' mstyle{mod(s-1,numel(mstyle))+1}], ...
        'Color', mcolor(s,:), 'MarkerFaceColor', mcolor(s,:), ...
        'MarkerSize', 6, 'LineWidth', 1.8);
    lb{end+1} = sprintf('$(%d,%d)$', modes_kl(s,1), modes_kl(s,2));
end
xg = [min(di_list) max(di_list)];
Aref = err_mode(sel(1),end) / di_list(end);
plot(xg, Aref*xg, ':k', 'LineWidth', 1.2);
lb{end+1} = '$\propto \delta$ (guide)'; hb(end+1) = plot(nan,nan,':k','LineWidth',1.2);
set(gca, 'XScale','log', 'YScale','log', 'FontSize', FS_ticks, 'LineWidth', 1.1);
xlim([min(di_list)*0.8, max(di_list)*1.25]);
xlabel('internal DOF $\delta$', 'FontSize', FS_labels);
ylabel('$|\lambda_{(k,l)} - \lambda^{\mathrm{WCU}}_{(k,l)}|$', 'FontSize', FS_labels);
title('\textbf{(b) per-mode convergence to WCU}', 'FontSize', FS_title);
legend(hb, lb, 'Location','southeast', 'FontSize', FS_legend);

ax_a.Toolbar.Visible = 'off';  ax_b.Toolbar.Visible = 'off';
% Strip the interactive axes toolbar before export: exportgraphics otherwise
% bakes the hover icons into the vector PDF (it was found sitting on top of a
% panel title in fig_polyrelax_sweep_di.pdf), and MATLAB only warns softly.
for axh = findall(fig,'Type','axes')'
    try, axh.Toolbar = []; catch, end
end
set(findall(fig,'Type','axestoolbar'),'Visible','off');
exportgraphics(fig, fullfile(paperdir,'fig_wcu_limit.pdf'), 'ContentType', 'vector');
fprintf('Wrote %s\n', fullfile(paperdir,'fig_wcu_limit.pdf'));
