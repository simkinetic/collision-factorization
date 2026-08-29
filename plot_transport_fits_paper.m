% PAPER ARTIFACT: Figure 8 (fig_transport_fits) -- RENDER-ONLY pass. Re-renders
% the PDF from the existing fig_transport_fits_data.mat with the final marker
% scheme: three concentric, nested markers per panel, drawn
%   1. experiment            -- LARGE OPEN SQUARE  (black edge, no fill)
%   2. Djordjic Table-4 pars -- OPEN CIRCLE        (black edge, no fill)
%   3. this work (refit)     -- small BLACK DOT    (filled)
% in that order, so all three stay individually visible where they coincide --
% which, after the heat-flux normalisation correction, they essentially do in
% every panel. This is the pass that produced the SHIPPED fig_transport_fits.pdf;
% run benchmark_polyatomic_transport_fits.m first only if the underlying data must be rebuilt.
% No data file is read or written other than the .mat it loads.
%
% Production convention assumed by the cached operators this loads:
%   extended kernel eq. (43) e_tr^{zeta/2};  K_max = L_max = I_max = 2;
%   spatial padding (rad,tan) = (16,16);  internal-sector axes clamped
%   (exact at the Table-1 node counts on the spectral path);  auxiliary
%   Laplace node count N_lambda = 24;  conservation enforcement ON.
%
% Writes:
%   <paper repo>/figures/fig_transport_fits.pdf
%   <repo root>/fig_transport_fits_marker_check.png   (200 dpi PRINT-SIZE check,
%       i.e. downscaled to \textwidth = 468.33 pt = 6.505 in -> 1301 px wide,
%       which is how the PDF actually appears via \includegraphics[width=\textwidth])
%
% Recovered from session transcript; modified only for repo paths:
%   - the 200-dpi verification raster was written to the session scratchpad
%     ('.../scratchpad/results/fits_marker_check.png'), which no longer exists;
%     it now goes to fig_transport_fits_marker_check.png at the repo root,
%     beside the other loose fig_*.pdf checks.
%   - `root = fileparts(mfilename('fullpath'));` added to support that path.
% No other line was changed.
% plot_transport_fits_paper.m -- render-only pass for fig_transport_fits.pdf.
% Loads the existing data .mat; data files untouched.
root = fileparts(mfilename('fullpath'));

% --- nested marker sizes, in points on THIS canvas (1481 pt wide). The figure
% is included at \textwidth, i.e. shrunk by 468.33/1481 = 0.3162, so a canvas
% point is only ~0.32 pt on the printed page: sizes must be generous here to
% leave visible white space there. MATLAB's square marker draws a side of
% ~0.84*MarkerSize, its circle a diameter of ~MarkerSize, so the radial
% clearances are
%     square -> circle : 0.42*MS_SQ - MS_CI/2 - LW_CI/2  ~ 6.2 pt  (2.0 pt on page)
%     circle -> dot    : MS_CI/2 - LW_CI/2 - MS_DOT/2    ~ 5.1 pt  (1.6 pt on page)
% i.e. ~5 px and ~4 px of clear white on the 200 dpi print-size check raster
% (measured: 5 and 4 px in all three panels).
MS_SQ  = 36;   % experiment      (open square, background)
MS_CI  = 16.5; % Table-4 params  (open circle, middle)
MS_DOT = 5;    % this work       (filled dot, foreground)
LW_SQ  = 1.6;  LW_CI = 1.3;
% legend proxies are drawn at a fixed fraction of the above so the legend rows
% stay compact; the nesting order they show is the same.
LEG_F  = 0.30;
pf = '/Users/ekke/dev/simkinetic/papers/polyatomic-collision-factorization/figures';
S = load(fullfile(pf,'fig_transport_fits_data.mat'));
D = S.D;

set(groot,'defaultTextInterpreter','latex','defaultLegendInterpreter','latex', ...
    'defaultAxesTickLabelInterpreter','latex');
fig = figure('Position',[50 50 1560 500],'Color','w');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
pan = {'(a)','(b)','(c)'};
for gi = 1:numel(D)
    g = D(gi); ax = nexttile; hold(ax,'on');
    hr = fill(g.bx, g.by, [0.80 0.88 0.97], 'EdgeColor', [0.55 0.7 0.9], 'FaceAlpha', 0.85);
    hp = plot(g.pbx, g.pby, 'r-', 'LineWidth', 2.2);
    % 1. experiment: LARGE OPEN SQUARE, drawn first (background)
    he = plot(g.Pr_meas, g.numu_meas, 's', 'MarkerSize', MS_SQ, ...
        'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'k', 'LineWidth', LW_SQ);
    % 2. Djordjic et al. Table-4 parameters: OPEN CIRCLE inside the square
    ht = plot(g.table4(1), g.table4(2), 'o', 'MarkerSize', MS_CI, ...
        'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'k', 'LineWidth', LW_CI);
    % 3. this work: small BLACK DOT, drawn last so it sits on top of both
    hf = plot(g.refit(1), g.refit(2), 'o', 'MarkerSize', MS_DOT, ...
        'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    set(ax,'YScale','log','FontSize',12,'LineWidth',1.05); grid(ax,'on');
    xlabel(ax,'Prandtl number $\mathrm{Pr}$','FontSize',14);
    if gi == 1, ylabel(ax,'bulk/shear ratio $\mu_b/\mu$','FontSize',14); end
    title(ax, sprintf('\\textbf{%s %s}\\quad $\\delta=%.2f$, $\\gamma=%.3f$', ...
        pan{gi}, g.label, g.delta, g.zeta), 'FontSize', 14);
    if gi == 1
        % scaled-down proxies, so the legend rows stay compact
        pe = plot(ax, nan, nan, 's', 'MarkerSize', LEG_F*MS_SQ, ...
            'MarkerFaceColor','none','MarkerEdgeColor','k','LineWidth',1.1);
        pt = plot(ax, nan, nan, 'o', 'MarkerSize', LEG_F*MS_CI, ...
            'MarkerFaceColor','none','MarkerEdgeColor','k','LineWidth',0.9);
        pd = plot(ax, nan, nan, 'o', 'MarkerSize', max(3,LEG_F*MS_DOT+1.5), ...
            'MarkerFaceColor','k','MarkerEdgeColor','k','LineWidth',0.5);
        legend([hr hp pe pt pd], {'reachable $(\omega,\hat\eta_f)$', ...
            'positivity bound $\hat\eta_f=-1/2$', ...
            'experimental value', ...
            'Djordji\''{c} et al. (2023), Table 4 parameters', ...
            'this work (two-parameter fit)'}, ...
            'Location','southwest', 'FontSize', 10);
    end
    xlim(ax, [floor(100*(min(g.bx)-0.001))/100, ceil(100*(max(g.bx)+0.001))/100]);
    if gi == 3
        ylim(ax, [1, max(g.by)*1.15]);
    else
        ylim(ax, [min(g.by)*0.95, 8]);
    end
end
exportgraphics(fig, fullfile(pf,'fig_transport_fits.pdf'), 'ContentType','vector');
% Verification raster at PRINT SIZE: render high-res, then scale to the width
% the figure actually occupies on the page (\textwidth = 468.33 pt = 6.505 in)
% at 200 dpi -> 1301 px. Judge marker clearance on THIS image, not on the
% oversized canvas.
tmpng = [tempname '.png'];
exportgraphics(fig, tmpng, 'Resolution', 600);
Ichk = imread(tmpng); delete(tmpng);
Ichk = imresize(Ichk, [NaN round(6.505*200)]);
imwrite(Ichk, fullfile(root,'fig_transport_fits_marker_check.png'));
fprintf('RND| wrote %s\n', fullfile(pf,'fig_transport_fits.pdf'));
fprintf('RND| print-size check raster %dx%d px -> %s\n', size(Ichk,2), size(Ichk,1), ...
    fullfile(root,'fig_transport_fits_marker_check.png'));
