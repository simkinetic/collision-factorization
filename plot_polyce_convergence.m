function plot_polyce_convergence(R, outpdf)
% PLOT_POLYCE_CONVERGENCE  Render the polyatomic Chapman-Enskog convergence
% figure from saved results (decoupled from the expensive operator builds).
%
%   plot_polyce_convergence()            % load paper/data/polyce_results.mat,
%                                        % write paper/figures/fig_polyce_convergence.pdf
%   plot_polyce_convergence(R)           % R struct (see benchmark) or a .mat path
%   plot_polyce_convergence(R, outpdf)   % explicit output path
%
% R fields: kk, n2_f_mu, n2_numu, n2_Pr, hs_kk, hs_f_mu, anchor_f_mu/numu/Pr,
%           f_mu_pekeris, Pr_exp, numu_exp.

    if nargin < 1 || isempty(R)
        R = fullfile('paper','data','polyce_results.mat');
    end
    if ischar(R) || isstring(R)
        d = load(char(R));  R = d.R;
    end
    if nargin < 2 || isempty(outpdf)
        outpdf = fullfile('paper','figures','fig_polyce_convergence.pdf');
    end

    c_nu = [0 0.447 0.741];    % nu/mu (blue)
    c_pr = [0.85 0.325 0.098]; % Pr    (orange)
    c_hs = [0.466 0.674 0.188];% hard spheres (green)
    c_ho = [0.494 0.184 0.556];% increments accent (purple)
    dj   = [0.35 0.35 0.35];   % Djordjic reference (grey)

    fig = figure('Position',[100 100 1050 780],'Color','w');
    tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');

    % (a) Shear-viscosity correction f_mu: HS-frozen -> Pekeris (validation) + N2.
    nexttile; hold on; box on;
    plot(R.hs_kk, R.hs_f_mu, '-o','Color',c_hs,'LineWidth',1.4,'DisplayName','HS frozen (base)');
    plot(R.kk,    R.n2_f_mu, '-s','Color',c_nu,'LineWidth',1.4,'DisplayName','N2 (extended)');
    yline(R.f_mu_pekeris,'--k','DisplayName','Pekeris limit (HS)');
    yline(1.0,':k','DisplayName','1st-order ($f_\mu\equiv1$)');
    intticks(R.hs_kk);
    xlabel('$K_{\max}$','Interpreter','latex'); ylabel('$f_\mu$','Interpreter','latex');
    title('(a) Shear-viscosity correction','Interpreter','latex');
    legend('Location','east','Interpreter','latex'); grid on;

    % (b) Spectral convergence: successive increments |q(K)-q(K-1)|.
    nexttile; hold on; box on;
    [xf,df] = incr(R.kk, R.n2_f_mu);
    [xn,dn] = incr(R.kk, R.n2_numu);
    [xp,dp] = incr(R.kk, R.n2_Pr);
    plot(xf,df,'-o','Color',c_nu,'LineWidth',1.4,'DisplayName','$f_\mu$');
    plot(xn,dn,'-s','Color',c_ho,'LineWidth',1.4,'DisplayName','$\nu/\mu$');
    plot(xp,dp,'-^','Color',c_pr,'LineWidth',1.4,'DisplayName','$\mathrm{Pr}$');
    set(gca,'YScale','log'); intticks(R.kk);
    xlabel('$K_{\max}$','Interpreter','latex');
    ylabel('$|q(K_{\max})-q(K_{\max}\!-\!1)|$','Interpreter','latex');
    title('(b) Spectral convergence (increments)','Interpreter','latex');
    legend('Location','northeast','Interpreter','latex'); grid on;

    % (c) Prandtl number: our infinite-order convergence vs Djordjic vs 1st-order.
    nexttile; hold on; box on;
    yline(R.Pr_exp, '--','Color',dj,  'LineWidth',1.3,'DisplayName','Djordjic (= exp.)');
    yline(R.anchor_Pr,'-.','Color',c_pr,'LineWidth',1.3,'DisplayName','1st-order (prod. coeffs)');
    plot(R.kk, R.n2_Pr, '-^','Color',c_pr,'LineWidth',1.5,'MarkerFaceColor',c_pr, ...
         'DisplayName','ours (inf-order)');
    intticks(R.kk);
    xlabel('$K_{\max}$','Interpreter','latex'); ylabel('$\mathrm{Pr}$','Interpreter','latex');
    title('(c) Prandtl number (N2)','Interpreter','latex');
    legend('Location','east','Interpreter','latex'); grid on;

    % (d) Bulk/shear ratio nu/mu: same three references.
    nexttile; hold on; box on;
    yline(R.numu_exp, '--','Color',dj,  'LineWidth',1.3,'DisplayName','Djordjic (= exp.)');
    yline(R.anchor_numu,'-.','Color',c_nu,'LineWidth',1.3,'DisplayName','1st-order (prod. coeffs)');
    plot(R.kk, R.n2_numu, '-s','Color',c_nu,'LineWidth',1.5,'MarkerFaceColor',c_nu, ...
         'DisplayName','ours (inf-order)');
    intticks(R.kk);
    xlabel('$K_{\max}$','Interpreter','latex'); ylabel('$\nu/\mu$','Interpreter','latex');
    title('(d) Bulk/shear viscosity ratio (N2)','Interpreter','latex');
    legend('Location','east','Interpreter','latex'); grid on;

    if ~isempty(outpdf)
        odir = fileparts(outpdf);
        if ~isempty(odir) && ~isfolder(odir), mkdir(odir); end
        exportgraphics(fig, outpdf, 'ContentType','vector');
        fprintf('  figure written: %s\n', outpdf);
    end
end

function intticks(kk)
    set(gca,'XTick', min(kk):max(kk));   % integer K_max ticks only
end

function [xk, dq] = incr(kk, q)
    dq = abs(diff(q));  xk = kk(2:end);
    good = ~isnan(dq);  xk = xk(good);  dq = dq(good);
end
