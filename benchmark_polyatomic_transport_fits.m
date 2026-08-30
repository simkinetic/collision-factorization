% PAPER ARTIFACT: Table 4 and Figure 8 (fig_transport_fits) of the polyatomic
% collision-factorization paper -- the transport-coefficient (Pr, mu_b/mu) plane
% per gas, the reachable (omega, eta_hat_f) region, the Table-4 operating point
% and the re-fitted point.
%
% Production convention assumed by the cached operators this loads:
%   extended kernel eq. (43) e_tr^{zeta/2};  K_max = L_max = I_max = 2;
%   spatial padding (rad,tan) = (16,16);  internal-sector axes clamped
%   (exact at the Table-1 node counts on the spectral path);  auxiliary
%   Laplace node count N_lambda = 24;  conservation enforcement ON.
%
% Writes (all into the paper repo figures/ directory, `pf` below):
%   fig_transport_fits_data.mat
%   fig_transport_fits_points.csv
%   fig_transport_fits_{N2,CO,H2}_region_boundary.csv
%   fig_transport_fits_{N2,CO,H2}_omega_curve.csv
%   fig_transport_fits.pdf
% Reads fig_transport_fits_data.mat from the same directory first, to reuse the
% shipped (w, ehf) grids and to print old-vs-new deltas -- so it OVERWRITES its
% own input. Back that file up before re-running.
%
% Recovered from session transcript; modified only for repo paths:
%   - `repo` was the absolute scratchpad-era literal
%     '/Users/ekke/dev/matlab/collision-factorization'; now
%     fileparts(mfilename('fullpath')), which resolves to the same directory
%     now that the script lives at the repo root (matches the convention in
%     plot_wcu_limit_paper.m / plot_spatial_quadrature_convergence_paper.m).
% No other line was changed. `pf` is left as an absolute literal because every
% other plot_*_paper.m driver in this repo hard-codes the paper-repo path the
% same way.
% 2026-08-29: the operator fetch was routed through build_or_load_dsmc_tensor
% (config structs) instead of load() on hard-coded ..._p16-16-8_... filenames,
% so a cache miss self-builds. Same operators, same numerics; the loader's
% legacy read fallback still serves the shipped requested-padding filenames.
% benchmark_polyatomic_transport_fits.m -- regenerate fig_transport_fits data + CSVs + PDF from the
% CURRENT caches (_pfshift non-frozen, spectral frozen), reusing the exact
% (w, ehf) grids of the shipped artifact. Prints old-vs-new deltas.
repo = fileparts(mfilename('fullpath'));
addpath(repo, fullfile(repo,'src'), fullfile(repo,'src','mex')); addpath(genpath(fullfile(repo,'src','SHL')));
pf = paper_output_dir();

OLD = load(fullfile(pf,'fig_transport_fits_data.mat'));
ehf_grid = OLD.ehf_grid; w_grid = OLD.w_grid;

% zeta ('zt') is taken DIRECTLY from Djordjic et al. (2023), Table 1, which
% tabulates it per gas: 0.533 (N2), 0.53 (CO), 0.607 (H2). It is NOT recomputed
% here as 2*(1 - s_visc) from the tabulated viscosity exponent, which would give
% 0.534 / 0.530 / 0.608. The difference is in the third decimal only, but the
% calibration this driver is compared against (their Table 4) was produced at
% their tabulated zeta, so using it keeps the comparison like-for-like: at the
% derived zeta our re-fitted (omega, eta_hat_f) miss their published pair by
% ~1%, at their tabulated zeta they agree to ~1e-6. CO is unchanged (0.53).
gases = { ...
  struct('gas','N2','label','N$_2$','zt','0.533','dt','2.01','zhf','0.3',  'Pr',0.717,'nm',0.73, 'p4',[0.312052 -0.3 -0.207793 0.3]), ...
  struct('gas','CO','label','CO',   'zt','0.53', 'dt','2.01','zhf','0.965','Pr',0.743,'nm',0.55, 'p4',[0.540506 -0.453 0.570111 0.965]), ...
  struct('gas','H2','label','H$_2$','zt','0.607','dt','1.94','zhf','0.965','Pr',0.686,'nm',30.0, 'p4',[0.0101187 -0.453 -0.133879 0.965]) };

D = struct([]);
for gi = 1:numel(gases)
    g = gases{gi};
    % rho = d_q/d_s = sqrt(5/delta): l=1 heat-flux moment ratio, counterpart of
    % the sqrt(3/delta) that fixes the l=0 energy invariant.
    delta = str2double(g.dt); rho = sqrt(5/delta);
    om4 = g.p4(1); eh = g.p4(2); ehf4 = g.p4(3); zhf4 = g.p4(4);
    % Four channels per gas, routed through the self-building cache layer
    % (build_or_load_dsmc_tensor) instead of load() on hard-coded filenames, so
    % a missing cache rebuilds instead of erroring. Same production convention:
    % K=L=I=2, (rad,tan,int)=(16,16,8), N_lambda=24, conservation ON.
    % On the spectral path GeneralCollisionTensor clamps the internal padding to
    % zero, so the loader names new files ..._p16-16-0_...; the shipped
    % ..._p16-16-8_... caches are still served by its legacy read fallback.
    cfgP = struct('K_max',2,'L_max',2,'I_max',2, ...
                  'zeta',str2double(g.zt),'delta',delta, ...
                  'rad_pad',16,'tan_pad',16,'int_pad',8, ...
                  'laplace_Ns',24,'conserve',true);
    cc = cfgP; cc.omega = 1;                                       % non-frozen base
    Jnf0 = loadJ(cc);
    cc = cfgP; cc.omega = 1; cc.laplace = true;                    % non-frozen modulated (_pfshift)
    cc.eta_hat = 1; cc.zeta_hat = 0.965;
    JmN  = loadJ(cc);
    cc = cfgP; cc.omega = 0; cc.laplace = true;                    % spectral frozen base
    Jf0  = loadJ(cc);
    cc = cfgP; cc.omega = 0; cc.laplace = true;                    % spectral frozen modulated
    cc.eta_hat_f = 1; cc.zeta_hat_f = str2double(g.zhf);
    Jfm  = loadJ(cc);
    Jnfe = JmN - Jnf0;  Jfe = Jfm - Jf0;
    N_I = 3; N_Q = 9; idx = @(k,i,l,m) (k*N_I+i)*N_Q + (l^2+l+m) + 1;
    ii = [idx(0,0,2,0), idx(1,0,1,0), idx(0,1,1,0), idx(1,0,0,0), idx(0,1,0,0)];
    % element pack: rows = the 8 scalars needed, as linear coeffs
    pick = @(J) [ -J(ii(1),ii(1)); -J(ii(2),ii(2)); -J(ii(3),ii(3)); ...
                  -J(ii(2),ii(3)); -J(ii(3),ii(2)); ...
                   J(ii(4),ii(4)); J(ii(4),ii(5)); J(ii(5),ii(4)); J(ii(5),ii(5)) ];
    a0 = pick(Jnf0); a1 = pick(Jnfe); b0 = pick(Jf0); b1 = pick(Jfe);
    % maps over the (w, ehf) grid, orientation (iw, ie)
    [Wg, Eg] = ndgrid(w_grid, ehf_grid);
    PRm = zeros(size(Wg)); NMm = zeros(size(Wg));
    for q = 1:numel(Wg)
        e = Wg(q)*(a0 + eh*a1) + (1-Wg(q))*(b0 + Eg(q)*b1);
        [PRm(q), NMm(q)] = prnm(e, delta, rho);
    end
    % orientation match against the old artifact
    if max(abs(PRm(:) - OLD.D(gi).PR(:))) > max(abs(PRm(:) - reshape(OLD.D(gi).PR', [], 1)))
        PRm = PRm'; NMm = NMm';  % transpose to match stored orientation
    end
    fprintf('FIT| %s map delta: maxPR=%.3e maxNM(rel)=%.3e\n', g.gas, ...
        max(abs(PRm(:)-OLD.D(gi).PR(:))), max(abs(NMm(:)-OLD.D(gi).NM(:))./abs(OLD.D(gi).NM(:))));
    % boundary: pick the edge-trace assembly best matching the old artifact
    [bx, by] = best_boundary(PRm, NMm, OLD.D(gi).bx, OLD.D(gi).by);
    % curves on the old wc grid
    wc = OLD.D(gi).wc;
    [Pc, Mc]  = curve_at(wc, ehf4, a0, a1, b0, b1, eh, delta, rho);
    [pbx,pby] = curve_at(wc, -0.5, a0, a1, b0, b1, eh, delta, rho);
    % points
    e3 = om4*(a0+eh*a1) + (1-om4)*(b0+ehf4*b1);
    [P3, M3] = prnm(e3, delta, rho);
    p = [om4; ehf4];                                  % refit Newton
    obj = @(p) refit_resid(p, a0, a1, b0, b1, eh, delta, rho, g.Pr, g.nm);
    for it = 1:80
        r0 = obj(p); if norm(r0) < 1e-13, break; end
        h = 1e-7; Jc = zeros(2,2);
        for jj = 1:2, pp = p; pp(jj) = pp(jj)+h; Jc(:,jj) = (obj(pp)-r0)/h; end
        p = p - Jc\r0;
    end
    rres = norm(obj(p));
    ef = refit_eval(p, a0, a1, b0, b1, eh, delta, rho);
    % flags
    tgt_in = inpolygon(g.Pr, log10(g.nm), bx, log10(by));
    inb = p(2) >= -0.5;
    in_box = OLD.D(gi).in_box;   % carried: search-box semantics external to this fig
    D(gi).gas = g.gas; D(gi).label = OLD.D(gi).label; D(gi).delta = delta;
    D(gi).zeta = str2double(g.zt); D(gi).Pr_meas = g.Pr; D(gi).numu_meas = g.nm;
    D(gi).wc = wc; D(gi).Pc = Pc; D(gi).Mc = Mc;
    D(gi).bx = bx; D(gi).by = by; D(gi).pbx = pbx; D(gi).pby = pby;
    D(gi).PR = PRm; D(gi).NM = NMm; D(gi).encl = double(tgt_in); D(gi).tgt_in = tgt_in;
    D(gi).table4 = [P3, M3]; D(gi).refit = ef;
    D(gi).table4_params = g.p4; D(gi).refit_params = [p(1), eh, p(2), zhf4];
    D(gi).inb = inb; D(gi).in_box = in_box; D(gi).res = rres; D(gi).Ns = 24;
    % report
    fprintf('FIT| %s table4 old=(%.8f, %.8f) new=(%.8f, %.8f)\n', g.gas, ...
        OLD.D(gi).table4(1), OLD.D(gi).table4(2), P3, M3);
    fprintf('FIT| %s refit  old=(%.8f, %.8f) new=(%.8f, %.8f) resid=%.2e tgt_in=%d inb=%d\n', g.gas, ...
        OLD.D(gi).refit_params(1), OLD.D(gi).refit_params(3), p(1), p(2), rres, tgt_in, inb);
    fprintf('FIT| %s reachable Pr old=[%.6f %.6f] new=[%.6f %.6f]  numu new=[%.4f %.4f]\n', g.gas, ...
        min(OLD.D(gi).PR(:)), max(OLD.D(gi).PR(:)), min(PRm(:)), max(PRm(:)), min(NMm(:)), max(NMm(:)));
end

% ---- write artifacts ----------------------------------------------------
save(fullfile(pf,'fig_transport_fits_data.mat'), 'D', 'ehf_grid', 'w_grid');
fidp = fopen(fullfile(pf,'fig_transport_fits_points.csv'), 'w');
fprintf(fidp, 'gas,delta,gamma,point,Pr,mu_b_over_mu,omega,eta_hat,eta_hat_f,zeta_hat_f,inbounds,target_inside_region\n');
for gi = 1:numel(D)
    g = D(gi);
    fprintf(fidp, '%s,%s,%s,experiment,%.8g,%.8g,,,,,,%d\n', g.gas, gases{gi}.dt, gases{gi}.zt, g.Pr_meas, g.numu_meas, g.tgt_in);
    fprintf(fidp, '%s,%s,%s,table4_parameters,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,\n', g.gas, gases{gi}.dt, gases{gi}.zt, ...
        g.table4(1), g.table4(2), g.table4_params(1), g.table4_params(2), g.table4_params(3), g.table4_params(4), 1);
    fprintf(fidp, '%s,%s,%s,refitted,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,\n', g.gas, gases{gi}.dt, gases{gi}.zt, ...
        g.refit(1), g.refit(2), g.refit_params(1), g.refit_params(2), g.refit_params(3), g.refit_params(4), g.inb);
end
fclose(fidp);
for gi = 1:numel(D)
    g = D(gi);
    Tb = table(g.bx(:), g.by(:), 'VariableNames', {'Pr','mu_b_over_mu'});
    writetable(Tb, fullfile(pf, sprintf('fig_transport_fits_%s_region_boundary.csv', g.gas)));
    Tc = table(g.wc(:), g.Pc(:), g.Mc(:), 'VariableNames', {'omega','Pr','mu_b_over_mu'});
    writetable(Tc, fullfile(pf, sprintf('fig_transport_fits_%s_omega_curve.csv', g.gas)));
end

% ---- render -------------------------------------------------------------
set(groot,'defaultTextInterpreter','latex','defaultLegendInterpreter','latex', ...
    'defaultAxesTickLabelInterpreter','latex');
% NOTE: this script writes DATA ONLY. The paper's Figure 8 is rendered by
% plot_transport_fits_paper.m, which owns the marker scheme described in the
% figure caption (large open square = experiment, open circle = Djordjic
% Table-4 parameters, filled dot = this work). This script previously drew
% its own figure to the same path with a different marker scheme and silently
% overwrote the rendered one; that block has been removed. Run the renderer
% after this script to refresh the figure.
fprintf('FIT| artifacts written to %s\n', pf);
fprintf('FIT| done\n');

% ---- helpers ------------------------------------------------------------
function J = loadJ(cfg)
    T = build_or_load_dsmc_tensor(cfg);
    C = T.assemble_full_tensor();
    J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
end
function [Pr, nm] = prnm(e, delta, rho)
    % e = [Psig; Pq0; Ps0; -J(q,s); -J(s,q); B11; B12; B21; B22]
    Psig = e(1); Pq0 = e(2); Ps0 = e(3);
    Pq1 = -(-e(4))*rho;  % J(q,s) = -e(4); Pq1 = -J(q,s)*rho
    Ps1 = -(-e(5))/rho;
    Pr = (5+delta)/2 * 2*(Pq0*Ps0 - Pq1*Ps1) / (Psig*(5*(Ps0-Ps1) + delta*(Pq0-Pq1)));
    tr = e(6)+e(9); dsc = sqrt((e(6)-e(9))^2 + 4*e(7)*e(8));
    ev2 = (tr - dsc)/2;                    % smaller (more negative) eigenvalue
    P_Pi = -ev2;
    nm = (2*delta/(3*(3+delta))) * (Psig/P_Pi);
end
function [P, M] = curve_at(wc, ehf, a0, a1, b0, b1, eh, delta, rho)
    P = zeros(size(wc)); M = zeros(size(wc));
    for q = 1:numel(wc)
        e = wc(q)*(a0+eh*a1) + (1-wc(q))*(b0+ehf*b1);
        [P(q), M(q)] = prnm(e, delta, rho);
    end
end
function r = refit_resid(p, a0, a1, b0, b1, eh, delta, rho, Prm, nmm)
    v = refit_eval(p, a0, a1, b0, b1, eh, delta, rho);
    r = [v(1)-Prm; v(2)-nmm];
end
function v = refit_eval(p, a0, a1, b0, b1, eh, delta, rho)
    e = p(1)*(a0+eh*a1) + (1-p(1))*(b0+p(2)*b1);
    [P, M] = prnm(e, delta, rho); v = [P, M];
end
function [bx, by] = best_boundary(PRm, NMm, bx_old, by_old)
    n = size(PRm,1);
    segs = {@(M) [M(1,:), M(:,end).', fliplr(M(end,:)), flipud(M(:,1)).'], ...
            @(M) [M(:,1).', M(end,:), flipud(M(:,end)).', fliplr(M(1,:))], ...
            @(M) [fliplr(M(1,:)), M(:,1).', M(end,:), flipud(M(:,end)).'], ...
            @(M) [M(1,:), M(:,end).', fliplr(M(end,:)), flipud(M(:,1)).']};
    best = inf; bx = []; by = [];
    for si = 1:numel(segs)
        for tr = 0:1
            if tr, A = PRm'; B = NMm'; else, A = PRm; B = NMm; end
            cx = segs{si}(A); cy = segs{si}(B);
            if numel(cx) ~= numel(bx_old), continue; end
            d = max(abs(cx - bx_old));
            if d < best, best = d; bx = cx; by = cy; end
        end
    end
    fprintf('FIT|   boundary assembly matched, max|dPr|=%.3e\n', best);
end
