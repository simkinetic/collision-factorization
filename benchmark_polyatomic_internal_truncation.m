% PAPER ARTIFACT: the internal-truncation convergence study behind the closing
% paragraph of Section 5.5.4. Measures how the RESOLVED (Schur-reduced) bulk-to-
% shear ratio mu_b/mu depends on the internal truncation I_max, against the
% first-order (17-moment) coefficient, which reads only i = 0, 1 and is
% I_max-independent by construction.
%
% K_max = 3 and L_max = 2 are held fixed; I_max = 1, 2, 3, 4. I_max = 2 is the
% production truncation and is already cached; 3 and 4 are the expensive rungs
% (~20 min and ~50 min per gas) and are gated on a wall-clock deadline -- what
% is not affordable is reported as skipped, never silently omitted.
%
% zeta and the fitted (omega, eta_hat_f) pairs match the production drivers:
% zeta from Djordjic et al. (2023) Table 1, the pairs from the "refitted" rows
% of the paper's figures/fig_transport_fits_points.csv.
%
% Writes results/internal_truncation.{mat,csv} (results/ is untracked).
% Prints key lines tagged 'RB|'.
% =========================================================================
% For each gas at its current fitted (omega, eta_hat_f) pair we assemble the
% linearized operator
%     J(omega, eta_hat_f) = omega    * (Jnf0 + eta_hat   * Jnfe)
%                         + (1-omega)* (Jf0  + eta_hat_f * Jfe)
% (the same four production channels and the same linear combination the
% transport fits use), and evaluate mu_b/mu two ways:
%
%   RESOLVED   the full block inversion / Schur reduction of
%              benchmark_polyatomic_chapman_enskog.m's ce_sweep, at the top
%              radial order kk = K_max: the shear rate is the Schur
%              complement of the L=2 block onto its lead mode, and the bulk
%              rate is the slowest non-null eigenvalue of the L=0 block with
%              mass excluded. BOTH blocks contain every internal index
%              i = 0..I_max, which is exactly why the result moves with
%              I_max.
%   FIRST-ORDER the Djordjic 17-moment closure: the L=2 lead diagonal over
%              the 2x2 L=0 block. It touches only i = 0, 1, so it should be
%              I_max-independent to roundoff -- reported as the contrast.
% =========================================================================

REPO = fileparts(mfilename('fullpath'));
OUT  = fullfile(REPO,'results');
if ~exist(OUT,'dir'), mkdir(OUT); end
addpath(REPO, fullfile(REPO,'src'), fullfile(REPO,'src','mex'));
addpath(genpath(fullfile(REPO,'src','SHL')));

MATOUT = fullfile(OUT,'internal_truncation.mat');
CSVOUT = fullfile(OUT,'internal_truncation.csv');

K_MAX = 3;      % radial truncation, held fixed
L_MAX = 2;      % angular truncation, held fixed
IMAXS = [1 2 3 4];

BUDGET_HOURS = 8;   % wall-clock budget; no new rung is STARTED after it
DEADLINE = now + BUDGET_HOURS/24; %#ok<TNOW1>

% gas, zeta, delta, eta_hat, zeta_hat_f, fitted (omega, eta_hat_f)
G = { ...
 struct('gas','N2','zt',0.533,'dt',2.01,'eh',-0.3,  'zhf',0.3,  'w',0.31205166,'ehf',-0.20779265), ...
 struct('gas','CO','zt',0.53, 'dt',2.01,'eh',-0.453,'zhf',0.965,'w',0.54050581,'ehf', 0.57011043), ...
 struct('gas','H2','zt',0.607,'dt',1.94,'eh',-0.453,'zhf',0.965,'w',0.010118716,'ehf',-0.13387950) };

fprintf('\n=== I_max convergence of the RESOLVED mu_b/mu (Section 5.5.4) ===\n');
fprintf('RB| K_max=%d (fixed), L_max=%d (fixed), I_max sweep = [%s]\n', ...
        K_MAX, L_MAX, strjoin(compose('%d', IMAXS), ' '));
fprintf('RB| production convention: eq.(43)/etr43 spectral, pad(16,16) internal clamped, N_lambda=24, conservation ON\n');
fprintf('RB| deadline for starting a new rung: %s\n', datestr(DEADLINE,'yyyy-mm-dd HH:MM:SS')); %#ok<DATST>
for gi = 1:numel(G)
    fprintf('RB| %s: zeta=%g delta=%g eta_hat=%g zeta_hat_f=%g  fitted (omega, eta_hat_f) = (%.6f, %+.6f)\n', ...
            G{gi}.gas, G{gi}.zt, G{gi}.dt, G{gi}.eh, G{gi}.zhf, G{gi}.w, G{gi}.ehf);
end

B = struct('gas',{},'I_max',{},'K_max',{},'L_max',{}, ...
           'numu_resolved',{},'numu_first',{},'numu_kk',{}, ...
           'lam_shear',{},'P_Pi',{},'lam_shear_1',{},'P_Pi_1',{}, ...
           'nulls_L0',{},'dur_s',{},'status',{},'message',{});

est_per_gas = 220;    % seconds, seed estimate for the I_max = 1 rung
for ai = 1:numel(IMAXS)
    Im = IMAXS(ai);
    fprintf('\nRB| ================ I_max = %d ================\n', Im);
    fprintf('RB| estimated %.0f s per gas at this rung\n', est_per_gas);
    rung_durs = [];
    for gi = 1:numel(G)
        g = G{gi};
        b = struct('gas',g.gas,'I_max',Im,'K_max',K_MAX,'L_max',L_MAX, ...
                   'numu_resolved',NaN,'numu_first',NaN,'numu_kk',nan(1,K_MAX+1), ...
                   'lam_shear',NaN,'P_Pi',NaN,'lam_shear_1',NaN,'P_Pi_1',NaN, ...
                   'nulls_L0',NaN,'dur_s',NaN,'status','NOT RUN','message','');

        if (now + est_per_gas/86400) > DEADLINE %#ok<TNOW1>
            b.status  = 'SKIPPED';
            b.message = sprintf(['unaffordable: an estimated %.0f s does not fit before ' ...
                                 'the %s deadline'], est_per_gas, datestr(DEADLINE,'HH:MM')); %#ok<DATST>
            fprintf('RB| %s I_max=%d SKIPPED -- %s\n', g.gas, Im, b.message);
            B(end+1) = b; %#ok<SAGROW>
            continue;
        end

        tg = tic;
        try
            cfgP = struct('K_max',K_MAX,'L_max',L_MAX,'I_max',Im, ...
                          'zeta',g.zt,'delta',g.dt, ...
                          'rad_pad',16,'tan_pad',16,'int_pad',8, ...
                          'laplace_Ns',24,'conserve',true);
            cc = cfgP; cc.omega = 1;                        % non-frozen base
            Jnf0 = loadJ(cc);
            cc = cfgP; cc.omega = 1; cc.laplace = true;     % non-frozen modulated
            cc.eta_hat = 1; cc.zeta_hat = 0.965;
            JmN  = loadJ(cc);
            cc = cfgP; cc.omega = 0; cc.laplace = true;     % frozen base
            Jf0  = loadJ(cc);
            cc = cfgP; cc.omega = 0; cc.laplace = true;     % frozen modulated
            cc.eta_hat_f = 1; cc.zeta_hat_f = g.zhf;
            Jfm  = loadJ(cc);

            J = g.w*(Jnf0 + g.eh*(JmN - Jnf0)) + (1-g.w)*(Jf0 + g.ehf*(Jfm - Jf0));

            o = ce_numu(J, Im, L_MAX, K_MAX, g.dt);
            b.numu_resolved = o.numu_r;
            b.numu_first    = o.numu_1;
            b.numu_kk       = o.numu_kk;
            b.lam_shear     = o.lam_sh;
            b.P_Pi          = o.P_Pi;
            b.lam_shear_1   = o.lam_sh_1;
            b.P_Pi_1        = o.P_Pi_1;
            b.nulls_L0      = o.nulls_L0;
            b.status        = 'OK';

            fprintf('RB| %s I_max=%d : RESOLVED mu_b/mu = %.10f   (lam_shear %.6f, P_Pi %.6f, L=0 nulls %d)\n', ...
                    g.gas, Im, b.numu_resolved, b.lam_shear, b.P_Pi, b.nulls_L0);
            fprintf('RB| %s I_max=%d : first-order mu_b/mu = %.10f  (lam_shear %.6f, P_Pi %.6f)\n', ...
                    g.gas, Im, b.numu_first, b.lam_shear_1, b.P_Pi_1);
            fprintf('RB| %s I_max=%d : resolved over kk=0..%d: %s\n', g.gas, Im, K_MAX, ...
                    strjoin(compose('%.8f', b.numu_kk), '  '));
        catch err
            b.status  = 'ERROR';
            b.message = sprintf('%s: %s', err.identifier, err.message);
            fprintf(2, 'RB| %s I_max=%d FAILED: %s\n', g.gas, Im, err.message);
        end
        b.dur_s = toc(tg);
        fprintf('RB| %s I_max=%d elapsed %.1f s\n', g.gas, Im, b.dur_s);
        if strcmp(b.status,'OK'), rung_durs(end+1) = b.dur_s; end %#ok<SAGROW>
        B(end+1) = b; %#ok<SAGROW>
        write_out(B, MATOUT, CSVOUT);   % checkpoint after every build
    end
    % Cost grows steeply with I_max; project the next rung from this one.
    if ~isempty(rung_durs)
        est_per_gas = max(220, 3.5 * max(rung_durs));
    else
        est_per_gas = est_per_gas * 3.5;
    end
end

% --- per-gas convergence report -----------------------------------------
fprintf('\nRB| ===== CONVERGENCE OF THE RESOLVED mu_b/mu IN I_max =====\n');
for gi = 1:numel(G)
    gas = G{gi}.gas;
    sel = B(strcmp({B.gas}, gas) & strcmp({B.status}, 'OK'));
    if isempty(sel)
        fprintf('RB| %s: nothing completed.\n', gas);
        continue;
    end
    [ims, ord] = sort([sel.I_max]);  sel = sel(ord);
    vals = [sel.numu_resolved];
    firs = [sel.numu_first];
    fprintf('RB| %s  %-6s %-16s %-14s %-16s %-14s\n', gas, ...
            'I_max','resolved mu_b/mu','diff','first-order','diff');
    for k = 1:numel(ims)
        if k == 1
            fprintf('RB| %s  %-6d %-16.10f %-14s %-16.10f %-14s\n', ...
                    gas, ims(k), vals(k), '-', firs(k), '-');
        else
            fprintf('RB| %s  %-6d %-16.10f %-+14.3e %-16.10f %-+14.3e\n', ...
                    gas, ims(k), vals(k), vals(k)-vals(k-1), ...
                    firs(k), firs(k)-firs(k-1));
        end
    end
    % first-order must be I_max-independent by construction
    if numel(firs) > 1
        fo = max(abs(firs - firs(1)))/abs(firs(1));
        fprintf('RB| %s  first-order mu_b/mu spread over I_max: %.3e  (%s -- it is built from i<=1 only)\n', ...
                gas, fo, verdict_flat(fo));
    end
    % convergence verdict on the resolved value
    d = diff(vals);
    if numel(d) < 2
        fprintf('RB| %s  VERDICT: only %d rung(s) available -- cannot assess convergence.\n', gas, numel(vals));
    else
        rat = abs(d(2:end)) ./ max(abs(d(1:end-1)), realmin);
        fprintf('RB| %s  successive |diff|: %s ; ratios: %s\n', gas, ...
                strjoin(compose('%.3e', abs(d)), '  '), ...
                strjoin(compose('%.3f', rat), '  '));
        rl = rat(end);
        if rl < 0.5
            lim = vals(end) + d(end)*rl/(1-rl);   % geometric-tail extrapolation
            fprintf(['RB| %s  VERDICT: CONVERGING geometrically, last ratio %.3f ' ...
                     '(~%.1f digits per rung); extrapolated limit %.8f, ' ...
                     'remaining error at I_max=%d about %.2e.\n'], ...
                    gas, rl, -log10(rl), lim, ims(end), abs(lim - vals(end)));
        elseif rl < 0.95
            lim = vals(end) + d(end)*rl/(1-rl);
            fprintf(['RB| %s  VERDICT: converging SLOWLY, last ratio %.3f; ' ...
                     'extrapolated limit %.8f but the tail is long -- ' ...
                     'remaining error at I_max=%d about %.2e, so the ' ...
                     'quoted value is not yet converged.\n'], ...
                    gas, rl, lim, ims(end), abs(lim - vals(end)));
        else
            fprintf(['RB| %s  VERDICT: NOT CONVERGING over the rungs measured -- ' ...
                     'the successive differences are not decreasing (last ratio %.3f). ' ...
                     'The drift the paper reports is real and unresolved at I_max=%d.\n'], ...
                    gas, rl, ims(end));
        end
        fprintf('RB| %s  relative drift from I_max=%d to %d: %.3e\n', ...
                gas, ims(1), ims(end), abs(vals(end)-vals(1))/abs(vals(1)));
    end
end

skipped = B(strcmp({B.status},'SKIPPED'));
if ~isempty(skipped)
    fprintf('\nRB| UNAFFORDABLE (not run):\n');
    for k = 1:numel(skipped)
        fprintf('RB|   %s I_max=%d -- %s\n', skipped(k).gas, skipped(k).I_max, skipped(k).message);
    end
    fprintf('RB| The convergence statement above therefore covers only the rungs listed.\n');
end

write_out(B, MATOUT, CSVOUT);
fprintf('RB| results: %s\n', MATOUT);
fprintf('RB| results: %s\n', CSVOUT);
fprintf('RB| done\n');

% =========================================================================
% helpers
% =========================================================================
function J = loadJ(cfg)
    T = build_or_load_dsmc_tensor(cfg);
    Cf = T.assemble_full_tensor();
    J = squeeze(Cf(:,:,1)) + squeeze(Cf(:,1,:));
end

function o = ce_numu(J, I_max, L_max, kk, delta)
% Resolved and first-order bulk-to-shear ratio. Transcribed from ce_sweep /
% ce_anchor in benchmark_polyatomic_chapman_enskog.m.
    N_I = I_max + 1;  N_Q = (L_max + 1)^2;
    idx = @(k,i,l,m) (k*N_I+i)*N_Q + (l^2+l+m) + 1;
    pref = 2*delta/(3*(3+delta));
    o = struct('lam_sh_1',NaN,'P_Pi_1',NaN,'numu_1',NaN,'numu_kk',nan(1,kk+1), ...
               'lam_sh',NaN,'P_Pi',NaN,'nulls_L0',NaN,'numu_r',NaN);

    % ---- first-order (Djordjic 17-moment): lead L=2 diagonal, 2x2 L=0 ----
    o.lam_sh_1 = -J(idx(0,0,2,0), idx(0,0,2,0));
    bidx = [idx(1,0,0,0), idx(0,1,0,0)];
    evb  = sort(real(eig(J(bidx,bidx))), 'descend');
    o.P_Pi_1 = -evb(2);                       % 1st is the ~0 energy invariant
    o.numu_1 = pref * (o.lam_sh_1 / o.P_Pi_1);

    % ---- resolved, at every radial order up to kk ------------------------
    for a = 1:kk
        B2 = mk_block(idx, 2, [0 0], zeros(0,2), a, I_max);
        M2 = -J(B2,B2);  M2inv = inv(M2);
        lam = 1 / M2inv(1,1);
        B0 = mk_block(idx, 0, [1 0; 0 1], [0 0], a, I_max);
        M0 = -J(B0,B0);
        ev  = sort(real(eig(M0)), 'ascend');
        thr = 1e-8 * max(1, norm(M0,'fro'));
        pos = ev(ev > thr);
        if isempty(pos)
            error('night_study_resolved_bulk:NoBulkMode', ...
                  'L=0 block at kk=%d has no non-null eigenvalue.', a);
        end
        o.numu_kk(a+1) = pref * lam / min(pos);
        if a == kk
            o.lam_sh   = lam;
            o.P_Pi     = min(pos);
            o.nulls_L0 = sum(abs(ev) < thr);
            o.numu_r   = o.numu_kk(a+1);
        end
    end
end

function Bl = mk_block(idx, l, lead_ki, excl_ki, kk, Imax)
% Verbatim from benchmark_polyatomic_chapman_enskog.m.
    Bl = zeros(1, size(lead_ki,1));
    for r = 1:size(lead_ki,1)
        Bl(r) = idx(lead_ki(r,1), lead_ki(r,2), l, 0);
    end
    corr = zeros(1, (kk+1)*(Imax+1));
    nc = 0;
    for k = 0:kk
        for i = 0:Imax
            ki = [k i];
            if row_in(lead_ki, ki) || row_in(excl_ki, ki), continue; end
            nc = nc + 1;  corr(nc) = idx(k, i, l, 0);
        end
    end
    Bl = [Bl, corr(1:nc)];
end

function t = row_in(M, r)
    t = ~isempty(M) && any(all(M == r, 2));
end

function s = verdict_flat(e)
    if e < 1e-10
        s = 'flat to roundoff, as expected';
    elseif e < 1e-6
        s = 'essentially flat';
    else
        s = 'NOT flat -- unexpected, the first-order closure should not see I_max';
    end
end

function write_out(B, MATOUT, CSVOUT)
    save(MATOUT, 'B');
    fid = fopen(CSVOUT, 'w');
    if fid < 0, return; end
    fprintf(fid, ['gas,K_max,L_max,I_max,mu_b_over_mu_resolved,mu_b_over_mu_first_order,' ...
                  'lambda_shear_resolved,P_Pi_resolved,lambda_shear_first,P_Pi_first,' ...
                  'nulls_L0,duration_s,status,message\n']);
    for k = 1:numel(B)
        fprintf(fid, '%s,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%g,%.2f,%s,"%s"\n', ...
            B(k).gas, B(k).K_max, B(k).L_max, B(k).I_max, ...
            B(k).numu_resolved, B(k).numu_first, ...
            B(k).lam_shear, B(k).P_Pi, B(k).lam_shear_1, B(k).P_Pi_1, ...
            B(k).nulls_L0, B(k).dur_s, B(k).status, B(k).message);
    end
    fclose(fid);
end
