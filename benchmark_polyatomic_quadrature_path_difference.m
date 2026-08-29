% PAPER ARTIFACT: the spectral-vs-algebraic quadrature-path comparison quoted
% alongside Table 4 (no figure, no file output -- stdout only). Per gas, at the
% Table-4 parameters, it forms the transport quantities on the PRODUCTION
% SPECTRAL path (auxiliary Laplace, internal axes at the exactness-bound counts,
% N_lambda = 24) and on the LEGACY ALGEBRAIC path (pointwise modulation,
% internal padding 8), and reports the relative path difference; plus the H2
% refit-amplitude shift between the two paths.
%
% Production convention assumed by the cached operators this loads:
%   extended kernel eq. (43) e_tr^{zeta/2};  K_max = L_max = I_max = 2;
%   spatial padding (rad,tan) = (16,16);  internal-sector axes clamped
%   (exact at the Table-1 node counts on the spectral path);  auxiliary
%   Laplace node count N_lambda = 24;  conservation enforcement ON.
%
% Note: ALL operators (spectral and algebraic) are fetched through
% build_or_load_dsmc_tensor; all of them are present in src/precalc, so this
% normally runs as cache hits and builds nothing -- but a missing one now
% rebuilds instead of erroring.
%
% Writes: nothing. Prints 'PD| ...' lines.
%
% Recovered from session transcript; modified only for repo paths:
%   - `repo` was the absolute scratchpad-era literal
%     '/Users/ekke/dev/matlab/collision-factorization'; now
%     fileparts(mfilename('fullpath')), same directory, self-locating.
% 2026-08-29: the operator fetch was routed through build_or_load_dsmc_tensor
% (config structs) instead of load() on hard-coded ..._p16-16-8_... filenames,
% so a cache miss self-builds. Same operators, same numerics; the loader's
% legacy read fallback still serves the shipped requested-padding filenames.
% No other line was changed.
% benchmark_polyatomic_quadrature_path_difference.m -- per-gas quadrature-path difference of the transport quantities
% at Table-4 parameters: PRODUCTION SPECTRAL path (laplace aux, internal axes
% at exactness-bound counts, N_lambda=24) vs LEGACY ALGEBRAIC path (pointwise
% modulation, internal padding 8 -- its convergence knob) at (16,16) spatial.
% Also: the H2 refit-amplitude shift between the two paths.
repo = fileparts(mfilename('fullpath'));
addpath(repo, fullfile(repo,'src'), fullfile(repo,'src','mex')); addpath(genpath(fullfile(repo,'src','SHL')));

G = { ...
 {'N2','0.534','2.01','0.3',  0.717,0.73, [0.312052,-0.3,-0.207793]}, ...
 {'CO','0.53', '2.01','0.965',0.743,0.55, [0.540506,-0.453,0.570111]}, ...
 {'H2','0.608','1.94','0.965',0.686,30.0, [0.0101187,-0.453,-0.133879]} };

mxPr = 0; mxNM = 0;
for gi = 1:numel(G)
    g = G{gi}; nm=g{1}; zt=g{2}; dt=g{3}; zhf=g{4};
    % rho = d_q/d_s = sqrt(5/delta): l=1 heat-flux moment ratio, counterpart of
    % the sqrt(3/delta) that fixes the l=0 energy invariant.
    delta=str2double(dt); nu=delta/2-1; rho=sqrt(5/delta);
    Prx=g{5}; nmx=g{6}; p4=g{7};
    % Every operator now goes through the self-building cache layer
    % (build_or_load_dsmc_tensor) instead of load() on hard-coded filenames, so
    % a missing cache rebuilds instead of erroring. Same production convention:
    % K=L=I=2, (rad,tan,int)=(16,16,8), N_lambda=24, conservation ON.
    % On the spectral path GeneralCollisionTensor clamps the internal padding to
    % zero, so the loader names new files ..._p16-16-0_...; the shipped
    % ..._p16-16-8_... caches are still served by its legacy read fallback.
    cfgP = struct('K_max',2,'L_max',2,'I_max',2, ...
                  'zeta',str2double(zt),'delta',delta, ...
                  'rad_pad',16,'tan_pad',16,'int_pad',8, ...
                  'laplace_Ns',24,'conserve',true);
    % spectral (production) operator set
    cc = cfgP; cc.omega = 1;                                       % non-frozen base
    Jnf0 = loadJ(cc);
    cc = cfgP; cc.omega = 1; cc.laplace = true;                    % non-frozen modulated (_pfshift)
    cc.eta_hat = 1; cc.zeta_hat = 0.965;
    JmS  = loadJ(cc);
    cc = cfgP; cc.omega = 0; cc.laplace = true;                    % spectral frozen base
    Jf0S = loadJ(cc);
    cc = cfgP; cc.omega = 0; cc.laplace = true;                    % spectral frozen modulated
    cc.eta_hat_f = 1; cc.zeta_hat_f = str2double(zhf);
    JfmS = loadJ(cc);
    % algebraic (legacy) operator set -- pointwise modulation, internal padding 8
    cfgA = cfgP; cfgA.laplace = false;
    c1 = cfgA; c1.omega = 1; c1.eta_hat = 1; c1.zeta_hat = 0.965;
    JmA  = loadJ(c1);
    c2 = cfgA; c2.omega = 0; c2.eta_hat_f = 1; c2.zeta_hat_f = str2double(zhf);
    JfmA = loadJ(c2);
    c3 = cfgA; c3.omega = 0;                                       % algebraic frozen base
    Jf0A = loadJ(c3);
    % blends at Table-4
    JS = p4(1)*(Jnf0 + p4(2)*(JmS - Jnf0)) + (1-p4(1))*(Jf0S + p4(3)*(JfmS - Jf0S));
    JA = p4(1)*(Jnf0 + p4(2)*(JmA - Jnf0)) + (1-p4(1))*(Jf0A + p4(3)*(JfmA - Jf0A));
    [PrS, nmS] = trans(JS, delta, rho);
    [PrA, nmA] = trans(JA, delta, rho);
    dPr = abs(PrA-PrS)/abs(PrS); dNM = abs(nmA-nmS)/abs(nmS);
    mxPr = max(mxPr, dPr); mxNM = max(mxNM, dNM);
    fprintf('PD| %s Pr: spec=%.8f alg=%.8f relpathdiff=%.3e | numu: spec=%.8f alg=%.8f relpathdiff=%.3e\n', ...
        nm, PrS, PrA, dPr, nmS, nmA, dNM);
    if strcmp(nm,'H2')
        pS = refit(Jnf0, JmS-Jnf0, Jf0S, JfmS-Jf0S, p4, delta, rho, Prx, nmx);
        pA = refit(Jnf0, JmA-Jnf0, Jf0A, JfmA-Jf0A, p4, delta, rho, Prx, nmx);
        fprintf('PD| H2 refit (omega, ehf): spec=(%.8f, %.8f) alg=(%.8f, %.8f)  |d ehf|=%.3e (rel %.3e)\n', ...
            pS(1), pS(2), pA(1), pA(2), abs(pA(2)-pS(2)), abs(pA(2)-pS(2))/abs(pS(2)));
    end
end
fprintf('PD| MAXIMA across gases: Pr %.3e   numu %.3e\n', mxPr, mxNM);
fprintf('PD| done\n');

function J = loadJ(cfg)
    T = build_or_load_dsmc_tensor(cfg); J = Jof(T);
end
function J = Jof(T)
    C = T.assemble_full_tensor(); J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
end
function [Pr, nmr] = trans(J, delta, rho)
    N_I=3; N_Q=9; idx=@(k,i,l,m)(k*N_I+i)*N_Q+(l^2+l+m)+1;
    Psig = -J(idx(0,0,2,0), idx(0,0,2,0));
    iq=idx(1,0,1,0); is=idx(0,1,1,0); B=J([iq is],[iq is]);
    Pq0=-B(1,1); Ps0=-B(2,2); Pq1=-B(1,2)*rho; Ps1=-B(2,1)/rho;
    Pr = (5+delta)/2 * 2*(Pq0*Ps0-Pq1*Ps1) / (Psig*(5*(Ps0-Ps1)+delta*(Pq0-Pq1)));
    bidx=[idx(1,0,0,0), idx(0,1,0,0)];
    ev=sort(real(eig(J(bidx,bidx))),'descend');
    nmr = 2*delta/(3*(3+delta)) * Psig/(-ev(2));
end
function p = refit(Jnf0, Jnfe, Jf0, Jfe, p4, delta, rho, Prm, nmm)
    obj = @(p) resid(p, Jnf0, Jnfe, Jf0, Jfe, p4(2), delta, rho, Prm, nmm);
    p = [p4(1); p4(3)];
    for it = 1:80
        r0 = obj(p); if norm(r0) < 1e-13, break; end
        h=1e-7; Jc=zeros(2,2);
        for jj=1:2, pp=p; pp(jj)=pp(jj)+h; Jc(:,jj)=(obj(pp)-r0)/h; end
        p = p - Jc\r0;
    end
end
function r = resid(p, Jnf0, Jnfe, Jf0, Jfe, eh, delta, rho, Prm, nmm)
    J = p(1)*(Jnf0 + eh*Jnfe) + (1-p(1))*(Jf0 + p(2)*Jfe);
    [Pr, nmr] = trans(J, delta, rho);
    r = [Pr-Prm; nmr-nmm];
end
