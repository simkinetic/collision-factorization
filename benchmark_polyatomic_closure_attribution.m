% PAPER ARTIFACT: the closure cross-check behind Table 4 (no figure, no file
% output -- stdout only). Evaluates Djordjic's own 17-moment closure, eqs (39)
% [Pr] and (58) [nu/mu] of s00161-022-01167-8, using the production rates OUR
% final operator supplies, at the Table-4 parameters and at the E4 re-fit.
% Reports P_Pi under both readings (block eigenvalue vs diagonal Grad
% projection) so the closure-attribution ambiguity is visible.
%
% Production convention assumed by the cached operators this loads:
%   extended kernel eq. (43) e_tr^{zeta/2};  K_max = L_max = I_max = 2;
%   spatial padding (rad,tan) = (16,16);  internal-sector axes clamped
%   (exact at the Table-1 node counts on the spectral path);  auxiliary
%   Laplace node count N_lambda = 24;  conservation enforcement ON.
%
% Writes: nothing. Prints 'CT| ...' lines.
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
% benchmark_polyatomic_closure_attribution.m -- closure-attribution test (Rene): evaluate Djordjic's own
% 17-moment closure, eqs (39) [Pr] and (58) [nu/mu], with the production
% rates OUR final-production operator supplies, at his Table-4 parameters.
%
% Definitions implemented (equation numbers from s00161-022-01167-8.pdf):
%   (35) linearized production terms: P = -P_Pi0*Pi, P_<ij> = -P_sig0*sig_ij,
%        Q_i = -(P_q0 q_i + P_q1 s_i), S_i = -(P_s0 s_i + P_s1 q_i)
%   (39) Pr = (5+d)/2 * 2(Pq0*Ps0 - Pq1*Ps1) / (Psig*(5(Ps0-Ps1)+d(Pq0-Pq1)))
%   (58) nu/mu = 2d/(3(3+d)) * Psig/P_Pi0
% Production rates from our operator: 2x2 (q,s) block elements with the
% d_q/d_s normalization rho = sqrt(5/d), the l=1 counterpart of the sqrt(3/d)
% that fixes the l=0 energy invariant (validated: frozen zeta=0 Pr
% reproduces Eucken (38) to 1e-14); P_sig = diagonal l=2 element (the sigma
% moment IS a basis function). P_Pi0 in TWO readings:
%   eig : nonzero eigenvalue of the (psi_100, psi_010) block  [banked Sec 5.5]
%   diag: -<w|J|w>, w = unit vector of the dynamic-pressure moment
%         Pi ~ delta*|v|^2 - 3*I  =>  w ~ (delta*sqrt(3/2), -3*sqrt(nu+1)),
%         orthogonal to the energy invariant (a,b)=(sqrt(3/2),sqrt(nu+1))
%         [the literal Maxwellian-iteration / Grad-projection coefficient]
repo = fileparts(mfilename('fullpath'));
addpath(repo, fullfile(repo,'src'), fullfile(repo,'src','mex')); addpath(genpath(fullfile(repo,'src','SHL')));

% gas, zt, dt, zhf, Pr_exp, numu_exp, Table4 [om, eh, ehf], E4 refit [om, ehf]
% The E4-refit pairs are the first-order-extraction fits produced by benchmark_polyatomic_transport_fits.m
% under the corrected heat-flux normalization rho = sqrt(5/delta); they are the
% `refitted` rows of figures/fig_transport_fits_points.csv. (The pairs carried
% here before 2026-08-29 were fitted under the superseded rho = sqrt(5/(4 delta))
% and reproduced only mu_b/mu, missing Pr by ~2.6-3.0% for N2 and CO.)
G = { ...
 {'N2','0.534','2.01','0.3',  0.717,0.73, [0.312052,-0.3,-0.207793],  [0.31276402,-0.20588939]}, ...
 {'CO','0.53', '2.01','0.965',0.743,0.55, [0.540506,-0.453,0.570111], [0.54050581, 0.57011043]}, ...
 {'H2','0.608','1.94','0.965',0.686,30.0, [0.0101187,-0.453,-0.133879],[0.01012942,-0.13240930]} };

for gi = 1:numel(G)
    % rho = d_q/d_s = sqrt(5/delta): l=1 heat-flux moment ratio, counterpart of
    % the sqrt(3/delta) that fixes the l=0 energy invariant.
    g = G{gi}; nm=g{1}; delta=str2double(g{3}); nu=delta/2-1; rho=sqrt(5/delta);
    Prx=g{5}; nmx=g{6}; p4=g{7}; rf=g{8};
    % Four channels per gas, routed through the self-building cache layer
    % (build_or_load_dsmc_tensor) instead of load() on hard-coded filenames, so
    % a missing cache rebuilds instead of erroring. Same production convention:
    % K=L=I=2, (rad,tan,int)=(16,16,8), N_lambda=24, conservation ON.
    % On the spectral path GeneralCollisionTensor clamps the internal padding to
    % zero, so the loader names new files ..._p16-16-0_...; the shipped
    % ..._p16-16-8_... caches are still served by its legacy read fallback.
    cfgP = struct('K_max',2,'L_max',2,'I_max',2, ...
                  'zeta',str2double(g{2}),'delta',delta, ...
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
    cc.eta_hat_f = 1; cc.zeta_hat_f = str2double(g{4});
    Jfm  = loadJ(cc);
    Jnfe = JmN - Jnf0; Jfe = Jfm - Jf0;
    Jat = @(w,eh,ehf) w*(Jnf0 + eh*Jnfe) + (1-w)*(Jf0 + ehf*Jfe);
    for cse = {{p4(1), p4(2), p4(3), 'Table-4'}, {rf(1), p4(2), rf(2), 'E4-refit'}}
        c = cse{1}; J = Jat(c{1}, c{2}, c{3});
        [Pr, nm_eig, nm_diag, PPe, PPd, ang] = closures(J, delta, nu, rho);
        fprintf('CT| %s %-8s Pr=%.6f (dev %+ .2f%%)  numu_eig=%.6f (dev %+ .2f%%)  numu_diag=%.6f (dev %+ .2f%%)\n', ...
            nm, c{4}, Pr, 100*(Pr/Prx-1), nm_eig, 100*(nm_eig/nmx-1), nm_diag, 100*(nm_diag/nmx-1));
        fprintf('CT| %s %-8s   P_Pi eig=%.8f diag=%.8f  (rel diff %.2e, Pi-eigvec angle %.2e rad)\n', ...
            nm, c{4}, PPe, PPd, abs(PPd-PPe)/PPe, ang);
    end
end
fprintf('CT| done\n');

function J = loadJ(cfg)
    T = build_or_load_dsmc_tensor(cfg); C = T.assemble_full_tensor();
    J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
end
function [Pr, nm_eig, nm_diag, PPe, PPd, ang] = closures(J, delta, nu, rho)
    N_I=3; N_Q=9; idx=@(k,i,l,m)(k*N_I+i)*N_Q+(l^2+l+m)+1;
    Psig = -J(idx(0,0,2,0), idx(0,0,2,0));
    iq=idx(1,0,1,0); is=idx(0,1,1,0); B=J([iq is],[iq is]);
    Pq0=-B(1,1); Ps0=-B(2,2); Pq1=-B(1,2)*rho; Ps1=-B(2,1)/rho;
    Pr = (5+delta)/2 * 2*(Pq0*Ps0-Pq1*Ps1) / (Psig*(5*(Ps0-Ps1)+delta*(Pq0-Pq1)));   % eq (39)
    bidx=[idx(1,0,0,0), idx(0,1,0,0)]; Jb=J(bidx,bidx);
    [V,E]=eig(Jb); [ev,ord]=sort(real(diag(E)),'descend'); V=V(:,ord);
    PPe = -ev(2);                                              % banked: eigenvalue
    w = [delta*sqrt(3/2); -3*sqrt(nu+1)]; w = w/norm(w);       % dynamic-pressure moment
    PPd = -(w'*Jb*w);                                          % diagonal Grad projection
    v2 = V(:,2)/norm(V(:,2));
    ang = acos(min(abs(w'*v2),1));
    nm_eig  = 2*delta/(3*(3+delta)) * Psig/PPe;                % eq (58), two P_Pi readings
    nm_diag = 2*delta/(3*(3+delta)) * Psig/PPd;
end
