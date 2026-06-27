% benchmark_dsmc_extended.m
% =========================================================================
% Extended-model (eq 43) transport benchmark: reach the EXPERIMENTAL Prandtl
% number (Table 2) and bulk/shear viscosity ratio nu/mu (Table 3) for the
% polyatomic gases N2 / CO / H2, using the internal-energy-modulated kernel
%
%   B = b(u.s)|u|^z { K_d w R^{z/2}[1 + eh ((r(1-R)i)^{zh/2}+((1-r)(1-R)i*)^{zh/2})]
%                   + (1-w) d_{r-r'}d_{R-R'} e_tr^{z/2}[1 + ehf (i^{zhf}+i*^{zhf})] }
%
% with i=I/E, i*=J/E.  eh,ehf may be negative (>= -1/2), which lowers Pr below
% the frozen/Eucken curve and decouples nu/mu from Pr -- the two knobs the base
% DSMC (eq 54) model lacked.
%
% Stages:
%   E3  -- assemble at the paper's Table-4 fit parameters; report Pr & nu/mu vs
%          experiment. Expect *close* but not exact (our quadrature is more
%          accurate than the paper's -- see benchmark_dsmc_transport.m notes).
%   E4  -- with (eta_hat, zeta_hat) fixed at Table-4, RE-FIT to hit the
%          experimental Pr and nu/mu exactly in our operator. Primary fit frees
%          (omega, eta_hat_f) -- linear in the convex blend, so it is a fast 2x2
%          solve on pre-built operators. If the experimental point lies outside
%          the in-bounds image (eta_hat_f >= -1/2), the frozen exponent
%          zeta_hat_f is freed too (nonlinear, rebuilds the frozen channel).
% =========================================================================

proj = fileparts(mfilename('fullpath'));
addpath(fullfile(proj,'src'));
addpath(genpath(fullfile(proj,'src','SHL')));
addpath(fullfile(proj,'src','mex'));

K_max = 2; L_max = 2; I_max = 2;
pad = [16 16 8];     % [radial tangential internal]; E2 plateau

% name, delta, zeta, Pr_meas, numu_meas, [Table-4: omega, eta_hat, zeta_hat, eta_hat_f, zeta_hat_f]
gases = {
  'N2', 2.01, 0.533, 0.717, 0.73, 0.312052, -0.3,   0.965, -0.207793, 0.3
  'CO', 2.01, 0.530, 0.743, 0.55, 0.540506, -0.453, 0.965,  0.570111, 0.965
  'H2', 1.94, 0.607, 0.686, 30.0, 0.0101187,-0.453, 0.965, -0.133879, 0.965
};

fprintf('\n=== Extended model (eq 43): experimental Pr and nu/mu ===\n');
fprintf('K=%d L=%d I=%d  padding=(%d,%d,%d)\n\n', K_max,L_max,I_max,pad(1),pad(2),pad(3));

for g = 1:size(gases,1)
    process_gas(gases(g,:), K_max, L_max, I_max, pad);
end

% ========================= per-gas driver ================================
function process_gas(gs, K_max, L_max, I_max, pad)
    nm=gs{1}; delta=gs{2}; zeta=gs{3}; Pr_m=gs{4}; numu_m=gs{5};
    om4=gs{6}; eh=gs{7}; zh=gs{8}; ehf4=gs{9}; zhf=gs{10};
    nu = delta/2-1; rho = sqrt(5/(4*delta));
    Basis = SpectralBasis(K_max,L_max,I_max,nu);
    N_I=I_max+1; N_Q=(L_max+1)^2;

    % Non-frozen pair (eta_hat linear), built once (zeta_hat fixed):
    Jnf0 = buildJ(Basis,delta,zeta,1.0, 0.0,zh, 0.0,zhf, pad);  % eh=0
    Jnf1 = buildJ(Basis,delta,zeta,1.0, 1.0,zh, 0.0,zhf, pad);  % eh=1
    Jnf_e = Jnf1 - Jnf0;

    % Frozen pair (eta_hat_f linear) at a given zeta_hat_f, cached so we only
    % re-MEX the frozen channel when zeta_hat_f changes in the nonlinear search.
    zc = NaN; Jf0 = []; Jf_e = [];
    function [a,b] = frozenPair(zf)
        if isnan(zc) || abs(zf-zc) > 1e-12
            f0 = buildJ(Basis,delta,zeta,0.0, 0.0,zh, 0.0,zf, pad);
            f1 = buildJ(Basis,delta,zeta,0.0, 0.0,zh, 1.0,zf, pad);
            Jf0 = f0; Jf_e = f1 - f0; zc = zf;
        end
        a = Jf0; b = Jf_e;
    end
    function J = Jof(w,eh_,ehf,zf)
        [a,b] = frozenPair(zf);
        J = w*(Jnf0 + eh_*Jnf_e) + (1-w)*(a + ehf*b);
    end
    Pr_of   = @(w,eh_,ehf,zf) prandtl(Jof(w,eh_,ehf,zf), delta, rho, N_I, N_Q);
    numu_of = @(w,eh_,ehf,zf) bulkshear(Jof(w,eh_,ehf,zf), delta, N_I, N_Q);

    % E3: paper's Table-4 parameters in our operator.
    Pr3 = Pr_of(om4, eh, ehf4, zhf); numu3 = numu_of(om4, eh, ehf4, zhf);

    % E4 primary: free (omega, eta_hat_f) -- linear, fast.
    fopts = optimoptions('fsolve','Display','off','FunctionTolerance',1e-12,'StepTolerance',1e-12);
    F1 = @(x)[ Pr_of(x(1),eh,x(2),zhf)-Pr_m ; numu_of(x(1),eh,x(2),zhf)-numu_m ];
    [x1,Fv1,fl1] = fsolve(F1, [om4; ehf4], fopts);
    wfit=x1(1); ehffit=x1(2); zffit=zhf; Fval=Fv1; flag=fl1; how='(omega,eta_hat_f)';
    if ehffit < -0.5 || wfit < 0 || wfit > 1 || norm(Fv1) > 1e-6
        % Experimental point outside the in-bounds (omega,eta_hat_f) image at the
        % Table-4 zeta_hat_f. Scan the frozen exponent zeta_hat_f on a coarse grid
        % (one frozen rebuild each); at each value the inner (omega,eta_hat_f) solve
        % is a cheap 2x2 root-find on the pre-built operators (the (Pr,nu/mu) targets
        % are met exactly for any zeta_hat_f -- only the bounds decide feasibility).
        % Pick the feasible zeta_hat_f whose eta_hat_f is most interior to [-1/2, inf).
        % At each zeta_hat_f the (omega,eta_hat_f) solution hits both targets EXACTLY;
        % the bound eta_hat_f >= -1/2 is what decides physical admissibility. Report the
        % exact-hit solution whose eta_hat_f is largest (closest to / above the bound).
        zgrid = [0.05:0.05:0.5, 0.6:0.1:2.0]; best = []; besteh = -inf;
        for zf = zgrid
            Fz = @(x)[ Pr_of(x(1),eh,x(2),zf)-Pr_m ; numu_of(x(1),eh,x(2),zf)-numu_m ];
            [xz,Fvz,flz] = fsolve(Fz, [max(om4,0.02); -0.45], fopts);
            if flz>0 && norm(Fvz)<1e-7 && xz(1)>=0 && xz(1)<=1 && xz(2)>besteh
                besteh = xz(2); best=[xz; zf]; Fval=Fvz; flag=flz;
            end
        end
        if ~isempty(best)
            wfit=best(1); ehffit=best(2); zffit=best(3);
        else
            wfit=x1(1); ehffit=x1(2); zffit=zhf; flag=0;
            Fval=[ Pr_of(wfit,eh,ehffit,zffit)-Pr_m ; numu_of(wfit,eh,ehffit,zffit)-numu_m ];
        end
        inb = ehffit >= -0.5;
        how = sprintf('(omega,eta_hat_f) scan zeta_hat_f [in-bounds=%d]', inb);
    end
    Pr4 = Pr_of(wfit,eh,ehffit,zffit); numu4 = numu_of(wfit,eh,ehffit,zffit);

    fprintf('%-3s  delta=%.2f zeta=%.3f  (zeta_hat=%.3f)\n', nm,delta,zeta,zh);
    fprintf('  experiment:        Pr=%.4f   nu/mu=%.4f\n', Pr_m, numu_m);
    fprintf('  E3 Table-4 params: Pr=%.4f   nu/mu=%.4f   (omega=%.4f eta_hat=%.4f eta_hat_f=%.4f zeta_hat_f=%.3f)\n', Pr3,numu3,om4,eh,ehf4,zhf);
    fprintf('  E4 re-fit ours:    Pr=%.4f   nu/mu=%.4f   (omega=%.4f eta_hat=%.4f eta_hat_f=%.4f zeta_hat_f=%.3f)\n', Pr4,numu4,wfit,eh,ehffit,zffit);
    fprintf('     free %s  flag=%d  residual=%.1e\n\n', how, flag, norm(Fval));
end

% ========================= helpers =======================================
function J = buildJ(Basis, delta, zeta, omega, eh, zh, ehf, zhf, pad)
    Kd = ScatteringKernel('DSMC', struct('zeta',zeta,'delta',delta,'omega',omega, ...
        'eta_hat',eh,'zeta_hat',zh,'eta_hat_f',ehf,'zeta_hat_f',zhf));
    T = GeneralCollisionTensor(Basis, Kd);
    T.generate_R_tensor_sumfac(pad(1), pad(2), pad(3));
    C = T.assemble_full_tensor();
    J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
end

function Pr = prandtl(J, delta, rho, N_I, N_Q)
    idx = @(k,i,l,m) (k*N_I+i)*N_Q + (l^2+l+m) + 1;
    P_sigma = -J(idx(0,0,2,0), idx(0,0,2,0));
    iq = idx(1,0,1,0); is = idx(0,1,1,0); JL1 = J([iq is],[iq is]);
    Pq0 = -JL1(1,1); Ps0 = -JL1(2,2);
    Pq1 = -JL1(1,2)*rho; Ps1 = -JL1(2,1)/rho;
    Pr = (5+delta)/2 * 2*(Pq0*Ps0 - Pq1*Ps1) / ...
         (P_sigma*(5*(Ps0-Ps1) + delta*(Pq0-Pq1)));
end

function r = bulkshear(J, delta, N_I, N_Q)
    idx = @(k,i,l,m) (k*N_I+i)*N_Q + (l^2+l+m) + 1;
    P_sigma = -J(idx(0,0,2,0), idx(0,0,2,0));
    bidx = [idx(1,0,0,0), idx(0,1,0,0)];
    evb = sort(real(eig(J(bidx,bidx))),'descend');
    P_Pi = -evb(2);
    r = (2*delta/(3*(3+delta))) * (P_sigma/P_Pi);
end
