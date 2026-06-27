% benchmark_dsmc_transport.m
% =========================================================================
% Transport-coefficient benchmark for the validated DSMC (eq 54) collision
% kernel of Djordjic, Oblapenko, Pavic-Colic & Torrilhon (Continuum Mech.
% Thermodyn. 2023), evaluated with the Wigner-Eckart spectral operator.
%
% For each calorically-perfect gas (Table 1: delta, zeta) we build the
% non-frozen (omega=1) and frozen (omega=0) linearized operators ONCE, then
% sweep the convex-split parameter omega in J(omega)=omega*J_nf+(1-omega)*J_fr
% (linear in the blend). From the 17-moment production terms we extract:
%   - Shear   : P_sigma^(0)   (L=2, n=0 mode)            -> shear viscosity mu
%   - Bulk    : P_Pi^(0)      (L=0, n=1 mode, prop. omega) -> bulk viscosity nu
%   - Heatflux: L=1 2x2 block (q,s) -> Prandtl Pr (eq 39)
% and report Pr, the bulk/shear ratio nu/mu (eq 58), against the paper's
% Tables 2-3 and the analytic Eucken (38) / Frozen (42) formulas.
%
% VALIDATION NOTES (see commit / memory):
%   * Velocity sector is exact: P_sigma/Pq0 = 3/2 for all zeta (monatomic
%     first-Sonine Pr = 2/3), so mu and the shear mode are exact.
%   * The frozen Prandtl (omega=0) at zeta=0 reproduces Eucken (38) to 1e-14.
%   * At zeta!=0 the frozen Pr DIFFERS from the paper's analytic frozen
%     formula (42); an independent Monte-Carlo of the elastic-collision
%     H-theorem bracket confirms OUR spectral value (b=Ps0/Pq0) to 3-4
%     digits, i.e. eq (42) carries an approximation and the spectral
%     quadrature is the accurate one.
% =========================================================================

proj = fileparts(mfilename('fullpath'));
addpath(fullfile(proj,'src'));
addpath(genpath(fullfile(proj,'src','SHL')));
addpath(fullfile(proj,'src','mex'));

% ---- spectral resolution (heat flux is a degree-3 velocity moment) -------
K_max = 2; L_max = 2; I_max = 2;
rad_pad = 16; tan_pad = 16; int_pad = 4;     % padding beyond exactness

% ---- Table 1 inputs + Table 2/3 reference data ---------------------------
% name, delta, zeta, Pr_measured, Pr_Eucken(paper), Pr_Frozen(paper eq42), nu/mu(Table3 or NaN)
gases = {
  'N2', 2.01, 0.533, 0.717, 0.737,  0.697, 0.73
  'O2', 2.07, 0.441, 0.717, 0.739,  0.705, NaN
  'NO', 2.18, 0.420, 0.740, 0.7417, 0.708, NaN
  'CO', 2.01, 0.530, 0.743, 0.737,  0.698, 0.55
  'H2', 1.94, 0.607, 0.686, 0.735,  0.691, 30.0
};

eucken = @(d)         2*(d+5)/(2*d+15);
frozen42 = @(d,z)     2*(d+5)/( d*(2+0.8*z)*(2*d+2*z+7)/(2*d+z+7) + 15 );

fprintf('\n=== DSMC transport coefficients (Wigner-Eckart spectral operator) ===\n');
fprintf('K_max=%d L_max=%d I_max=%d  padding=(%d,%d,%d)\n\n', ...
        K_max,L_max,I_max,rad_pad,tan_pad,int_pad);

nG = size(gases,1);
res = struct();

for g = 1:nG
    nm = gases{g,1}; delta = gases{g,2}; zeta = gases{g,3};
    Pr_meas = gases{g,4}; Pr_euc_p = gases{g,5}; Pr_frz_p = gases{g,6}; numu_t = gases{g,7};
    nu = delta/2 - 1;
    rho = sqrt(5/(4*delta));                 % heat-flux moment ratio d_q/d_s

    Basis = SpectralBasis(K_max,L_max,I_max,nu);
    % Build non-frozen (omega=1) and frozen (omega=0) operators once each
    Jnf = build_J(Basis, delta, zeta, 1.0, rad_pad,tan_pad,int_pad);
    Jfr = build_J(Basis, delta, zeta, 0.0, rad_pad,tan_pad,int_pad);
    N_I = I_max+1; N_Q = (L_max+1)^2;

    Pr_of  = @(w) prandtl(w*Jnf+(1-w)*Jfr, delta, rho, N_I, N_Q);
    numu_of= @(w) bulkshear(w*Jnf+(1-w)*Jfr, delta, N_I, N_Q);

    Pr0 = Pr_of(0.0);    % frozen
    Pr1 = Pr_of(1.0);    % non-frozen

    % fit omega to the measured Prandtl (Pr monotone in omega); flag if outside
    lo = min(Pr0,Pr1); hi = max(Pr0,Pr1);
    if Pr_meas >= lo && Pr_meas <= hi
        wstar = fzero(@(w) Pr_of(w)-Pr_meas, [0,1]);
        reach = '';
    elseif Pr_meas < lo
        wstar = 0.0; reach = ' (below frozen: needs ext. model 7.2)';
    else
        wstar = 1.0; reach = ' (above non-frozen)';
    end
    if wstar > 1e-4
        numu_star = numu_of(wstar);
    else
        numu_star = NaN;                 % omega*->0: P_Pi prop omega -> nu/mu undefined
    end

    res(g).nm=nm; res(g).delta=delta; res(g).zeta=zeta;
    res(g).Pr0=Pr0; res(g).Pr1=Pr1; res(g).wstar=wstar;
    res(g).numu=numu_star; res(g).Pr_frz_p=Pr_frz_p; res(g).numu_t=numu_t;
    res(g).Pr_meas=Pr_meas; res(g).Pr_euc_p=Pr_euc_p; res(g).reach=reach;

    fprintf('%-3s  delta=%.2f zeta=%.3f\n', nm, delta, zeta);
    fprintf('   Eucken (38):  ours=%.4f  paper=%.4f\n', eucken(delta), Pr_euc_p);
    fprintf('   Frozen (w=0): ours=%.4f  paper-eq42=%.4f  [MC-confirmed ours]\n', Pr0, frozen42(delta,zeta));
    fprintf('   Pr range over omega in [0,1]: [%.4f, %.4f]\n', lo, hi);
    fprintf('   measured Pr=%.3f -> omega*=%.4f%s\n', Pr_meas, wstar, reach);
    if isnan(numu_star)
        fprintf('   nu/mu at omega* (eq58)=n/a (omega*->0, P_Pi->0)');
    else
        fprintf('   nu/mu at omega* (eq58)=%.4f', numu_star);
    end
    if ~isnan(numu_t), fprintf('   Table3=%.2f', numu_t); end
    fprintf('\n\n');
end

% ---- summary table -------------------------------------------------------
fprintf('=== Summary ===\n');
fprintf('%-3s %6s %6s | %7s %7s | %8s %8s | %7s | %6s %6s | %6s %5s\n', ...
    'gas','delta','zeta','Euc','Euc_p','Frz_our','Frz_p','meas','omega*','reach?','nu/mu','T3');
for g=1:nG
    r=res(g);
    fprintf('%-3s %6.2f %6.3f | %7.4f %7.4f | %8.4f %8.4f | %7.3f | %6.4f %6s | %8s %5s\n', ...
        r.nm, r.delta, r.zeta, eucken(r.delta), r.Pr_euc_p, r.Pr0, r.Pr_frz_p, ...
        r.Pr_meas, r.wstar, ternary(isempty(r.reach),'in','OUT'), num3(r.numu), num3(r.numu_t));
end
fprintf('\nNotes: Euc=our Eucken, Frz_our=our frozen Pr (MC-confirmed), Frz_p=paper eq42.\n');
fprintf('omega* fits measured Pr; nu/mu (eq58) is then a prediction (cannot be\n');
fprintf('independently tuned in the DSMC eq-54 model -- that needs the 7.2 model).\n');

% ========================= helpers =======================================
function J = build_J(Basis, delta, zeta, omega, rp, tp, ip)
    Kd = ScatteringKernel('DSMC', struct('zeta',zeta,'delta',delta,'omega',omega));
    T = GeneralCollisionTensor(Basis, Kd);
    T.generate_R_tensor_sumfac(rp, tp, ip);
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
    P_Pi = -evb(2);                       % 1st is the energy invariant (~0)
    r = (2*delta/(3*(3+delta))) * (P_sigma/P_Pi);
end

function s = ternary(c,a,b), if c, s=a; else, s=b; end, end
function s = num3(x), if isnan(x), s='   n/a'; else, s=sprintf('%.3f',x); end, end
