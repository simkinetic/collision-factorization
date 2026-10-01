%% REVISION ITEM 15: generalized eigenvalues of the linearized hard-sphere operator per l block,
% for comparison with Alterman, Frankowski & Pekeris (1962), Table 2. Their lambda_rl solve
%   e^{-p^2} M(p) phi = (lambda/pi) int e^{-p1^2} phi(p1) (R - 2/R e^{w^2}) dp1          (their eq. 20)
% so that J(phi_rl) = -a^2 sqrt(pi) (1 + 1/lambda_rl) M(p) e^{-p^2} phi_rl                (their eq. 21),
% i.e. phi_rl are eigenfunctions of the generalized problem  L phi = mu nu phi  with nu the hard-sphere
% collision frequency (nu ∝ M(p) e^{-p^2}) and mu ∝ -(1 + 1/lambda_rl). Ratios mu_rl / mu_02 are
% independent of all constants and of the basis; we compute them from the Galerkin blocks:
%   L_l c = mu N_l c,   N_l(k,k') = int R_kl R_k'l nu(v) v^2 e^{-v^2} dv,  nu(v) = e^{-v^2}/sqrt(pi) + (v + 1/(2v)) erf(v).
clear; clc;
addpath('src', 'src/mex', 'src/SHL');
out_dir = '/Users/ekke/dev/simkinetic/papers/collision-kernel-factorization/revision/data';
K_max = 16; L_max = 4; gam = 1.0; pad = 16;
Basis = SpectralBasis(K_max, L_max); Kernel = ScatteringKernel(gam);
T = GeneralCollisionTensor(Basis, Kernel); tic; T.generate_R_tensor_sumfac(pad, pad); fprintf('assembly %.1f s\n', toc);
N_Q = Basis.N_Q; C = T.assemble_full_tensor(); J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
q0 = @(l) l^2 + 1 + l;
% radial quadrature on x = v^2 with weight x^{1/2} e^{-x}:  int g(v) v^2 e^{-v^2} dv = 1/2 int g(sqrt x) x^{1/2} e^{-x} dx
qr = Gauss.generalized_laguerre(80, 0.5); v = sqrt(qr.x); w = qr.w / 2;
nu = exp(-v.^2) / sqrt(pi) + (v + 1 ./ (2 * v)) .* erf(v);
fid = fopen(fullfile(out_dir, 'afp62_generalized_eigenvalues.csv'), 'w');
fprintf(fid, 'l,r,mu,mu_over_mu02\n');
mu_all = cell(L_max + 1, 1);
for l = 0:L_max
    idx = (0:K_max) * N_Q + q0(l); Jl = J(idx, idx); Jl = (Jl + Jl') / 2;
    Rl = zeros(numel(v), K_max + 1);
    for k = 0:K_max, Rl(:, k + 1) = Basis.evaluate_radial(k + 1, l, v); end
    Nl = Rl' * (Rl .* (w .* nu)); Nl = (Nl + Nl') / 2;
    mu = sort(eig(Jl, Nl), 'descend'); mu_all{l + 1} = mu;
end
mu02 = mu_all{3}(1);                           % l = 2, r = 0 (slowest l=2 mode)
fprintf('generalized eigenvalues mu (code units) and ratios mu / mu_{02}; K_max = %d\n', K_max);
for l = 0:L_max
    mu = mu_all{l + 1};
    fprintf('l=%d: ', l); fprintf('%10.6f ', mu(1:8)); fprintf('\n');
    for r = 0:numel(mu) - 1, fprintf(fid, '%d,%d,%.12e,%.12f\n', l, r, mu(r + 1), mu(r + 1) / mu02); end
end
fclose(fid);
% orthonormality check of the radial quadrature (should be identity)
l = 2; Rl = zeros(numel(v), K_max + 1); for k = 0:K_max, Rl(:, k + 1) = Basis.evaluate_radial(k + 1, l, v); end
fprintf('radial orthonormality error (l=2): %.2e\n', max(abs(Rl' * (Rl .* w) - eye(K_max + 1)), [], 'all'));
fprintf('Written: %s\n', out_dir);
