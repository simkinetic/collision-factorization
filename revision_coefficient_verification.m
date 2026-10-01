%% REVISION ITEM 15: direct verification of linearized hard-sphere coefficients
% Linearized operator at the absolute Maxwellian, block-diagonal in (l, m). For hard spheres the
% classical Chapman-Cowling bracket matrices (Sonine basis, normalized by the (1,1) element) are
%   viscosity       (l=2, k>=0): [1, 1/4; 1/4, 205/48]   -> [mu]_2/[mu]_1 = 205/202
%   heat conduction (l=1, k>=1): [1, 1/4; 1/4, 45/16]    -> [lambda]_2/[lambda]_1 = 45/44
% (from Omega^(2,2)=2, Omega^(2,3)=8, Omega^(2,4)=40 for rigid spheres; off-diagonal sign is a
% basis convention). Successive-approximation sequences and infinite-order limits:
%   Pekeris & Alterman 1957, eqs (54) and (49): mu: 1, 1.01485, 1.01588, 1.01600 -> 1.016034
%                                              lambda: 1, 1.02273, 1.02482, 1.02513 -> 1.025218
% We assemble a fresh hard-sphere tensor at K_max=10, L_max=4 (padding 16) and report both blocks,
% the sequences f^(K) = [H_K^{-1}]_11 and the eigenvalues of every (l, m=0) block (for comparison
% with Alterman, Frankowski & Pekeris 1962 once that table is available).
clear; clc;
addpath('src', 'src/mex', 'src/SHL');
out_dir = '/Users/ekke/dev/simkinetic/papers/collision-kernel-factorization/revision/data';
K_max = 10; L_max = 4; gamma = 1.0; pad = 16;

Basis = SpectralBasis(K_max, L_max); Kernel = ScatteringKernel(gamma);
T = GeneralCollisionTensor(Basis, Kernel); T.generate_R_tensor_sumfac(pad, pad);
N_Q = Basis.N_Q; N = Basis.N_terms;
C = T.assemble_full_tensor();
J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));          % Jacobian at c_eq = e_1
q0 = @(l) l^2 + 1 + l;                               % index of (l, m=0)

% ---- l=2 block (viscosity), k = 0..K ----
idx2 = (0:K_max) * N_Q + q0(2); H2 = J(idx2, idx2); H2 = H2 / H2(1,1);
% ---- l=1 block (heat conduction), k = 1..K (k=0 is momentum, null) ----
idx1 = (1:K_max) * N_Q + q0(1); H1 = J(idx1, idx1); H1 = H1 / H1(1,1);

% Convert from the orthonormal Laguerre basis to Chapman-Cowling's unnormalized Sonine polynomials:
% a_pq = H_pq * sqrt(c_p c_q), c_k = Gamma(k + l + 3/2) / k!, then normalize by a_11.
cc = @(k, l) exp(gammaln(k + l + 1.5) - gammaln(k + 1));   % Gamma(k+l+3/2)/k!  ('gamma' is shadowed above)
k2 = 0:K_max; A2 = H2 .* sqrt(cc(k2', 2) * cc(k2, 2)); A2 = A2 / A2(1,1);
k1 = 1:K_max; A1 = H1 .* sqrt(cc(k1', 1) * cc(k1, 1)); A1 = A1 / A1(1,1);
A2_exact = [1, -1/4; -1/4, 205/48]; A1_exact = [1, -1/4; -1/4, 45/16];
err2 = max(abs(A2(1:2,1:2) - A2_exact), [], 'all'); err1 = max(abs(A1(1:2,1:2) - A1_exact), [], 'all');
fprintf('--- 2x2 bracket matrices in Sonine normalization (a_pq / a_11), hard spheres ---\n');
fprintf('viscosity   [%.12f %.12f; %.12f %.12f]  exact [1 -1/4; -1/4 205/48]  max err %.2e\n', A2(1,1), A2(1,2), A2(2,1), A2(2,2), err2);
fprintf('conduction  [%.12f %.12f; %.12f %.12f]  exact [1 -1/4; -1/4 45/16]   max err %.2e\n', A1(1,1), A1(1,2), A1(2,1), A1(2,2), err1);
fid = fopen(fullfile(out_dir, 'coefficient_verification_matrices.csv'), 'w');
fprintf(fid, 'block,a12_over_a11,a22_over_a11,a12_exact,a22_exact,max_err\n');
fprintf(fid, 'viscosity_l2,%.12f,%.12f,%.12f,%.12f,%.3e\n', A2(1,2), A2(2,2), -0.25, 205/48, err2);
fprintf(fid, 'conduction_l1,%.12f,%.12f,%.12f,%.12f,%.3e\n', A1(1,2), A1(2,2), -0.25, 45/16, err1);
fclose(fid);

fprintf('\n--- successive approximations f^(K) = [H_K^-1]_11 ---\n');
fid = fopen(fullfile(out_dir, 'coefficient_verification_transport.csv'), 'w');
fprintf(fid, 'order,f_mu,f_mu_ref,f_lambda,f_lambda_ref\n');
mu_ref = [1, 205/202, 1.01588, 1.01600, nan(1, K_max - 3)]; la_ref = [1, 45/44, 1.02482, 1.02513, nan(1, K_max - 4)];
fprintf('%6s %12s %12s %12s %12s\n', 'order', 'f_mu', 'PA57', 'f_lambda', 'PA57');
for k = 1:K_max
    Hi2 = inv(H2(1:k, 1:k)); Hi1 = inv(H1(1:k, 1:k));
    fprintf('%6d %12.6f %12.6f %12.6f %12.6f\n', k, Hi2(1,1), mu_ref(k), Hi1(1,1), la_ref(k));
    fprintf(fid, '%d,%.8f,%.6f,%.8f,%.6f\n', k, Hi2(1,1), mu_ref(k), Hi1(1,1), la_ref(k));
end
fclose(fid);
Hi2 = inv(H2); Hi1 = inv(H1);
fprintf('infinite-order: f_mu(K=%d) = %.6f (PA57: 1.016034)   f_lambda(K=%d) = %.6f (PA57: 1.025218)\n', ...
    K_max + 1, Hi2(1,1), K_max, Hi1(1,1));

% ---- eigenvalues of each (l, m=0) block, k = 0..K ----
fid = fopen(fullfile(out_dir, 'linearized_hs_eigenvalues.csv'), 'w');
fprintf(fid, 'l,index,eigenvalue_code_units,ratio_to_l2k0\n');
lam2 = eig(J(idx2, idx2)); lam_ref = -max(abs(lam2(abs(lam2) > 1e-10))); % will be replaced below by the l=2 lowest mode
lam_ref = max(lam2(lam2 < -1e-10));                                       % least negative l=2 eigenvalue (stress relaxation)
fprintf('\n--- eigenvalues per l block (m=0), K_max=%d, code units; ratio to the slowest l=2 mode ---\n', K_max);
for l = 0:L_max
    idx = (0:K_max) * N_Q + q0(l); lam = sort(eig(J(idx, idx)), 'descend');
    fprintf('l=%d: %s\n', l, sprintf('%9.5f ', lam(1:min(6, end))));
    for i = 1:numel(lam), fprintf(fid, '%d,%d,%.10e,%.10f\n', l, i, lam(i), lam(i) / lam_ref); end
end
fclose(fid);
fprintf('max |imag| over all block eigenvalues: %.2e\n', max(abs(imag(eig(J)))));
fprintf('\nWritten: %s\n', out_dir);
