%% REVISION ITEM 16: Galilean-invariance test for Maxwell molecules AND hard spheres (K=4, L=6)
% Shifted Maxwellian projected onto the truncated basis; Q(c,c) evaluated with the full tensor.
% Reported per bulk velocity U: ||Q||_2 over all modes (truncation residual; exact Q is zero),
% max |Q| over the l=2 stress modes, max |Q| over all non-invariant modes, max |Q| over the five
% collision invariants. Hard spheres have no degree selection rule, so the non-invariant residual
% is a genuine truncation error while the invariants stay at machine precision.
clear; clc;
addpath('src', 'src/mex', 'src/SHL');
out_dir = fullfile('revision', 'data'); if ~exist(out_dir, 'dir'), mkdir(out_dir); end
K_max = 4; L_max = 6;

% --- 3D quadrature grid for the projection (thermal scaling: weight e^{-v^2}) ---
N_rad = 60; qr_rad = Gauss.generalized_laguerre(N_rad, 0.5); r_nodes = sqrt(qr_rad.x); w_rad = qr_rad.w / 2;
N_pol = 30; qr_pol = Gauss.legendre(N_pol, -1, 1); cos_t = qr_pol.x; sin_t = sqrt(1 - cos_t.^2); w_pol = qr_pol.w;
N_azi = 30; phi = linspace(0, 2*pi, N_azi+1); phi(end) = []; w_azi = 2*pi / N_azi;
[ia, ip, iz] = ndgrid(1:N_rad, 1:N_pol, 1:N_azi);
v_vec = [r_nodes(ia(:)) .* sin_t(ip(:)) .* cos(phi(iz(:))'), ...
         r_nodes(ia(:)) .* sin_t(ip(:)) .* sin(phi(iz(:))'), ...
         r_nodes(ia(:)) .* cos_t(ip(:))];
W_tot = w_rad(ia(:)) .* w_pol(ip(:)) * w_azi;

fid = fopen(fullfile(out_dir, 'galilean_invariance.csv'), 'w');
fprintf(fid, 'gamma,U,normQ,maxQ_stress,maxQ_noninv,maxQ_inv\n');
U_tests = [0.0, 0.1, 0.3, 0.6];
for gamma = [0.0, 1.0]
    f = sprintf('src/precalc/collisiontensor_k%d_l%d_gamma%.2f.mat', K_max, L_max, gamma);
    d = load(f, 'TensorObj', 'Basis'); T = d.TensorObj; Basis = d.Basis;
    N_Q = Basis.N_Q; N = Basis.N_terms;
    C = T.assemble_full_tensor(); C_flat = reshape(C, N, N^2);
    Psi = Basis.evaluate(v_vec);
    idx_inv = [1, 2, 3, 4, 1*N_Q + 1];          % mass; momentum (l=1, m=-1,0,1); energy (k=1,l=0)
    idx_stress = 5:9;                            % k=0, l=2
    idx_noninv = setdiff(1:N, idx_inv);
    fprintf('\ngamma = %.1f (K=%d, L=%d, N=%d)\n', gamma, K_max, L_max, N);
    fprintf('%8s %12s %12s %12s %12s\n', 'U', '||Q||_2', 'max|Q| l=2', 'max|Q| nonin', 'max|Q| inv');
    for U = U_tests
        F_v = pi^(-1.5) * exp(2 * (v_vec * [U; 0; 0]) - U^2);   % f / e^{-v^2}
        c = Psi' * (W_tot .* F_v);
        cc = c * c'; Q = C_flat * cc(:);
        vals = [norm(Q), max(abs(Q(idx_stress))), max(abs(Q(idx_noninv))), max(abs(Q(idx_inv)))];
        fprintf('%8.1f %12.3e %12.3e %12.3e %12.3e\n', U, vals);
        fprintf(fid, '%.1f,%.1f,%.6e,%.6e,%.6e,%.6e\n', gamma, U, vals);
    end
end
fclose(fid);
fprintf('\nWritten: %s\n', fullfile(out_dir, 'galilean_invariance.csv'));
