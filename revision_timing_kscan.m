%% REVISION ITEM 17: execution time of the contraction orders for K_max = 4, 6, 8 at L_max = 12 (hard spheres).
% Same method as benchmark_collision_contraction_scaling.m (timeit on the MEX kernels, random coefficients).
% The dense baseline is timed only where the dense tensor fits in memory (K_max = 4: 4.8 GB).
clear; clc; addpath('src', 'src/mex', 'src/SHL');
out_dir = '/Users/ekke/dev/simkinetic/papers/collision-kernel-factorization/revision/data';
L = 12; fid = fopen(fullfile(out_dir, 'timing_kscan.csv'), 'w');
fprintf(fid, 'K,L,N,N_T,N_G,N_S,t_dense,t_naive,t_radial,t_angular\n');
for K = [4 6 8]
    d = load(sprintf('src/precalc/collisiontensor_k%d_l%d_gamma1.00.mat', K, L), 'TensorObj', 'Basis'); T = d.TensorObj; B = d.Basis;
    K_len = K + 1; N_Q = B.N_Q; N = N_Q * K_len; N_G = numel(T.gaunt_vals); N_T = size(T.R_tensor, 4);
    [~, si] = sortrows([T.ic_map, T.gaunt_labels(:, 1)]); lb = T.gaunt_labels(si, :); vs = T.gaunt_vals(si); ic = T.ic_map(si); R = T.R_tensor;
    N_S = size(unique([ic, lb(:, 1)], 'rows'), 1);
    f = rand(N_Q, K_len);
    t_naive   = timeit(@() call_out(@naive_collision_kernel_mex,         zeros(N_Q, K_len), f, T.gaunt_labels, T.gaunt_vals, T.ic_map, R, N_Q, K_len));
    t_angular = timeit(@() call_out(@angular_first_collision_kernel_mex, zeros(N_Q, K_len), f, lb, vs, ic, R, N_Q, K_len));
    t_radial  = timeit(@() call_out(@radial_first_collision_kernel_mex,  zeros(N_Q, K_len), f, lb, vs, ic, R, N_Q, K_len));
    t_dense = NaN;
    if 8 * N^3 < 8e9
        fprintf('  assembling dense tensor K=%d L=%d (N=%d)...\n', K, L, N); tic;
        C = zeros(N, N, N);
        for z = 1:N_G
            q1 = T.gaunt_labels(z, 1); q2 = T.gaunt_labels(z, 2); q3 = T.gaunt_labels(z, 3); g = T.gaunt_vals(z); t = T.ic_map(z);
            for k1 = 0:K, for k2 = 0:K, for k3 = 0:K
                C(k1 * N_Q + q1, k2 * N_Q + q2, k3 * N_Q + q3) = R(k1 + 1, k2 + 1, k3 + 1, t) * g;
            end, end, end
        end
        fprintf('  dense assembled in %.0f s\n', toc);
        t_dense = timeit(@() call_out(@dense_tensor_kernel_mex, zeros(N, 1), f(:), C(:), N)); clear C
    end
    fprintf('K=%d L=%d N=%d N_G=%d N_S=%d | dense %.4f s | naive %.4f s | radial %.4f s | angular %.4f s | naive/angular %.2f\n', ...
        K, L, N, N_G, N_S, t_dense, t_naive, t_radial, t_angular, t_naive / t_angular);
    fprintf(fid, '%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e\n', K, L, N, N_T, N_G, N_S, t_dense, t_naive, t_radial, t_angular);
end
fclose(fid); fprintf('written\n');
function out = call_out(fn, out, varargin), fn(out, varargin{:}); end
