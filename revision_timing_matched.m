%% REVISION: execution time of the contraction orders at the sizes of the Hanke-Torrilhon comparison (general kernel).
% Timing does not depend on the values of the impact tensor, so R is filled with random numbers of the
% right shape and only the Gaunt geometry (exact, from the constructor) is built: no quadrature needed.
% Sizes: (K,L) = (10,20) is the tightest tensor-product superspace of the total-degree set 2k+l <= 20
% (1771 coefficients, Hanke-Torrilhon's M = 20); (4,18) and (6,14) have about the same number of
% coefficients as that set; (8,12) and (4,12) are the rows of Table 14 (sanity check against it).
% Single core: run MATLAB with -singleCompThread is NOT required, the MEX kernels are sequential.
clear; clc; addpath('src', 'src/mex', 'src/SHL');
out_dir = getenv('TIMING_OUT'); if isempty(out_dir), out_dir = 'revision'; end
fid = fopen(fullfile(out_dir, 'timing_matched.csv'), 'w');
fprintf(fid, 'K,L,N,N_T,N_G,N_S,t_naive,t_radial,t_angular\n');
for KL = {[10 20], [4 18], [6 14], [8 12], [4 12]}
    K = KL{1}(1); L = KL{1}(2);
    B = SpectralBasis(K, L); T = GeneralCollisionTensor(B, ScatteringKernel(1.0));   % geometry only
    K_len = K + 1; N_Q = B.N_Q; N = N_Q * K_len; N_G = numel(T.gaunt_vals); N_T = max(T.ic_map);
    rng(1); R = rand(K_len, K_len, K_len, N_T);
    [~, si] = sortrows([T.ic_map, T.gaunt_labels(:, 1)]); lb = T.gaunt_labels(si, :); vs = T.gaunt_vals(si); ic = T.ic_map(si);
    N_S = size(unique([ic, lb(:, 1)], 'rows'), 1);
    f = rand(N_Q, K_len);
    t_naive   = timeit(@() call_out(@naive_collision_kernel_mex,         zeros(N_Q, K_len), f, T.gaunt_labels, T.gaunt_vals, T.ic_map, R, N_Q, K_len));
    t_angular = timeit(@() call_out(@angular_first_collision_kernel_mex, zeros(N_Q, K_len), f, lb, vs, ic, R, N_Q, K_len));
    t_radial  = timeit(@() call_out(@radial_first_collision_kernel_mex,  zeros(N_Q, K_len), f, lb, vs, ic, R, N_Q, K_len));
    fprintf('K=%2d L=%2d N=%5d N_T=%5d N_G=%8d N_S=%6d | term-by-term %8.2f ms | radial-first %8.2f ms | angular-first %8.2f ms | ratio %.2f\n', ...
        K, L, N, N_T, N_G, N_S, 1e3*t_naive, 1e3*t_radial, 1e3*t_angular, t_naive / t_angular);
    fprintf(fid, '%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e\n', K, L, N, N_T, N_G, N_S, t_naive, t_radial, t_angular);
    clear T B R
end
fclose(fid); fprintf('written to %s\n', fullfile(out_dir, 'timing_matched.csv'));
function out = call_out(fn, out, varargin), fn(out, varargin{:}); end
