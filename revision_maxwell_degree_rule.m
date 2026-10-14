%% REVISION ITEM 14: Maxwell-molecule degree selection rule on the assembled tensor
% For a u-independent kernel (gamma = 0) the collision tensor vanishes unless
%     2 k1 + l1 = 2 k2 + l2 + 2 k3 + l3     (Cai-Fan-Wang 2020 Thm 2; Hanke-Torrilhon 2023 eq. 36).
% The rule acts on the reduced tensor R^{(c)}_{k1 k2 k3} channel-wise (the Gaunt factor does not
% change it). We report the largest |R| over the forbidden entries relative to the largest |R|,
% for the precomputed tensors and for a freshly assembled one, as a quadrature validation.
% Note: the null-space zeroing (l1=0: k1 in {0,1}; l1=1: k1=0) is already applied in R.
clear; clc;
addpath('src', 'src/mex', 'src/SHL');
out_dir = fullfile('revision', 'data'); if ~exist(out_dir, 'dir'), mkdir(out_dir); end

cases = {};
for L = [2, 4, 6]
    f = sprintf('src/precalc/collisiontensor_k4_l%d_gamma0.00.mat', L);
    d = load(f, 'TensorObj'); cases{end+1} = struct('name', sprintf('precalc K=4 L=%d', L), 'T', d.TensorObj, 'K', 4, 'L', L); %#ok<SAGROW>
end
% fresh assembly, moderate and high padding, to show the rule is met by the quadrature itself
for pad = [0, 8, 16]
    Basis = SpectralBasis(4, 4); Kernel = ScatteringKernel(0.0);
    T = GeneralCollisionTensor(Basis, Kernel);
    T.generate_R_tensor_sumfac(pad, pad);
    cases{end+1} = struct('name', sprintf('fresh K=4 L=4 pad=%d', pad), 'T', T, 'K', 4, 'L', 4); %#ok<SAGROW>
end

fid = fopen(fullfile(out_dir, 'maxwell_degree_rule.csv'), 'w');
fprintf(fid, 'case,K,L,n_channels,n_entries,n_forbidden,max_abs_R,max_abs_forbidden,ratio\n');
fprintf('%-24s %6s %9s %11s %12s %12s %10s\n', 'case', 'K', 'entries', 'forbidden', 'max|R|', 'max|R|forb', 'ratio');
for i = 1:numel(cases)
    T = cases{i}.T; K = cases{i}.K; R = T.R_tensor;              % [K+1, K+1, K+1, N_T]
    labels = T.gaunt_labels; l_of_q = floor(sqrt(double(labels) - 1));
    [chan_l, ~, ic] = unique(l_of_q, 'rows');                     % row c = (l1,l2,l3) of channel c
    assert(isequal(ic(:), double(T.ic_map(:))), 'channel ordering mismatch');
    NT = size(chan_l, 1); assert(NT == size(R, 4));
    [k1, k2, k3, c] = ndgrid(0:K, 0:K, 0:K, 1:NT);
    l1 = chan_l(:, 1); l2 = chan_l(:, 2); l3 = chan_l(:, 3);
    deg1 = 2*k1 + l1(c); deg23 = 2*k2 + l2(c) + 2*k3 + l3(c);
    forb = (deg1 ~= deg23);
    maxR = max(abs(R(:))); maxF = max(abs(R(forb)));
    fprintf('%-24s %6d %9d %11d %12.3e %12.3e %10.2e\n', cases{i}.name, K, numel(R), nnz(forb), maxR, maxF, maxF/maxR);
    fprintf(fid, '%s,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e\n', cases{i}.name, K, cases{i}.L, NT, numel(R), nnz(forb), maxR, maxF, maxF/maxR);
end
fclose(fid);
fprintf('\nWritten: %s\n', fullfile(out_dir, 'maxwell_degree_rule.csv'));
