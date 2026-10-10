%% REVISION ITEM 15: export the Maxwell-molecule impact tensor for the comparison with Hanke & Torrilhon (2023)
% Writes one row per entry R^{(l1 l2 l3)}_{k1 k2 k3} of the impact tensor (gamma = 0, K_max = L_max = 10) to
%   revision/impact_tensor_maxwell_k10_l10.csv      columns: l1,l2,l3,k1,k2,k3,R
% which is the input of revision/compare_hanke_torrilhon.py:
%   HT_DIR=<dir with SmatrixAMX1099887766554433221100A> python3 revision/compare_hanke_torrilhon.py revision/impact_tensor_maxwell_k10_l10.csv 10 10
% The tensor is assembled here if src/precalc/collisiontensor_k10_l10_gamma0.00.mat does not exist (about 25 min on 12 cores).
clear; clc; addpath('src', 'src/mex', 'src/SHL');

K_max = 10; L_max = 10; gamma = 0.0; pad = 20;
fn = fullfile('src', 'precalc', sprintf('collisiontensor_k%d_l%d_gamma%.2f.mat', K_max, L_max, gamma));
if exist(fn, 'file')
    fprintf('Loading %s\n', fn);
    data = load(fn, 'TensorObj'); TensorObj = data.TensorObj;
else
    fprintf('Assembling K=%d, L=%d, gamma=%.2f (padding %d)...\n', K_max, L_max, gamma, pad);
    Basis = SpectralBasis(K_max, L_max); Kernel = ScatteringKernel(gamma);
    TensorObj = GeneralCollisionTensor(Basis, Kernel);
    tic; TensorObj.generate_R_tensor_sumfac(pad, pad); fprintf('assembled in %.1f s\n', toc);
    save(fn, 'TensorObj', 'Basis', 'Kernel', '-v7.3');
end

% channel c <-> degrees (l1, l2, l3): ic_map numbers the unique rows of the degrees of the Gaunt labels
q2l = @(q) floor(sqrt(double(q) - 1));
channel_l = unique(q2l(TensorObj.gaunt_labels), 'rows');          % [N_T x 3], row c is channel c
N_K = K_max + 1; N_T = size(channel_l, 1);
assert(size(TensorObj.R_tensor, 4) == N_T);

out = fullfile('revision', sprintf('impact_tensor_maxwell_k%d_l%d.csv', K_max, L_max));
fid = fopen(out, 'w'); fprintf(fid, 'l1,l2,l3,k1,k2,k3,R\n');
n = 0;
for c = 1:N_T
    for k3 = 0:K_max
        for k2 = 0:K_max
            for k1 = 0:K_max
                R = TensorObj.R_tensor(k1+1, k2+1, k3+1, c);
                if R ~= 0
                    fprintf(fid, '%d,%d,%d,%d,%d,%d,%.16e\n', channel_l(c,1), channel_l(c,2), channel_l(c,3), k1, k2, k3, R);
                    n = n + 1;
                end
            end
        end
    end
end
fclose(fid);
fprintf('%d nonzero entries in %d channels written to %s\n', n, N_T, out);
