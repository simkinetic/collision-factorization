%% Precompute and Save Spectral Collision Tensors (Hard Spheres Only)
% Generates the factorized collision tensors (R_tensor and Gaunt lists)
% for various angular resolutions.
clear; clc; close all;
addpath('src', 'src/mex','src/SHL');

fprintf('==============================================================\n');
fprintf('  BATCH PRECOMPUTATION: Hard Sphere Collision Tensors\n');
fprintf('==============================================================\n\n');

% Ensure the target directory exists
out_dir = fullfile('src', 'precalc');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Created directory: %s\n', out_dir);
end

% --- Configuration ---
K_max = 4;
L_max_list = [2, 4, 6];
pad = 20; % Quadrature padding for high-precision precomputation

% Interaction Potential / Kernel Singularity Order
% gamma = 1.0 corresponds to Hard Spheres, gamma = 0.0 to Maxwell molecules.
gamma = 0.0; 

fprintf('==============================================================\n');
fprintf('  Processing Hard Sphere Kernel: gamma = %.2f\n', gamma);
fprintf('  Quadrature Padding: %d\n', pad);
fprintf('==============================================================\n');

for l_idx = 1:length(L_max_list)
    L = L_max_list(l_idx);
    
    % Target filename updated to reflect gamma instead of omega
    filename = sprintf('collisiontensor_k%d_l%d_gamma%.2f.mat', K_max, L, gamma);
    filepath = fullfile(out_dir, filename);
    
    if exist(filepath, 'file')
        fprintf('\n  [SKIP] File already exists: %s\n', filename);
        continue;
    end
    
    fprintf('\nGenerating Tensor for K_max = %d, L_max = %d...\n', K_max, L);
    
    % 1. Initialize Basis and Kernel directly with gamma
    Basis = SpectralBasis(K_max, L);
    Kernel = ScatteringKernel(gamma);
    
    % 2. Initialize the Tensor Object
    TensorObj = GeneralCollisionTensor(Basis, Kernel);
    
    % 3. Compute the dense physical R_tensor using the sum-factorized routine
    tic;
    TensorObj.generate_R_tensor_sumfac(pad, pad);
    t_gen = toc;
    fprintf('  -> Tensor generation complete in %.2f seconds.\n', t_gen);
    
    % 4. Save the objects to disk (-v7.3 required for large files)
    fprintf('  -> Saving to %s...\n', filepath);
    save(filepath, 'TensorObj', 'Basis', 'Kernel', '-v7.3');
    fprintf('  -> Save complete.\n');
end

fprintf('\nAll Hard Sphere tensors have been precomputed and saved successfully.\n');