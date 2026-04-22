%% Precompute and Save Spectral Collision Tensors (Hard Spheres Only)
% Generates the factorized collision tensors (R_tensor and Gaunt lists)
% for various angular resolutions with VHS viscosity index omega = 0.5.
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
L_max_list = [4, 6, 8, 10, 13, 16];

% VHS Viscosity Index
vhs_omega = 1.0; 

% For VHS, the collision kernel depends on relative velocity as |u|^alpha
% where alpha = 2 * (1 - omega). For Hard Spheres, alpha = 1.0.
alpha = 2 * (1 - vhs_omega);

fprintf('==============================================================\n');
fprintf('  Processing Hard Sphere Kernel: omega = %.2f (alpha = %.2f)\n', vhs_omega, alpha);
fprintf('==============================================================\n');

for l_idx = 1:length(L_max_list)
    L = L_max_list(l_idx);
    
    % Target filename
    filename = sprintf('collisiontensor_k%d_l%d_vhs_w%.2f.mat', K_max, L, vhs_omega);
    filepath = fullfile(out_dir, filename);
    
    if exist(filepath, 'file')
        fprintf('\n  [SKIP] File already exists: %s\n', filename);
        continue;
    end
    
    fprintf('\nGenerating Tensor for K_max = %d, L_max = %d...\n', K_max, L);
    
    % 1. Initialize Basis and Kernel
    Basis = SpectralBasis(K_max, L);
    
    Kernel = ScatteringKernel(K_max, L, 5, alpha, 1e-6);
    
    % 2. Initialize the Tensor Object
    TensorObj = GeneralCollisionTensor(Basis, Kernel);
    
    % 3. Compute the dense physical R_tensor using the fast C++ MEX routine
    tic;
    TensorObj.generate_R_tensor(true);
    t_gen = toc;
    fprintf('  -> Tensor generation complete in %.2f seconds.\n', t_gen);
    
    % 4. Save the objects to disk (-v7.3 required for large files at L=13, L=16)
    fprintf('  -> Saving to %s...\n', filepath);
    save(filepath, 'TensorObj', 'Basis', 'Kernel', '-v7.3');
    fprintf('  -> Save complete.\n');
end

fprintf('\nAll Hard Sphere tensors have been precomputed and saved successfully.\n');