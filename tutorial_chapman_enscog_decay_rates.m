%% VALIDATION: Chapman-Enskog Infinite-Order Viscosity Benchmark
% This script loads a precomputed Wigner-Eckart collision tensor, 
% extracts the discrete Jacobian for the L=2 stress mode, and compares 
% it directly against the exact analytical Bracket Integral matrix 
% derived in classical Chapman-Cowling kinetic theory.

clear; clc; addpath('src', 'src/mex','src/SHL');
fprintf('==============================================================\n');
fprintf('  BENCHMARK: Chapman-Enskog Matrix & Viscosity Corrections\n');
fprintf('==============================================================\n\n');

%% 1. Load Precomputed Tensor
K_max = 4;
L_max = 4;
vhs_omega = 0.5; % 0.5 corresponds to Hard Spheres

filename = sprintf('src/precalc/collisiontensor_k%d_l%d_vhs_w%.2f.mat', K_max, L_max, vhs_omega);
if ~exist(filename, 'file')
    error('Precomputed tensor %s not found. Please run the batch generation script.', filename);
end

fprintf('1. Loading Precomputed Tensor: %s\n', filename);
data = load(filename);
Basis = data.Basis;
TensorObj = data.TensorObj;

N_terms = Basis.N_terms;
N_Q = Basis.N_Q;

C_assembled = TensorObj.assemble_full_tensor();

%% 2. Extract the L=2 Jacobian Block
% Evaluate the collision operator Jacobian at the equilibrium Maxwellian
c_eq = zeros(N_terms, 1);
c_eq(1) = 1.0; 
J_eq = squeeze(C_assembled(:,:,1)) * c_eq(1) + squeeze(C_assembled(:,1,:)) * c_eq(1);

% Extract the specific KxK block for the L=2 mode.
% We use m=0, which corresponds to q=7 in the angular basis.
q_stress = 7; 
idx_L2 = (0:K_max) * N_Q + q_stress;
J_L2 = J_eq(idx_L2, idx_L2);

%% 3. Normalize the Numerical Matrix
% In CE theory, the Bracket Integral matrix H is normalized such that H(1,1) = 1.
% We divide our extracted Jacobian block by its top-left element.
H_num = J_L2 / J_L2(1,1);

%% 4. The Exact Analytical Chapman-Cowling Matrix (2x2 Block)
% For Hard Spheres, classical kinetic theory gives exact rational fractions:
H_ana = [1.0,  0.25; 
         0.25, 205/48];

fprintf('\n--- MATRIX ELEMENT VALIDATION ---\n');
fprintf('Exact Analytical H Matrix (2x2):\n');
disp(H_ana);

fprintf('Numerical Spectral H Matrix (Top 2x2 Block):\n');
disp(H_num(1:2, 1:2));

matrix_error = max(abs(H_num(1:2, 1:2) - H_ana), [], 'all');
fprintf('Maximum Matrix Element Error: %.4e\n\n', matrix_error);

%% 5. Compute "Infinite-Order" Viscosity Correction Factors
% The correction factor f_mu^(K) is the top-left element of the inverse 
% of the truncated (KxK) bracket integral matrix.
fprintf('--- VISCOSITY CORRECTION FACTORS f_mu ---\n');
fprintf('Theoretical Limits:\n');
fprintf('  1st-Order (1x1) : 1.000000\n');
fprintf('  2nd-Order (2x2) : 1.014851  (205/202)\n');
fprintf('  Inf-Order (inf) : 1.016034\n\n');

fprintf('Numerical Spectral Limits (Increasing K_max):\n');
for k = 1:(K_max+1)
    % Truncate the matrix to k x k
    H_sub = H_num(1:k, 1:k);
    
    % Invert the submatrix
    H_inv = inv(H_sub);
    
    % The viscosity correction is the top-left element
    f_mu = H_inv(1,1);
    
    fprintf('  Using K_max = %d (%dx%d matrix): f_mu = %.6f\n', k-1, k, k, f_mu);
end
fprintf('==============================================================\n');