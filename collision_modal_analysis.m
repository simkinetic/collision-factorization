%% VALIDATION: Chapman-Enskog Infinite-Order Viscosity Benchmark
% Modified for MAXWELL MOLECULES (omega = 1.0)
clear; clc; addpath('src', 'src/mex','src/SHL');

fprintf('==============================================================\n');
fprintf('  BENCHMARK: Chapman-Enskog Matrix (MAXWELL MOLECULES)\n');
fprintf('==============================================================\n\n');

%% 1. Load Precomputed Tensor
K_max = 4;
L_max = 4;
vhs_omega = 0.5; % 1.0 corresponds to Maxwell Molecules

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
c_eq = zeros(N_terms, 1);
c_eq(1) = 1.0; 

J_eq = squeeze(C_assembled(:,:,1)) * c_eq(1) + squeeze(C_assembled(:,1,:)) * c_eq(1);

q_stress = 7; 
idx_L2 = (0:K_max) * N_Q + q_stress;
J_L2 = J_eq(idx_L2, idx_L2);

%% 3. Normalize the Numerical Matrix
H_num = J_L2 / J_L2(1,1);

%% 4. The Exact Analytical Chapman-Cowling Matrix (2x2 Block)
% For Isotropic Maxwell Molecules, the Burnett functions are exact 
% eigenfunctions. The off-diagonals vanish entirely.
% The analytical ratio of the first two L=2 eigenvalues is exactly 7/6.
H_ana = [1.0, 0.0; 
         0.0, 7/6];

fprintf('\n--- MATRIX ELEMENT VALIDATION ---\n');
fprintf('Exact Analytical H Matrix (2x2):\n');
disp(H_ana);

fprintf('Numerical Spectral H Matrix (Top 2x2 Block):\n');
disp(H_num(1:2, 1:2));

matrix_error = max(abs(H_num(1:2, 1:2) - H_ana), [], 'all');
fprintf('Maximum Matrix Element Error: %.4e\n\n', matrix_error);

%% 5. Compute "Infinite-Order" Viscosity Correction Factors
fprintf('--- VISCOSITY CORRECTION FACTORS f_mu ---\n');
fprintf('Theoretical Limits (Maxwell Molecules):\n');
fprintf('  1st-Order (1x1) : 1.000000\n');
fprintf('  2nd-Order (2x2) : 1.000000\n');
fprintf('  Inf-Order (inf) : 1.000000\n\n');

fprintf('Numerical Spectral Limits (Increasing K_max):\n');
for k = 1:(K_max+1)
    H_sub = H_num(1:k, 1:k);
    H_inv = inv(H_sub);
    f_mu = H_inv(1,1);
    
    fprintf('  Using K_max = %d (%dx%d matrix): f_mu = %.6f\n', k-1, k, k, f_mu);
end
fprintf('==============================================================\n');

%% 6. EIGENDECOMPOSITION: Extracting the Optimal VHS Eigenfunctions
fprintf('\n--- VHS OPTIMAL RADIAL EIGENFUNCTIONS (L=2) ---\n');

% 1. Solve the matrix eigenvalue problem for the L=2 stress block
[V, D] = eig(J_L2);
evals = diag(D);

% 2. Sort the modes by their relaxation rate (magnitude of eigenvalue)
[~, sort_idx] = sort(abs(evals));
V_sorted = V(:, sort_idx);

% 3. Extract the fundamental stress mode (the slowest decaying L=2 mode)
% We normalize the eigenvector so the k=0 contribution is exactly 1.0. 
% For a Maxwell molecule, all other coefficients would be exactly 0.0.
v_opt = V_sorted(:, 1);
v_opt = v_opt / v_opt(1); 

fprintf('Optimal Eigenvector coefficients (Burnett Basis mixing):\n');
for k = 0:K_max
    fprintf('  k=%d: %10.6f\n', k, v_opt(k+1));
end

% 4. Reconstruct the continuous radial functions
c = linspace(0, 4, 500); % Continuous velocity grid

R_maxwell = zeros(1, 500);
R_vhs     = zeros(1, 500);

for i = 1:length(c)
    % Extract scalar speed to prevent MATLAB matrix expansion
    v = c(i);
    
    % Re-apply the physical equilibrium weight!
    weight = exp(-v^2 / 2); 
    
    % Maxwell is just the pure k=0 basis function
    R_maxwell(i) = Basis.evaluate_radial(1, 2, v) * weight;
    
    % VHS is the superposition of all K states
    for k = 0:K_max
        phi_k = Basis.evaluate_radial(k+1, 2, v) * weight;
        R_vhs(i) = R_vhs(i) + v_opt(k+1) * phi_k;
    end
end

% 5. Visualize the warping
figure('Name', 'Optimal Radial Eigenfunctions (L=2)', 'Color', 'w', 'Position', [100, 100, 700, 600]);

% Top Plot: The absolute functions
subplot(2,1,1);
plot(c, R_maxwell, 'k--', 'LineWidth', 2); hold on;
plot(c, R_vhs, 'r-', 'LineWidth', 2);
title('Fundamental Stress Eigenfunction: Maxwell vs. VHS');
xlabel('Relative Speed (c)');
ylabel('Amplitude R_{0,2}(c)');
legend('Maxwell Molecule (Pure Laguerre)', 'Variable Hard Sphere (Optimal)', 'Location', 'best');
set(gca, 'FontSize', 11);
grid on;

% Bottom Plot: The physical warping (Difference)
subplot(2,1,2);
plot(c, R_vhs - R_maxwell, 'b-', 'LineWidth', 2);
title('The "Warping" Difference (VHS - Maxwell)');
xlabel('Relative Speed (c)');
ylabel('\Delta R(c)');
set(gca, 'FontSize', 11);
grid on;