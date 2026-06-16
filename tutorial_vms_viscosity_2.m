%% UNIFIED EXPERIMENT: VMS Subgrid Capacity with Generalized K & L Masking
% This script isolates the VMS mechanism by locking the resolved macroscopic 
% space and progressively expanding the unresolved fine-scale space.
% It uses a fully generalized logical masking approach to allow independent 
% truncation of both Radial (K) and Angular (L) modes.

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL', 'src/precalc');

fprintf('==============================================================\n');
fprintf('  EXPERIMENT: Generalized VMS Capacity (K & L Masking)\n');
fprintf('==============================================================\n\n');

export_to_pdf_figure = true;

%% 1. Configuration & Loading the Extended Tensor
K_res = 2;          % FIXED: Coarse resolved macroscopic grid
L_res = 2;          % FIXED: Macroscopic angular resolution

K_test_max = 8;     % MAXIMUM: The deepest radial subgrid cascade
L_test_max = 3;     % MAXIMUM: Subgrid angular resolution (e.g., L_res + 1)
L_max = max(L_res, L_test_max); 
gamma = 1.0;        % Hard Spheres

filename = sprintf('collisiontensor_vms_k%d_test%d_l%d_gamma%.2f.mat', K_res, K_test_max, L_max, gamma);
filepath = fullfile('src', 'precalc', filename);

% Ensure tensor exists 
if ~exist(filepath, 'file')
    fprintf('Extended tensor not found. Generating it now...\n');
    Basis = SpectralBasis(K_res, L_max);
    Kernel = ScatteringKernel(gamma);
    TensorObj = GeneralCollisionTensor(Basis, Kernel, K_test_max);
    TensorObj.generate_R_tensor_sumfac(20, 20);
    save(filepath, 'TensorObj', 'Basis', 'Kernel', '-v7.3');
end

fprintf('1. Loading Precomputed Tensor: %s\n', filename);
data = load(filepath, 'Basis', 'TensorObj');
Basis = data.Basis; TensorObj = data.TensorObj;

N_trial = Basis.N_terms; N_Q = Basis.N_Q; N_test_total = (K_test_max + 1) * N_Q;
C_assembled = TensorObj.assemble_full_tensor();

%% 2. Initialize VMS Metrics
fprintf('2. Assembling Full VMS Mass Metric at Equilibrium...\n');
VMS = VMSClosure(TensorObj, K_test_max, K_res, L_max);

% Compute frequency coefficients at equilibrium (Maxwellian)
c_eq = zeros(N_trial, 1); c_eq(1) = 1.0;
c_eq_rad_ang = reshape(c_eq, N_Q, K_res+1)';
nu_coeffs = zeros(VMS.K_N+1, N_Q);
for l = 0:L_max
    q_idx = (l^2+1):((l+1)^2);
    nu_coeffs(:, q_idx) = VMS.K_mat{l+1} * c_eq_rad_ang(:, q_idx);
end

% Assemble the Full N_test x N_test Mass Matrix
M_full = zeros(N_test_total, N_test_total);
for g_idx = 1:length(TensorObj.gaunt_vals)
    q1 = TensorObj.gaunt_labels(g_idx, 1); q2 = TensorObj.gaunt_labels(g_idx, 2); q3 = TensorObj.gaunt_labels(g_idx, 3);
    g_val = TensorObj.gaunt_vals(g_idx); t_chan = TensorObj.ic_map(g_idx);
    for k1 = 0:K_test_max
        for k2 = 0:K_test_max
            idx1 = k1 * N_Q + q1; idx2 = k2 * N_Q + q2;
            for k_g = 0:VMS.K_N
                if abs(nu_coeffs(k_g+1, q3)) > 1e-15
                    M_full(idx1, idx2) = M_full(idx1, idx2) + nu_coeffs(k_g+1, q3) * g_val * VMS.T_tensor(k1+1, k2+1, k_g+1, t_chan);
                end
            end
        end
    end
end

%% 3. Evaluate Global Equilibrium Jacobian
fprintf('3. Evaluating Global Full-Scale Jacobian...\n');
J_eq_full = squeeze(C_assembled(1:N_test_total, :, 1)) * c_eq(1) + squeeze(C_assembled(1:N_test_total, 1, :)) * c_eq(1);

%% 4. Compute Viscosity Corrections (Expanding Subgrid Space)
k_fine_vec = K_res:K_test_max;
num_steps = length(k_fine_vec);

f_mu_gal = zeros(num_steps, 1);
f_mu_vms = zeros(num_steps, 1);
f_mu_inf = 1.016034; % Theoretical Infinite-Order Limit

fprintf('\n--- VMS CAPACITY TEST (RESOLVED: K=%d, L=%d) ---\n', K_res, L_res);
fprintf(' Subgrid Depth (K_test) | Standard Galerkin | VMS Stabilized \n');
fprintf('------------------------------------------------------------\n');

% Identify global indices for the FIXED resolved macroscopic space
idx_res = [];
for k = 0:K_res
    for l = 0:L_res
        q_vec = (l^2 + 1):((l+1)^2);
        idx_res = [idx_res, k * N_Q + q_vec];
    end
end

% Standard Galerkin is computed once based on the fixed resolved space
J_res_global = J_eq_full(idx_res, idx_res);

% Locate the L=2, m=0 stress modes WITHIN the resolved block
q_stress = 7;
global_stress_idx = (0:K_res) * N_Q + q_stress;
[~, loc_stress] = ismember(global_stress_idx, idx_res); % Find local positions

J_gal_L2 = J_res_global(loc_stress, loc_stress);
H_gal = J_gal_L2 / J_gal_L2(1,1);
H_inv_gal = inv(H_gal);
gal_fixed_val = H_inv_gal(1,1);

for i = 1:num_steps
    k_test_current = k_fine_vec(i);
    f_mu_gal(i) = gal_fixed_val; % Flatline
    
    if k_test_current == K_res && L_test_max == L_res
        % No subgrid exists yet (Baseline Galerkin)
        f_mu_vms(i) = gal_fixed_val;
    else
        % Dynamically build the fine-scale index mask for this iteration
        idx_fine = [];
        for k = 0:k_test_current
            for l = 0:L_test_max
                q_vec = (l^2 + 1):((l+1)^2);
                global_idx = k * N_Q + q_vec;
                
                % Exclude if it belongs to the resolved space
                if ~(k <= K_res && l <= L_res)
                    idx_fine = [idx_fine, global_idx];
                end
            end
        end
        
        % Partition global matrices using the dynamic masks
        J_fine_res  = J_eq_full(idx_fine, idx_res);
        M_res_fine  = M_full(idx_res, idx_fine);
        M_fine_fine = M_full(idx_fine, idx_fine);
        
        % Compute Global Dynamic Schur Complement
        J_vms_global = J_res_global - M_res_fine * (M_fine_fine \ J_fine_res);
        
        % Extract ONLY the L=2 stress block from the stabilized global matrix
        J_vms_L2 = J_vms_global(loc_stress, loc_stress);
        
        % Compute Viscosity Correction Factor
        H_vms = J_vms_L2 / J_vms_L2(1,1);
        H_inv_vms = inv(H_vms);
        f_mu_vms(i) = H_inv_vms(1,1);
    end
    
    fprintf('          K=%d           |     %.6f      |    %.6f\n', k_test_current, f_mu_gal(i), f_mu_vms(i));
end
fprintf('============================================================\n');

%% 5. Publication-Quality Plot
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

fig = figure('Name', 'VMS Subgrid Capacity', 'Position', [100, 100, 700, 500], 'Color', 'w');
hold on; grid on;

% Plot ground truth limit
yline(f_mu_inf, 'k--', 'LineWidth', 2, 'DisplayName', 'Infinite-Order Limit ($f_\mu^\infty$)');

% Plot Galerkin (Flatline)
plot(k_fine_vec, f_mu_gal, '-x', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.8500, 0.3250, 0.0980], 'DisplayName', sprintf('Standard Galerkin (Locked $K_{\\max}=%d$)', K_res));

% Plot VMS Expansion
plot(k_fine_vec, f_mu_vms, '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.0000, 0.4470, 0.7410], 'DisplayName', 'VMS Stabilized (Expanding Cascade)');

xlabel('Subgrid Test Resolution ($K_{\text{test}}$)', 'FontSize', 16);
ylabel('Viscosity Correction Factor $f_\mu$', 'FontSize', 16);
title('\textbf{VMS Asymptotic Stabilization}', 'FontSize', 18);

xticks(k_fine_vec);
ylim([1.0145, 1.0170]); 
yticks(linspace(1.0145, 1.0170, 6));
legend('Location', 'southeast', 'FontSize', 14);
set(gca, 'FontSize', 14, 'LineWidth', 1.2);

if export_to_pdf_figure
    exportgraphics(fig, 'fig_vms_subgrid_capacity_generalized.pdf', 'ContentType', 'vector');
end