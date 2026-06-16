%% VALIDATION: VMS vs. Galerkin Viscosity Correction (Chapman-Enskog)
% This script extracts the linearized Jacobian for the L=2 stress mode 
% and computes the Chapman-Cowling viscosity correction factor f_mu.
% It demonstrates that the VMS Schur complement natively recovers 
% high-fidelity transport coefficients even at extremely coarse resolutions.

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL', 'src/precalc');

fprintf('==============================================================\n');
fprintf('  BENCHMARK: Viscosity Correction (Galerkin vs. VMS)\n');
fprintf('==============================================================\n\n');

% Do you want to export to PDF figure?
export_to_pdf_figure = true;

%% 1. Load the Extended VMS Tensor
K_max = 4;
L_max = 4;
K_test = 8; % Large test space to act as 'Ground Truth' for the cascade
gamma = 1.0; % Hard Spheres

filename = sprintf('collisiontensor_vms_k%d_test%d_l%d_gamma%.2f.mat', K_max, K_test, L_max, gamma);
filepath = fullfile('src', 'precalc', filename);

if ~exist(filepath, 'file')
    fprintf('Extended VMS tensor not found. Generating it now (K_max=%d, K_test=%d)...\n', K_max, K_test);
    Basis = SpectralBasis(K_max, L_max);
    Kernel = ScatteringKernel(gamma);
    TensorObj = GeneralCollisionTensor(Basis, Kernel, K_test);
    TensorObj.generate_R_tensor_sumfac(40, 0);
    save(filepath, 'TensorObj', 'Basis', 'Kernel', '-v7.3');
end

fprintf('1. Loading Precomputed VMS Tensor: %s\n', filename);
data = load(filepath, 'Basis', 'TensorObj');
Basis = data.Basis; TensorObj = data.TensorObj;
N_trial = Basis.N_terms; N_Q = Basis.N_Q; N_test = (K_test + 1) * N_Q;

C_assembled = TensorObj.assemble_full_tensor();
C_flat = reshape(C_assembled, N_test, N_trial^2);

%% 2. Initialize VMS Metrics
fprintf('2. Assembling VMS Mass Metric at Equilibrium...\n');
VMS = VMSClosure(TensorObj, K_test, K_max, L_max);

% Compute frequency coefficients at equilibrium (Maxwellian)
c_eq = zeros(N_trial, 1); c_eq(1) = 1.0;
c_eq_rad_ang = reshape(c_eq, N_Q, K_max+1)';
nu_coeffs = zeros(VMS.K_N+1, N_Q);
for l = 0:L_max
    q_idx = (l^2+1):((l+1)^2);
    nu_coeffs(:, q_idx) = VMS.K_mat{l+1} * c_eq_rad_ang(:, q_idx);
end

% Assemble the Full N_test x N_test Mass Matrix
M_full = zeros(N_test, N_test);
for g_idx = 1:length(TensorObj.gaunt_vals)
    q1 = TensorObj.gaunt_labels(g_idx, 1); q2 = TensorObj.gaunt_labels(g_idx, 2); q3 = TensorObj.gaunt_labels(g_idx, 3);
    g_val = TensorObj.gaunt_vals(g_idx); t_chan = TensorObj.ic_map(g_idx);
    for k1 = 0:K_test
        for k2 = 0:K_test
            idx1 = k1 * N_Q + q1; idx2 = k2 * N_Q + q2;
            for k_g = 0:VMS.K_N
                if abs(nu_coeffs(k_g+1, q3)) > 1e-15
                    M_full(idx1, idx2) = M_full(idx1, idx2) + nu_coeffs(k_g+1, q3) * g_val * VMS.T_tensor(k1+1, k2+1, k_g+1, t_chan);
                end
            end
        end
    end
end

%% 3. Extract L=2 Stress Blocks
fprintf('3. Extracting L=2 Stress Jacobians...\n');
J_eq_full = squeeze(C_assembled(1:N_test, :, 1)) * c_eq(1) + squeeze(C_assembled(1:N_test, 1, :)) * c_eq(1);

q_stress = 7; % L=2, m=0 mode
idx_L2_trial = (0:K_max) * N_Q + q_stress;
idx_L2_test = (0:K_test) * N_Q + q_stress;

J_L2 = J_eq_full(idx_L2_test, idx_L2_trial);
M_L2 = M_full(idx_L2_test, idx_L2_test);

%% 4. Compute Viscosity Corrections
f_mu_gal = zeros(K_max+1, 1);
f_mu_vms = zeros(K_max+1, 1);

% Theoretical Infinite-Order Limit for Hard Spheres
f_mu_inf = 1.016034; 

fprintf('\n--- VISCOSITY CORRECTION FACTOR f_mu ---\n');
fprintf('Theoretical Infinite-Order Limit: %.6f\n\n', f_mu_inf);
fprintf(' K_coarse | Standard Galerkin | VMS Stabilized \n');
fprintf('-----------------------------------------------\n');

for k_coarse = 0:K_max
    
    % --- Standard Galerkin ---
    % Truncate directly to coarse resolution
    J_gal = J_L2(1:k_coarse+1, 1:k_coarse+1);
    H_gal = J_gal / J_gal(1,1);
    H_inv_gal = inv(H_gal);
    f_mu_gal(k_coarse+1) = H_inv_gal(1,1);
    
    % --- VMS Stabilized ---
    if k_coarse == K_test
        f_mu_vms(k_coarse+1) = f_mu_gal(k_coarse+1); % No subgrid exists
    else
        % Partition matrices
        J_res = J_L2(1:k_coarse+1, 1:k_coarse+1);
        J_fine_res = J_L2(k_coarse+2:end, 1:k_coarse+1);
        
        M_res_fine = M_L2(1:k_coarse+1, k_coarse+2:end);
        M_fine_fine = M_L2(k_coarse+2:end, k_coarse+2:end);
        
        % VMS Schur Complement!
        J_vms = J_res - M_res_fine * (M_fine_fine \ J_fine_res);
        
        H_vms = J_vms / J_vms(1,1);
        H_inv_vms = inv(H_vms);
        f_mu_vms(k_coarse+1) = H_inv_vms(1,1);
    end
    
    fprintf('    K=%d   |     %.6f      |    %.6f\n', k_coarse, f_mu_gal(k_coarse+1), f_mu_vms(k_coarse+1));
end
fprintf('===============================================\n');

%% 5. Publication-Quality Plot
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

fig = figure('Name', 'Viscosity Convergence', 'Position', [100, 100, 700, 500], 'Color', 'w');
hold on; grid on;

% Plot ground truth limit
yline(f_mu_inf, 'k--', 'LineWidth', 2, 'DisplayName', 'Infinite-Order Limit ($f_\mu^\infty$)');

% Slice data to remove K=0 and isolate the convergence region
k_vec_plot = 1:K_max;
f_mu_gal_plot = f_mu_gal(2:end);
f_mu_vms_plot = f_mu_vms(2:end);

% Plot Galerkin vs VMS
plot(k_vec_plot, f_mu_gal_plot, '-x', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.8500, 0.3250, 0.0980], 'DisplayName', 'Standard Galerkin');
plot(k_vec_plot, f_mu_vms_plot, '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.0000, 0.4470, 0.7410], 'DisplayName', 'VMS Stabilized');

xlabel('Resolved Radial Resolution ($K_{\max}$)', 'FontSize', 16);
ylabel('Viscosity Correction Factor $f_\mu$', 'FontSize', 16);
title('\textbf{Viscosity Convergence: Galerkin vs. VMS}', 'FontSize', 18);

% Tightly frame the y-axis to highlight the from-above and from-below convergence
xticks(k_vec_plot);
ylim([1.0145, 1.0170]); 
yticks(linspace(1.0145, 1.0170, 6));

legend('Location', 'southeast', 'FontSize', 14);
set(gca, 'FontSize', 14, 'LineWidth', 1.2);

if export_to_pdf_figure
    exportgraphics(fig, 'fig_vms_viscosity.pdf', 'ContentType', 'vector');
end
