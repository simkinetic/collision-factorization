%% UNIFIED EXPERIMENT: Three-Level VMS with Two-Level Static Condensation
% Architecture:
% V_O : Coarse Macroscopic Space (K <= K_O)
% V_R : Resolved Intermediate Space (K_O < K <= K_test)
% V'  : Fine Subgrid Space (K_test < K <= K_max)
%
% Condensation Pipeline:
% 1. VMS Condensation: V' -> (V_O U V_R) using Mass Matrix metric
% 2. Static Condensation: V_R -> V_O using exact Schur complement

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL', 'src/precalc');

fprintf('==============================================================\n');
fprintf('  EXPERIMENT: Three-Level VMS vs Direct Galerkin\n');
fprintf('==============================================================\n\n');

%% 1. Configuration & Loading
K_O = 1;            % V_O: Target coarse space limit
L_O = 2;            
K_max = 5;          % V': Absolute fine scale ceiling
L_max = 2;          
gamma = 1.0;        

% Exact values for Hard Spheres
exact_f_mu_vals = [1.000000, 1.014851, 1.015879, 1.016006, 1.016028, 1.016032];

filename = sprintf('collisiontensor_unified_k%d_test%d_l%d_gamma%.2f.mat', K_O, K_max, L_max, gamma);
filepath = fullfile('src', 'precalc', filename);

if ~exist(filepath, 'file')
    fprintf('Tensor not found. Generating...\n');
    Basis = SpectralBasis(K_max, L_max);
    Kernel = ScatteringKernel(gamma);
    TensorObj = GeneralCollisionTensor(Basis, Kernel, K_max);
    TensorObj.generate_R_tensor_sumfac(20, 20);
    if ~exist('src/precalc', 'dir'), mkdir('src/precalc'); end
    save(filepath, 'TensorObj', 'Basis', 'Kernel', '-v7.3');
end

data = load(filepath, 'Basis', 'TensorObj');
Basis = data.Basis; TensorObj = data.TensorObj;
N_trial = Basis.N_terms; N_Q = Basis.N_Q; 
N_total = (K_max + 1) * N_Q;

C_assembled = TensorObj.assemble_full_tensor();

%% 2. Equilibrium State & VMS Metric Assembly
fprintf('Assembling Full VMS Mass Metric...\n');
VMS = VMSClosure(TensorObj, K_max, K_O, L_max);

c_eq = zeros(N_trial, 1); 
c_eq(1) = 1.0;

% Correct radial matrix dimensioning
c_eq_rad_ang = reshape(c_eq, N_Q, K_max+1)'; 
nu_coeffs = zeros(VMS.K_N+1, N_Q);
for l = 0:L_max
    q_idx = (l^2+1):((l+1)^2);
    nu_coeffs(:, q_idx) = VMS.K_mat{l+1} * c_eq_rad_ang(:, q_idx); 
end

M_full = zeros(N_total, N_total);
for g_idx = 1:length(TensorObj.gaunt_vals)
    q1 = TensorObj.gaunt_labels(g_idx, 1); q2 = TensorObj.gaunt_labels(g_idx, 2); q3 = TensorObj.gaunt_labels(g_idx, 3);
    g_val = TensorObj.gaunt_vals(g_idx); t_chan = TensorObj.ic_map(g_idx);
    for k1 = 0:K_max
        for k2 = 0:K_max
            idx1 = k1 * N_Q + q1; idx2 = k2 * N_Q + q2;
            for k_g = 0:VMS.K_N
                if abs(nu_coeffs(k_g+1, q3)) > 1e-15
                    M_full(idx1, idx2) = M_full(idx1, idx2) + ...
                        nu_coeffs(k_g+1, q3) * g_val * VMS.T_tensor(k1+1, k2+1, k_g+1, t_chan);
                end
            end
        end
    end
end

fprintf('Evaluating Global Equilibrium Jacobian...\n');
J_full = squeeze(C_assembled(1:N_total, :, 1)) * c_eq(1) + squeeze(C_assembled(1:N_total, 1, :)) * c_eq(1);

%% 3. Indexing V_O (Coarse Space) & Base Normalization
idx_O = [];
for k = 0:K_O
    for l = 0:L_O
        idx_O = [idx_O, k * N_Q + ((l^2 + 1):((l+1)^2))];
    end
end

J_OO = J_full(idx_O, idx_O);
q_stress = 7;
global_stress_idx_O = (0:K_O) * N_Q + q_stress;
[~, loc_stress_O] = ismember(global_stress_idx_O, idx_O); 

% Unperturbed base collision frequency
base_collision_freq = J_OO(loc_stress_O(1), loc_stress_O(1));

%% 4. Three-Level Resolution Sweeps
k_test_vec = K_O:K_max;
num_steps = length(k_test_vec);

f_mu_gal_direct = zeros(num_steps, 1);
f_mu_vms_3level = zeros(num_steps, 1);

fprintf('\n--- THREE-LEVEL SUBGRID EXPERIMENT ---\n');
fprintf(' K_test | Exact Limit | Galerkin (Direct) | VMS (3-Level)\n');
fprintf('-----------------------------------------------------------\n');

for i = 1:num_steps
    k_test = k_test_vec(i);
    true_exact_val = exact_f_mu_vals(k_test + 1);
    
    % Define V_R: Resolved Intermediate Space
    idx_R = [];
    for k = (K_O + 1):k_test
        for l = 0:L_max
            idx_R = [idx_R, k * N_Q + ((l^2 + 1):((l+1)^2))];
        end
    end
    
    % Define V': Fine Subgrid Space
    idx_F = [];
    for k = (k_test + 1):K_max
        for l = 0:L_max
            idx_F = [idx_F, k * N_Q + ((l^2 + 1):((l+1)^2))];
        end
    end
    
    % C = V_O U V_R (Total Resolved Space for this step)
    idx_C = [idx_O, idx_R];
    N_O = length(idx_O); 
    
    % Properly define stress indices for the FULL resolved space C
    global_stress_idx_C = (0:k_test) * N_Q + q_stress;
    [~, loc_stress_C] = ismember(global_stress_idx_C, idx_C);
    
    % ---------------------------------------------------------------------
    % METHOD A: STANDARD GALERKIN (Operating only on C)
    % ---------------------------------------------------------------------
    J_C = J_full(idx_C, idx_C);
    
    J_C_stress = J_C(loc_stress_C, loc_stress_C);
    H_C_inv = inv(J_C_stress / base_collision_freq);
    f_mu_gal_direct(i) = H_C_inv(1,1);
    
    % ---------------------------------------------------------------------
    % METHOD B: THREE-LEVEL VMS (V' -> C -> V_O)
    % ---------------------------------------------------------------------
    % Stage 1: VMS Condensation (V' -> C)
    if isempty(idx_F)
        J_C_vms = J_C;
    else
        J_FC = J_full(idx_F, idx_C);
        M_CF = M_full(idx_C, idx_F);
        M_FF = M_full(idx_F, idx_F);
        
        J_C_vms = J_C - M_CF * (M_FF \ J_FC);
    end
    
    % Stage 2: Exact Static Condensation (V_R -> V_O)
    if isempty(idx_R)
        J_O_vms = J_C_vms;
    else
        OO = 1:N_O;  RR = (N_O+1):length(idx_C);
        J_OO_vms = J_C_vms(OO, OO);
        J_OR_vms = J_C_vms(OO, RR);
        J_RO_vms = J_C_vms(RR, OO);
        J_RR_vms = J_C_vms(RR, RR);
        
        J_O_vms = J_OO_vms - J_OR_vms * (J_RR_vms \ J_RO_vms);
    end
    
    J_O_vms_stress = J_O_vms(loc_stress_O, loc_stress_O);
    H_O_vms_inv = inv(J_O_vms_stress / base_collision_freq);
    f_mu_vms_3level(i) = H_O_vms_inv(1,1);
    
    fprintf('   %d    |   %.6f  |     %.6f      |   %.6f \n', ...
        k_test, true_exact_val, f_mu_gal_direct(i), f_mu_vms_3level(i));
end
fprintf('===========================================================\n');

%% 5. Plotting
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

fig = figure('Name', 'Three-Level VMS', 'Position', [100, 100, 850, 600], 'Color', 'w');
hold on; grid on;

yline(exact_f_mu_vals(end), 'k--', 'LineWidth', 2, 'DisplayName', 'Infinite-Order Limit ($K_{max}=5$)');
plot(k_test_vec, f_mu_gal_direct, '-s', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.8500, 0.3250, 0.0980], 'DisplayName', 'Galerkin (Direct $V_O \cup V_R$)');
plot(k_test_vec, f_mu_vms_3level, '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'Color', [0.0000, 0.4470, 0.7410], 'DisplayName', '3-Level VMS ($V^\prime \rightarrow V_R \rightarrow V_O$)');

xlabel('Intermediate Resolved Space Limit ($K_{test}$)', 'FontSize', 16);
ylabel('Viscosity Correction Factor $f_\mu$', 'FontSize', 16);
title('\textbf{Three-Level VMS Architecture vs Standard Galerkin}', 'FontSize', 18);
xticks(k_test_vec);
ylim([1.0145, 1.0162]); 
legend('Location', 'northeast', 'FontSize', 12);
set(gca, 'FontSize', 14, 'LineWidth', 1.2);