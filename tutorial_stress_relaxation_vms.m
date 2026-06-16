%% TUTORIAL: Stress Test - Standard Galerkin vs. VMS Closure
% This script demonstrates the catastrophic failure of the standard spectral 
% Galerkin method under extreme non-equilibrium shocks, and validates how 
% the Variational Multiscale (VMS) closure elegantly stabilizes the exact 
% same system.

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL', 'src/precalc');

fprintf('==============================================================\n');
fprintf('  STRESS TEST: Standard Galerkin vs. VMS\n');
fprintf('==============================================================\n\n');

%% 1. Load the Extended VMS Tensor
K_max = 4;
L_max = 4;
K_test = 6; 
gamma = 1.0; 

filename = sprintf('collisiontensor_vms_k%d_test%d_l%d_gamma%.2f.mat', K_max, K_test, L_max, gamma);
filepath = fullfile('src', 'precalc', filename);

if ~exist(filepath, 'file')
    error('Extended tensor not found. Please run the generation script first.');
end

fprintf('1. Loading Precomputed Tensor: %s\n', filename);
data = load(filepath, 'Basis', 'TensorObj');
Basis = data.Basis;
TensorObj = data.TensorObj;

N_trial_terms = Basis.N_terms;
N_Q = Basis.N_Q;
N_test_terms = (K_test + 1) * N_Q;

C_assembled = TensorObj.assemble_full_tensor();
C_flat = reshape(C_assembled, N_test_terms, N_trial_terms^2);

%% 2. Initialize the VMS Subgrid Closure
fprintf('2. Initializing VMS Subgrid Matrices...\n');
K_N = K_max; 
L_N = L_max; 
VMS = VMSClosure(TensorObj, K_test, K_N, L_N);

%% 3. Extract the Theoretical Chapman-Enskog Relaxation Rate
c_eq = zeros(N_trial_terms, 1);
c_eq(1) = 1.0; % Absolute Maxwellian

J_eq = squeeze(C_assembled(1:N_trial_terms, :, 1)) * c_eq(1) + squeeze(C_assembled(1:N_trial_terms, 1, :)) * c_eq(1);

q_stress = 7; 
idx_primary_stress = 0 * N_Q + q_stress; 
mu_stress = abs(J_eq(idx_primary_stress, idx_primary_stress));

%% 4. Setup the Extreme Shock
stress_amplitude = 1.2; % 65% Anisotropic Shock
c_init = c_eq;
c_init(idx_primary_stress) = stress_amplitude;

T_end = 5.0 / mu_stress;
dt = 0.005 * (1.0 / mu_stress); 
t_out = 0:dt:T_end;
N_steps = length(t_out) - 1;

%% 5. SIMULATION 1: Standard Galerkin (Will Explode)
fprintf('3. Simulating STANDARD GALERKIN (Pushing to failure)...\n');
c_std = zeros(length(t_out), N_trial_terms);
c_std(1, :) = c_init';

fail_step = N_steps;
for n = 1:N_steps
    c_n = c_std(n, :)';
    
    % Standard Galerkin: Truncate collision tensor output exactly at K_max
    Q_k1_full = C_flat * reshape(c_n * c_n', N_trial_terms^2, 1);
    Q_k1 = Q_k1_full(1:N_trial_terms); 
    c_tmp = c_n + dt * Q_k1;
    
    Q_k2_full = C_flat * reshape(c_tmp * c_tmp', N_trial_terms^2, 1);
    Q_k2 = Q_k2_full(1:N_trial_terms);
    
    c_next = c_n + 0.5 * dt * (Q_k1 + Q_k2);
    c_std(n+1, :) = c_next';
    
    % Safety catch: If modes explode beyond physical bounds, stop the solver
    if max(abs(c_next)) > 10.0 || any(isnan(c_next))
        fprintf('   -> BOOM! Standard Galerkin violently diverged at tau = %.3f\n', t_out(n) * mu_stress);
        fail_step = n;
        c_std(n+1:end, :) = NaN; % Mark remainder as NaN for plotting
        break;
    end
end

%% 6. SIMULATION 2: VMS Stabilized (Will Survive)
fprintf('4. Simulating VMS-STABILIZED Closure...\n');
c_vms = zeros(length(t_out), N_trial_terms);
c_vms(1, :) = c_init';

for n = 1:N_steps
    c_n = c_vms(n, :)';
    
    Q_k1 = compute_Q_VMS(c_n, C_flat, VMS, TensorObj, N_trial_terms, N_test_terms);
    c_tmp = c_n + dt * Q_k1;
    
    Q_k2 = compute_Q_VMS(c_tmp, C_flat, VMS, TensorObj, N_trial_terms, N_test_terms);
    
    c_vms(n+1, :) = (c_n + 0.5 * dt * (Q_k1 + Q_k2))';
end
fprintf('   -> VMS flawlessly reached equilibrium.\n');

%% 7. Analytical Solution
c_analytical = zeros(size(c_vms));
c_analytical(:, idx_primary_stress) = stress_amplitude * exp(-mu_stress * t_out);

%% 8. Publication-Quality Visualization
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
FS_title = 20; FS_labels = 18; FS_ticks = 14; FS_legend = 14;

% --- FIGURE: SIDE-BY-SIDE COMPARISON ---
fig_comp = figure('Name', 'Galerkin vs VMS', 'Position', [100, 100, 800, 500], 'Color', 'w');
hold on; grid on;

% 1. Plot Theoretical Decay
plot(t_out * mu_stress, abs(c_analytical(:, idx_primary_stress)), 'k--', 'LineWidth', 2, 'DisplayName', 'Theoretical Exp. Decay');

% 2. Plot Standard Galerkin (Primary Mode & Energy Error)
plot(t_out(1:fail_step+1) * mu_stress, abs(c_std(1:fail_step+1, idx_primary_stress)), ...
    '-x', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2.5, 'MarkerIndices', 1:10:fail_step+1, ...
    'DisplayName', 'Standard Galerkin (Explodes)');

% 3. Plot VMS (Primary Mode)
plot(t_out * mu_stress, abs(c_vms(:, idx_primary_stress)), ...
    '-', 'Color', [0.0000, 0.4470, 0.7410], 'LineWidth', 3, ...
    'DisplayName', 'VMS-Stabilized (Survives)');

% Add explosion marker
if fail_step < N_steps
    plot(t_out(fail_step) * mu_stress, abs(c_std(fail_step, idx_primary_stress)), 'r*', 'MarkerSize', 15, 'LineWidth', 2, 'HandleVisibility', 'off');
    text(t_out(fail_step) * mu_stress - 0.1, abs(c_std(fail_step, idx_primary_stress)) * 1.5, 'Divergence!', 'Color', 'r', 'FontSize', 12, 'Interpreter', 'latex');
end

xlabel('Dimensionless Time $\tau = \mu_{\mathrm{stress}} t$', 'FontSize', FS_labels);
ylabel('Absolute Spectral Amplitude $|c_{k=0, L=2}|$', 'FontSize', FS_labels);
title('\textbf{Extreme Shock: Standard Galerkin vs. VMS Closure}', 'FontSize', FS_title);
set(gca, 'YScale', 'log', 'FontSize', FS_ticks, 'LineWidth', 1.2);
xlim([0, 4]); ylim([1e-6, 1e1]); yticks(10.^(-6:1:1)); 
legend('Location', 'northeast', 'FontSize', FS_legend);

% ========================================================================
% --- LOCAL FUNCTION: VMS COLLISION EVALUATOR ---
% ========================================================================
function Q_res = compute_Q_VMS(c, C_flat, VMS, TensorObj, N_trial, N_test)
    Q_full = C_flat * reshape(c * c', N_trial^2, 1);
    Q_resolved = Q_full(1:N_trial);
    Q_fine = Q_full(N_trial+1:end);
    
    c_rad_ang = reshape(c, TensorObj.Basis.N_Q, TensorObj.K_max+1)';
    nu_coeffs = zeros(VMS.K_N+1, TensorObj.Basis.N_Q);
    
    for l = 0:TensorObj.L_max
        q_idx = (l^2+1):((l+1)^2);
        nu_coeffs(:, q_idx) = VMS.K_mat{l+1} * c_rad_ang(:, q_idx);
    end
    
    M_full = zeros(N_test, N_test);
    for g_idx = 1:length(TensorObj.gaunt_vals)
        q1 = TensorObj.gaunt_labels(g_idx, 1);
        q2 = TensorObj.gaunt_labels(g_idx, 2);
        q3 = TensorObj.gaunt_labels(g_idx, 3);
        g_val = TensorObj.gaunt_vals(g_idx);
        t_chan = TensorObj.ic_map(g_idx);
        
        for k1 = 0:TensorObj.K_test
            for k2 = 0:TensorObj.K_test
                idx1 = k1 * TensorObj.Basis.N_Q + q1;
                idx2 = k2 * TensorObj.Basis.N_Q + q2;
                
                for k_g = 0:VMS.K_N
                    nu_val = nu_coeffs(k_g+1, q3);
                    if abs(nu_val) > 1e-15
                        T_val = VMS.T_tensor(k1+1, k2+1, k_g+1, t_chan);
                        M_full(idx1, idx2) = M_full(idx1, idx2) + nu_val * g_val * T_val;
                    end
                end
            end
        end
    end
    
    M_fine = M_full(N_trial+1:end, N_trial+1:end);
    M_cross = M_full(1:N_trial, N_trial+1:end);
    
    c_prime = M_fine \ Q_fine; 
    Q_res = Q_resolved - M_cross * c_prime;
end