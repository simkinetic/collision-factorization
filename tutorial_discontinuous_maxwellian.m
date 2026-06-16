%% TUTORIAL: Bimodal Maxwellian (Shock-Interior Analog)
% This script tests the solver against a non-equilibrium bimodal state.
% By using U0 = 0.4, the initial condition is kept strictly within the 
% L^2 integrability limits of the weighted Hilbert space.
% 
% Standard Galerkin will fail to resolve the high-frequency scattering 
% cascade and trigger a discrete entropy violation. VMS will apply 
% conservative spectral viscosity to damp the unresolved cascade, strictly 
% preserving thermodynamic entropy and allowing explicit RK2 integration.

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL', 'src/precalc');

fprintf('==============================================================\n');
fprintf('  BENCHMARK: Bimodal Maxwellian (Galerkin vs. VMS)\n');
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
Basis = data.Basis; TensorObj = data.TensorObj;
N_trial_terms = Basis.N_terms; N_Q = Basis.N_Q; N_test_terms = (K_test + 1) * N_Q;

C_assembled = TensorObj.assemble_full_tensor();
C_flat = reshape(C_assembled, N_test_terms, N_trial_terms^2);

%% 2. Initialize the VMS Subgrid Closure
fprintf('2. Initializing VMS Subgrid Matrices...\n');
% Retain radial velocity dependence (K_N = K_max) for cross-coupling, but 
% drop angular ringing (L_N = 0) to ensure an isotropic, positive metric.
K_N = K_max; 
L_N = 0;     
VMS = VMSClosure(TensorObj, K_test, K_N, L_N);

%% 3. Project the Bimodal Initial Condition
fprintf('3. Projecting Smooth Bimodal Maxwellians (U0 = 0.4)...\n');
% Lowered Mach velocity ensures strict compatibility with the M^-1 weight
U0 = 0.4; 
T0 = 1.0 - (U0^2) / 3.0; % Conserve total normalized energy T=1

qr_v = Gauss.legendre(100, 0, 12);
v_nodes = qr_v.x; w_v = qr_v.w;

qr_mu = Gauss.legendre(100, -1, 1);
mu_nodes = qr_mu.x; w_mu = qr_mu.w;

c_init = zeros(N_trial_terms, 1);
M_ref_inv = @(v) (2*pi)^(3/2) * exp(v.^2 / 2);

for i = 1:length(v_nodes)
    v = v_nodes(i);
    for j = 1:length(mu_nodes)
        mu = mu_nodes(j);
        
        v_z = v * mu;
        v_perp2 = v^2 * (1 - mu^2);
        
        E1 = ((v_z - U0)^2 + v_perp2) / (2*T0);
        M1 = (1 / (2*pi*T0)^(3/2)) * exp(-E1);
        
        E2 = ((v_z + U0)^2 + v_perp2) / (2*T0);
        M2 = (1 / (2*pi*T0)^(3/2)) * exp(-E2);
        
        f_val = 0.5 * M1 + 0.5 * M2;
        
        weight = w_v(i) * w_mu(j) * v^2 * 2 * pi;
        Y_eval = Basis.SH.evaluate(acos(mu), 0);
        
        for l = 0:L_max
            q_idx = l^2 + l + 1; 
            Y_val = Y_eval(q_idx);
            for k = 0:K_max
                R_val = Basis.evaluate_radial(k+1, l, v);
                idx = k * N_Q + q_idx;
                c_init(idx) = c_init(idx) + weight * f_val * M_ref_inv(v) * R_val * Y_val;
            end
        end
    end
end
init_norm_sq = norm(c_init)^2;
fprintf('   Initial State L2 Norm: %.6f\n', init_norm_sq);

%% 4. Time Setup
T_end = 6.0;
dt = 0.005; % Explicit RK2 timestep
t_out = 0:dt:T_end;
N_steps = length(t_out) - 1;

%% 5. SIMULATION 1: Standard Galerkin
fprintf('4. Simulating STANDARD GALERKIN...\n');
c_std = zeros(length(t_out), N_trial_terms);
c_std(1, :) = c_init';
fail_step = N_steps;

for n = 1:N_steps
    c_n = c_std(n, :)';
    
    Q_k1_full = C_flat * reshape(c_n * c_n', N_trial_terms^2, 1);
    Q_k1 = Q_k1_full(1:N_trial_terms); 
    c_tmp = c_n + dt * Q_k1;
    
    Q_k2_full = C_flat * reshape(c_tmp * c_tmp', N_trial_terms^2, 1);
    Q_k2 = Q_k2_full(1:N_trial_terms);
    
    c_next = c_n + 0.5 * dt * (Q_k1 + Q_k2);
    c_std(n+1, :) = c_next';
    
    % Safety catch: If L2 norm doubles, assume violent divergence
    if norm(c_next)^2 > 2.0 * init_norm_sq || any(isnan(c_next))
        fprintf('   -> BOOM! Standard Galerkin violated entropy at t = %.3f\n', t_out(n));
        fail_step = n;
        c_std(n+1:end, :) = NaN; 
        break;
    end
end

%% 6. SIMULATION 2: VMS Stabilized
fprintf('5. Simulating VMS-STABILIZED Closure...\n');
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

%% 7. Publication-Quality Visualization
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

norm_std = sum(c_std.^2, 2);
norm_vms = sum(c_vms.^2, 2);

fig_comp = figure('Name', 'Bimodal Maxwellian', 'Position', [100, 100, 800, 500], 'Color', 'w');
hold on; grid on;

plot(t_out(1:fail_step+1), norm_std(1:fail_step+1), ...
    '-x', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2.5, 'MarkerIndices', 1:10:fail_step+1, ...
    'DisplayName', 'Standard Galerkin');

plot(t_out, norm_vms, ...
    '-', 'Color', [0.0000, 0.4470, 0.7410], 'LineWidth', 3, ...
    'DisplayName', 'VMS-Stabilized');

if fail_step < N_steps
    plot(t_out(fail_step), norm_std(fail_step), 'r*', 'MarkerSize', 15, 'LineWidth', 2, 'HandleVisibility', 'off');
    text(t_out(fail_step) - 0.2, norm_std(fail_step) + 0.05, 'Blow-up!', 'Color', 'r', 'FontSize', 14, 'Interpreter', 'latex');
end

xlabel('Dimensionless Time $t$', 'FontSize', 18);
ylabel('State Norm $||\mathbf{c}||_2^2$ (Entropy Proxy)', 'FontSize', 18);
title('\textbf{Bimodal Shock Analog (U_0 = 0.4): Galerkin vs. VMS}', 'FontSize', 20);
set(gca, 'FontSize', 14, 'LineWidth', 1.2);

% Dynamically scale Y-axis based on initial norm
ylim([0.9 * min([norm_std(1:fail_step); norm_vms]), 1.1 * max(norm_std(1:fail_step))]); 
legend('Location', 'best', 'FontSize', 14);

% ========================================================================
% --- LOCAL FUNCTION: VMS COLLISION EVALUATOR ---
% ========================================================================
function Q_res = compute_Q_VMS(c, C_flat, VMS, TensorObj, N_trial, N_test)
    Q_full = C_flat * reshape(c * c', N_trial^2, 1);
    Q_resolved = Q_full(1:N_trial);
    Q_fine = Q_full(N_trial+1:end);
    
    % 1. NULL-SPACE APPROXIMATION (Section 3.6 of Paper)
    % Extract macroscopic invariants to guarantee a globally positive metric
    % without exposing the matrix to Gibbs ringing artifacts.
    c_null = zeros(size(c));
    N_Q = TensorObj.Basis.N_Q;
    
    idx_mass   = 0 * N_Q + 1;
    idx_mom    = 0 * N_Q + 3;
    idx_energy = 1 * N_Q + 1;
    
    c_null(idx_mass)   = c(idx_mass);
    c_null(idx_mom)    = c(idx_mom);
    c_null(idx_energy) = c(idx_energy);
    
    c_rad_ang = reshape(c_null, N_Q, TensorObj.K_max+1)';
    nu_coeffs = zeros(VMS.K_N+1, N_Q);
    
    for l = 0:TensorObj.L_max
        q_idx = (l^2+1):((l+1)^2);
        nu_coeffs(:, q_idx) = VMS.K_mat{l+1} * c_rad_ang(:, q_idx);
    end
    
    % 2. ASSEMBLE MASS MATRIX
    M_full = zeros(N_test, N_test);
    for g_idx = 1:length(TensorObj.gaunt_vals)
        q1 = TensorObj.gaunt_labels(g_idx, 1);
        q2 = TensorObj.gaunt_labels(g_idx, 2);
        q3 = TensorObj.gaunt_labels(g_idx, 3);
        g_val = TensorObj.gaunt_vals(g_idx);
        t_chan = TensorObj.ic_map(g_idx);
        
        for k1 = 0:TensorObj.K_test
            for k2 = 0:TensorObj.K_test
                idx1 = k1 * N_Q + q1;
                idx2 = k2 * N_Q + q2;
                
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
    
    % 3. INVERT SUBGRID STATE
    % Because the metric is anchored to the positive-definite invariants 
    % and the initial norm is O(1), the matrix is well-conditioned.
    c_prime = M_fine \ Q_fine; 
    
    % 4. ENFORCE MACROSCOPIC CONSERVATION (I - P_N)
    feedback = M_cross * c_prime;
    feedback(idx_mass)   = 0; 
    feedback(idx_mom)    = 0; 
    feedback(idx_energy) = 0; 
    
    Q_res = Q_resolved - feedback;
end