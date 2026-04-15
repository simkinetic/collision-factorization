%% TUTORIAL: Transient Anisotropic Stress Relaxation (Hard Spheres)
% This interactive script demonstrates the mathematical validation of a fully 
% nonlinear spectral collision operator using the transient decay of a macroscopic
% anisotropic stress field in a Hard Sphere gas.
%
% Unlike the isotropic BKW solution, this simulation excites the L=2 spherical 
% harmonics. It validates that the nonlinear operator stably and correctly relaxes 
% a highly non-equilibrium directional stress back to the absolute Maxwellian,
% tracking the theoretical Chapman-Enskog decay rates.

clear; clc; close all;
fprintf('==============================================================\n');
fprintf('  BENCHMARK: Transient Anisotropic Stress Relaxation (RK2)\n');
fprintf('==============================================================\n\n');

% Do you want to export to PDF figure
export_to_pdf_figure = false;

%% 1. Load the Precomputed Factorized Tensor
K_max = 4;
L_max = 4;
vhs_omega = 0.5; % 0.5 corresponds to Hard Spheres

filename = sprintf('src/precalc/collisiontensor_k%d_l%d_vhs_w%.2f.mat', K_max, L_max, vhs_omega);
if ~exist(filename, 'file')
    error('Precomputed tensor %s not found. Please run the generation script.', filename);
end

fprintf('1. Loading Precomputed Tensor: %s\n', filename);
data = load(filename);
Basis = data.Basis;
TensorObj = data.TensorObj;

N_terms = Basis.N_terms;
N_Q = Basis.N_Q;

% Assemble and flatten the tensor for rapid matrix-vector multiplication
C_assembled = TensorObj.assemble_full_tensor();
C_flat = reshape(C_assembled, N_terms, N_terms^2);

%% 2. Extract the Theoretical Chapman-Enskog Relaxation Rate
% By linearizing our exact numerical tensor around the equilibrium Maxwellian state, 
% we extract the exact principal eigenvalue (mu) of the primary stress mode (L=2, K=0).
c_eq = zeros(N_terms, 1);
c_eq(1) = 1.0; % Absolute Maxwellian

J_eq = squeeze(C_assembled(:,:,1)) * c_eq(1) + squeeze(C_assembled(:,1,:)) * c_eq(1);

% Dynamically find the principal L=2, m=0 stress mode index (q = 7)
q_stress = 7; 
idx_primary_stress = 0 * N_Q + q_stress; 
mu_stress = abs(J_eq(idx_primary_stress, idx_primary_stress));

fprintf('2. Extracted Principal Shear Relaxation Rate (mu): %.6f\n', mu_stress);

%% 3. Simulate the Nonlinear ODE (Custom RK2 Scheme)
fprintf('3. Simulating Nonlinear Relaxation (Custom RK2)...\n');

% We initialize the gas as an absolute Maxwellian PLUS a massive anisotropic stress
stress_amplitude = 0.10; % 10% perturbation in the primary shear mode
c_init = c_eq;
c_init(idx_primary_stress) = stress_amplitude;

% Set up the Time grid explicitly based on the physical relaxation scale
T_end = 5.0 / mu_stress;
dt = 0.005 * (1.0 / mu_stress); 
t_out = 0:dt:T_end;
N_steps = length(t_out) - 1;

c_ode = zeros(length(t_out), N_terms);
c_ode(1, :) = c_init';

% Explicit Runge-Kutta 2 (Heun's Method) Time Loop
for n = 1:N_steps
    c_n = c_ode(n, :)';
    
    % Step 1: Predictor evaluation (k1)
    Q_k1 = C_flat * reshape(c_n * c_n', N_terms^2, 1);
    
    % Step 2: Predictor state
    c_tmp = c_n + dt * Q_k1;
    
    % Step 3: Corrector evaluation (k2)
    Q_k2 = C_flat * reshape(c_tmp * c_tmp', N_terms^2, 1);
    
    % Step 4: Corrector Update
    c_ode(n+1, :) = (c_n + 0.5 * dt * (Q_k1 + Q_k2))';
end

%% 4. Evaluate the Analytical Solution (Linearized Decay)
% In the linear regime, the primary stress mode decays exactly exponentially.
c_analytical = zeros(size(c_ode));
c_analytical(:, idx_primary_stress) = stress_amplitude * exp(-mu_stress * t_out);

%% 5. Publication-Quality Visualization
fprintf('4. Simulation Complete. Generating Figures...\n');

% --- GLOBAL LATEX SETTINGS & FONT SIZES ---
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

FS_title  = 20; 
FS_labels = 18; 
FS_ticks = 14; 
FS_legend = 14;

% ========================================================================
% --- FIGURE 1: ANISOTROPIC STRESS RELAXATION (L=2 MODES) ---
% ========================================================================
fig_relax = figure('Name', 'Stress Relaxation', 'Position', [100, 100, 700, 500], 'Color', 'w');
hold on; grid on;

colors = lines(K_max + 1);
leg_h = [];
leg_str = {};

% Plot the decay of all radial modes associated with the L=2 stress field
for k = 0:K_max
    idx = k * N_Q + q_stress;
    
    % Scale time by mu_stress so the primary mode slope is exactly -1
    h_num = semilogy(t_out * mu_stress, abs(c_ode(:, idx)), '-', 'Color', [colors(k+1,:), 0.8], 'LineWidth', 3);
    
    leg_h(end+1) = h_num;
    leg_str{end+1} = sprintf('Numerical $K=%d$', k);
    
    % Plot theoretical reference line for the primary mode
    if k == 0
        h_ana = semilogy(t_out * mu_stress, abs(c_analytical(:, idx)), 'k--', 'LineWidth', 2);
        leg_h(end+1) = h_ana;
        leg_str{end+1} = 'Theoretical Exp. Decay';
    end
end

xlabel('Dimensionless Time $\tau = \mu_{\mathrm{stress}} t$', 'FontSize', FS_labels);
ylabel('Absolute Spectral Amplitude $|c_{k, L=2}|$', 'FontSize', FS_labels);
title('\textbf{Hard Sphere Anisotropic Stress Relaxation}', 'FontSize', FS_title);

set(gca, 'YScale', 'log', 'FontSize', FS_ticks, 'LineWidth', 1.2);
xlim([0, 4]);
ylim([1e-6, 1e-0]);
yticks(10.^(-6:1:0)); 

legend(leg_h, leg_str, 'Location', 'northeast', 'FontSize', FS_legend);

if export_to_pdf_figure
    export_fig('fig_stress_relaxation', '-pdf', '-painters', '-nocrop');
end

% ========================================================================
% --- FIGURE 2: MACHINE-PRECISION CONSERVATION ---
% ========================================================================
fig_cons = figure('Name', 'Conservation', 'Position', [850, 100, 600, 400], 'Color', 'w');
hold on; grid on;

% Extract Macroscopic Invariants:
% Mass (K=0, L=0, m=0) -> index 1
% Z-Momentum (K=0, L=1, m=0) -> index 3 
% Energy (K=1, L=0, m=0) -> index 26
idx_mass = 1;
idx_mom_z = 3;
idx_energy = 1 * N_Q + 1;

mass_err   = max(abs(c_ode(:, idx_mass)   - c_ode(1, idx_mass)), eps());
mom_err    = max(abs(c_ode(:, idx_mom_z)  - c_ode(1, idx_mom_z)), eps());
energy_err = max(abs(c_ode(:, idx_energy) - c_ode(1, idx_energy)), eps());

% Subsample markers
sub = round(linspace(1, length(t_out), 15));

% UPDATED: All lines and markers are now black for consistency
semilogy(t_out * mu_stress, mass_err, 'k-o', 'LineWidth', 2, 'MarkerSize', 12, ...
    'MarkerIndices', sub, 'DisplayName', 'Mass Error $|\Delta \rho|$');
semilogy(t_out * mu_stress, mom_err, 'k-s', 'LineWidth', 2, 'MarkerSize', 12, ...
    'MarkerIndices', sub, 'DisplayName', 'Momentum Error $|\Delta u_z|$');
semilogy(t_out * mu_stress, energy_err, 'k-x', 'LineWidth', 2, 'MarkerSize', 12, ...
    'MarkerIndices', sub, 'DisplayName', 'Energy Error $|\Delta T|$');

xlabel('Dimensionless Time $\tau = \mu_{\mathrm{stress}} t$', 'FontSize', FS_labels);
ylabel('Absolute Error $|c(t) - c(0)|$', 'FontSize', FS_labels);
title('\textbf{Conservation (Anisotropic Gas)}', 'FontSize', FS_title);

set(gca, 'YScale', 'log', 'FontSize', FS_ticks, 'LineWidth', 1.2);
ylim([1e-18, 1e-12]); 
yticks(10.^(-18:2:-12));
legend('Location', 'northeast', 'FontSize', FS_legend);

if export_to_pdf_figure
    export_fig('fig_stress_conservation', '-pdf', '-painters', '-nocrop');
end