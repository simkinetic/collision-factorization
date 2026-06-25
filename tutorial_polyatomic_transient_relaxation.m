%% TUTORIAL: Polyatomic Transient Relaxation (Hard Spheres)
% This interactive script demonstrates the mathematical validation of the 
% fully nonlinear spectral collision operator for a polyatomic gas.
%
% It computes the 9D polyatomic collision tensor on-the-fly and simulates
% both the decay of macroscopic anisotropic stress (L=2) and the thermal 
% relaxation between translational and internal energy modes.

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL');

fprintf('==============================================================\n');
fprintf('  BENCHMARK: Polyatomic Nonlinear Relaxation (RK2)\n');
fprintf('==============================================================\n\n');

export_to_pdf_figure = false;

%% 1. Initialize Basis, Kernel, and Compute Tensor On-the-Fly
% Adjust these limits depending on your RAM and patience. 
% For a quick test, K=2, L=2, I=2 is recommended.
K_max = 2;
L_max = 2;
I_max = 1;
nu_val = 1.0; % e.g., D=7 degrees of freedom
gamma = 1.0;  % Hard Spheres

fprintf('1. Initializing Basis and Kernel...\n');
% ---> INSTRUCTION: Replace these with your actual class constructors <---
Basis = SpectralBasis(K_max, L_max, I_max, nu_val); 
Kernel = ScatteringKernel('VHS', gamma);

fprintf('2. Computing 9D Polyatomic Collision Tensor...\n');
TensorObj = GeneralCollisionTensor(Basis, Kernel);

% Compute the continuous tensor using our newly refactored C++ MEX Fubini cascade
radial_pad = 20; angular_pad = 20;
TensorObj.generate_R_tensor_sumfac(radial_pad, angular_pad);

% Assemble the full dense tensor
C_assembled = TensorObj.assemble_full_tensor();
N_terms = Basis.N_terms;
N_Q = Basis.N_Q;

% Flatten for rapid Runge-Kutta matrix-vector multiplication
N_test_terms = size(C_assembled, 1);
C_flat = reshape(C_assembled, N_test_terms, N_terms^2);

%% 3. Index Mapping Helper
% The polyatomic tensor order is: ((k * N_I + i) * N_Q) + q
% Note: k and i are 0-indexed, q is 1-indexed.
N_I = I_max + 1;
idx_fun = @(k, i, q) (k * N_I + i) * N_Q + q;

%% 4. Extract Theoretical Chapman-Enskog Relaxation Rates
% Linearize around the absolute Maxwellian to extract exact eigenvalues.
c_eq = zeros(N_terms, 1);
c_eq(1) = 1.0; % Absolute Maxwellian

J_eq = squeeze(C_assembled(1:N_terms, 1:N_terms, 1)) * c_eq(1) + ...
       squeeze(C_assembled(1:N_terms, 1, 1:N_terms)) * c_eq(1);

% Extract Shear Relaxation Rate (L=2, m=0 -> q=7)
q_stress = 7; 
idx_primary_stress = idx_fun(0, 0, q_stress); 
mu_stress = abs(J_eq(idx_primary_stress, idx_primary_stress));

fprintf('3. Extracted Principal Shear Relaxation Rate (mu_stress): %.6f\n', mu_stress);

%% 5. Simulate the Nonlinear ODE (Custom RK2 Scheme)
fprintf('4. Simulating Nonlinear Relaxation (Custom RK2)...\n');

% Initialize the gas with two simultaneous perturbations:
% A) A massive anisotropic stress
stress_amplitude = 0.10; 

% B) Thermal Non-Equilibrium (T_trans != T_int)
% Excite the K=1, I=0 mode (Translational) and K=0, I=1 mode (Internal)
thermal_amplitude = 0.05; 
idx_T_trans = idx_fun(1, 0, 1);
idx_T_int   = idx_fun(0, 1, 1);

c_init = c_eq;
c_init(idx_primary_stress) = stress_amplitude;
c_init(idx_T_trans) = thermal_amplitude;
c_init(idx_T_int)   = -thermal_amplitude;

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
    Q_k1_full = C_flat * reshape(c_n * c_n', N_terms^2, 1);
    Q_k1 = Q_k1_full(1:N_terms); % Truncate to resolved space
    
    % Step 2: Predictor state
    c_tmp = c_n + dt * Q_k1;
    
    % Step 3: Corrector evaluation (k2)
    Q_k2_full = C_flat * reshape(c_tmp * c_tmp', N_terms^2, 1);
    Q_k2 = Q_k2_full(1:N_terms);
    
    % Step 4: Corrector Update
    c_ode(n+1, :) = (c_n + 0.5 * dt * (Q_k1 + Q_k2))';
end

% Analytical linear decay for reference
c_analytical = zeros(size(c_ode));
c_analytical(:, idx_primary_stress) = stress_amplitude * exp(-mu_stress * t_out);

%% 6. Publication-Quality Visualization
fprintf('5. Simulation Complete. Generating Figures...\n');

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

FS_title  = 20; 
FS_labels = 18; 
FS_ticks = 14; 
FS_legend = 14;

% ========================================================================
% --- FIGURE 1: ANISOTROPIC STRESS & THERMAL RELAXATION ---
% ========================================================================
fig_relax = figure('Name', 'Polyatomic Relaxation', 'Position', [100, 100, 800, 500], 'Color', 'w');
hold on; grid on;

% Plot Stress Decay
h_stress = semilogy(t_out * mu_stress, abs(c_ode(:, idx_primary_stress)), '-', 'Color', '#0072BD', 'LineWidth', 3);
h_ana = semilogy(t_out * mu_stress, abs(c_analytical(:, idx_primary_stress)), 'k--', 'LineWidth', 2);

% Plot Thermal Equilibration (Translational and Internal modes)
h_trans = semilogy(t_out * mu_stress, abs(c_ode(:, idx_T_trans)), '-', 'Color', '#D95319', 'LineWidth', 3);
h_int   = semilogy(t_out * mu_stress, abs(c_ode(:, idx_T_int)), '-', 'Color', '#EDB120', 'LineWidth', 3);

xlabel('Dimensionless Time $\tau = \mu_{\mathrm{stress}} t$', 'FontSize', FS_labels);
ylabel('Absolute Spectral Amplitude $|c_{\alpha}|$', 'FontSize', FS_labels);
title('\textbf{Polyatomic Thermal \& Stress Relaxation}', 'FontSize', FS_title);

set(gca, 'YScale', 'log', 'FontSize', FS_ticks, 'LineWidth', 1.2);
xlim([0, 4]);
ylim([1e-3, 1e-0]);

legend([h_stress, h_ana, h_trans, h_int], ...
       {'Shear Stress $(L=2)$', 'Theoretical Exp. Decay', 'Trans. Temp. Anomaly $(K=1, I=0)$', 'Internal Temp. Anomaly $(K=0, I=1)$'}, ...
       'Location', 'northeast', 'FontSize', FS_legend);

if export_to_pdf_figure
    exportgraphics(fig_relax, 'fig_polyatomic_relaxation.pdf', 'ContentType', 'vector');
end

% ========================================================================
% --- FIGURE 2: MACHINE-PRECISION CONSERVATION ---
% ========================================================================
fig_cons = figure('Name', 'Conservation', 'Position', [950, 100, 600, 400], 'Color', 'w');
hold on; grid on;

% Extract Macroscopic Invariants
idx_mass  = idx_fun(0, 0, 1);
idx_mom_z = idx_fun(0, 0, 3);

mass_err = max(abs(c_ode(:, idx_mass) - c_ode(1, idx_mass)), eps());
mom_err  = max(abs(c_ode(:, idx_mom_z) - c_ode(1, idx_mom_z)), eps());

% Subsample markers
sub = round(linspace(1, length(t_out), 15));

semilogy(t_out * mu_stress, mass_err, 'k-o', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerIndices', sub, 'DisplayName', 'Mass Error $|\Delta \rho|$');
semilogy(t_out * mu_stress, mom_err, 'k-s', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerIndices', sub, 'DisplayName', 'Momentum Error $|\Delta u_z|$');

xlabel('Dimensionless Time $\tau = \mu_{\mathrm{stress}} t$', 'FontSize', FS_labels);
ylabel('Absolute Error $|c(t) - c(0)|$', 'FontSize', FS_labels);
title('\textbf{Null-Space Conservation}', 'FontSize', FS_title);

set(gca, 'YScale', 'log', 'FontSize', FS_ticks, 'LineWidth', 1.2);
ylim([1e-18, 1e-12]); 
yticks(10.^(-18:2:-12));
legend('Location', 'northeast', 'FontSize', FS_legend);

if export_to_pdf_figure
    exportgraphics(fig_cons, 'fig_polyatomic_conservation.pdf', 'ContentType', 'vector');
end