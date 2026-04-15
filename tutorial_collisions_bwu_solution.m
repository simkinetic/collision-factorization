%% TUTORIAL: Validating the Spectral Boltzmann Operator against the BKW Exact Solution
% This interactive script demonstrates the mathematical validation of a fully 
% nonlinear spectral collision operator using the Bobylev-Krook-Wu (BKW) exact solution.
%
% The BKW mode is an exact analytical solution to the fully non-linear Boltzmann 
% equation for a spatially homogeneous gas of Maxwellian molecules. It starts as a 
% highly non-equilibrium "pulsating" distribution and perfectly relaxes into a 
% standard Maxwellian.
clear; clc; close all;

fprintf('==============================================================\n');
fprintf('  TUTORIAL: The Bobylev-Krook-Wu (BKW) Exact Solution (RK2)\n');
fprintf('==============================================================\n\n');

% Do you want to export to PDF figure
export_to_pdf_figure = false;

%% 1. Initialize the Isotropic Phase Space
% Because the BKW solution is perfectly spherically symmetric (isotropic), 
% it depends only on the velocity magnitude, not the direction. 
% In our spherical harmonic basis, this means we only need to evaluate L_max = 0.
K_max = 5;  % Number of radial Laguerre polynomials
L_max = 0;  % Restrict to isotropic modes only
Basis = SpectralBasis(K_max, L_max);
N_terms = Basis.N_terms;

fprintf('1. Building the Exact Numerical Collision Tensor (K=%d, L=0)...\n', K_max);

% We use Maxwellian molecules (omega = 1.0 -> alpha = 0.0), meaning the 
% collision cross-section is independent of the relative velocity 'u'.
alpha = 0.0;
N_max = 5;
tol   = 1e-6;

% Generate the scattering kernel and exact collision tensor
Kernel = ScatteringKernel(K_max, L_max, N_max, alpha, tol);

TensorObj = GeneralCollisionTensor(Basis, Kernel);
TensorObj.generate_R_tensor(true);

% Assemble and flatten the tensor for rapid matrix-vector multiplication
C_assembled = TensorObj.assemble_full_tensor();
C_flat = reshape(C_assembled, N_terms, N_terms^2);

%% 2. Extract the Exact Physical Relaxation Rate (mu)
% The entire nonlinear relaxation process is governed by a single timescale. 
% By linearizing our exact numerical tensor around the equilibrium Maxwellian state, 
% we can extract the exact numerical eigenvalue (mu) of the K=2 stress mode.
c_eq = project_bkw(0.0, K_max, Basis); % True Maxwellian Base State (alpha = 0)

% Compute the Jacobian of the collision operator at equilibrium
J_eq = squeeze(C_assembled(:,:,1)) * c_eq(1) + squeeze(C_assembled(:,1,:)) * c_eq(1);

% Dynamically find the K=2, L=0 mode index
idx_K2 = 2 * Basis.N_Q + 1; 
mu = abs(J_eq(idx_K2, idx_K2));

fprintf('2. Extracted BKW Relaxation Rate (mu): %.6f\n', mu);

%% 3. Simulate the Nonlinear ODE (Custom RK2 Scheme)
% We initialize the gas in a highly non-equilibrium state defined by alpha_0.
alpha_0 = 0.25;
c_init = project_bkw(alpha_0, K_max, Basis);

fprintf('3. Simulating Nonlinear Relaxation (Custom RK2)...\n');

% Set up the Time grid explicitly based on the physical relaxation scale
T_end = 4.0 / mu;
dt = 0.001 * (1.0 / mu); 
t_out = 0:dt:T_end;
N_steps = length(t_out) - 1;

c_ode = zeros(length(t_out), N_terms);
c_ode(1, :) = c_init';

% Explicit Runge-Kutta 2 (Heun's Method) Time Loop
% This explicit formulation is heavily optimized for MATLAB's BLAS backend.
for n = 1:N_steps
    c_n = c_ode(n, :)';
    
    % Step 1: Predictor evaluation (k1)
    % The collision operator is strictly quadratic: Q(c,c) = C * (c x c)
    Q_k1 = C_flat * reshape(c_n * c_n', N_terms^2, 1);
    
    % Step 2: Predictor state
    c_tmp = c_n + dt * Q_k1;
    
    % Step 3: Corrector evaluation (k2)
    Q_k2 = C_flat * reshape(c_tmp * c_tmp', N_terms^2, 1);
    
    % Step 4: Corrector Update
    c_ode(n+1, :) = (c_n + 0.5 * dt * (Q_k1 + Q_k2))';
end

%% 4. Evaluate the Analytical Solution
% We evaluate the exact continuous BKW theory to compare against our simulation.
c_analytical = zeros(size(c_ode));
for i = 1:length(t_out)
    % CRITICAL PHYSICS NOTE: 
    % While the physical stress mode c_2(t) decays at the linear rate exp(-mu * t), 
    % the internal BKW parameter alpha(t) scales as the square root of the mode. 
    % Therefore, alpha decays at exactly HALF the rate: exp(-(mu/2) * t).
    alpha_t = alpha_0 * exp(-(mu / 2) * t_out(i)); 
    c_analytical(i, :) = project_bkw(alpha_t, K_max, Basis)';
end

%% 5. Publication-Quality Visualization (BKW Validation)
fprintf('4. Simulation Complete. Generating Figures...\n');

% --- GLOBAL LATEX SETTINGS & FONT SIZES ---
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

FS_title  = 32; 
FS_labels = 32; 
FS_ticks = 14; 
FS_text = 14; 
FS_legend = 14;

% ========================================================================
% --- FIGURE 1: NONLINEAR RELAXATION (WITH LOG INSET & RATES) ---
% ========================================================================
fig_relax = figure('Name', 'BKW Relaxation', 'Position', [100, 100, 700, 550], 'Color', 'w');
ax_main = axes('Position', [0.12, 0.12, 0.83, 0.8]); % Define main axis bounds
hold on; grid on;

colors = lines(K_max - 1);
leg_h = []; 

% 1. Plot the Main Data
for k = 2:K_max
    idx = k * Basis.N_Q + 1;
    h_num = plot(t_out * mu, c_ode(:, idx), '-', 'Color', [colors(k-1,:), 0.4], 'LineWidth', 6);
    h_ana = plot(t_out * mu, c_analytical(:, idx), 'k:', 'LineWidth', 2);
    if k == 2
        leg_h = [h_num, h_ana];
    end
end

% 2. Format Main Axes
% title('\textbf{Nonlinear BKW Relaxation}', 'FontSize', FS_title);
xlabel('Dimensionless Time $\tau = \mu t$', 'FontSize', FS_labels);
ylabel('Spectral Amplitude $c_k(\tau)$', 'FontSize', FS_labels);
set(ax_main, 'FontSize', FS_ticks, 'LineWidth', 1.2);
xlim([0, 4]); 
ylim([-0.04, 0.002]); % Cap slightly above 0
yticks(-0.04:0.01:0); % Clean integer-like tick steps

% 3. Custom Positioned Legend (Shifted Up)
% leg = legend(leg_h, {'Numerical (RK2)', 'Exact Analytical'}, 'FontSize', FS_legend);
% set(leg, 'Position', [0.55, 0.69, 0.35, 0.08]); % Moved safely away from the inset

% 4. Create the Inset Plot (Enlarged)
% [left, bottom, width, height]
ax_inset = axes('Position', [0.4, 0.17, 0.5, 0.5]); 
hold on; grid on;

for k = 2:K_max
    idx = k * Basis.N_Q + 1;
    % Plot the absolute values on the log scale
    semilogy(t_out * mu, abs(c_ode(:, idx)), '-', 'Color', [colors(k-1,:), 0.4], 'LineWidth', 4);
    semilogy(t_out * mu, abs(c_analytical(:, idx)), 'k:', 'LineWidth', 1.5);
end

% 5. Format the Inset
set(ax_inset, 'YScale', 'log', 'FontSize', FS_text, 'LineWidth', 1.0);
title('\textbf{Exponential Decay} $|c_k|$', 'FontSize', FS_text);
xlim([0, 3]); 
ylim([1e-6, 1e-1]);
yticks(10.^(-6:2:-2)); 
box(ax_inset, 'on');

% 6. Add Dynamic Rates Text Box to the Inset (Moved to Bottom-Left)
rate_str = {'\textbf{Decay Rates} ($\lambda_k/\mu$)', 'Num. \quad Exact'};
for k = 2:K_max
    idx = k * Basis.N_Q + 1;
    num_rate = abs(J_eq(idx, idx)) / mu;
    
    % The TRUE Wang Chang-Uhlenbeck eigenvalue ratio for isotropic Maxwell molecules
    exact_rate = 3 * (k - 1) / (k + 1); 
    
    rate_str{end+1} = sprintf('$k=%d$: %4.3f \\quad %4.3f', k, num_rate, exact_rate);
end

% Place the text box in the bottom-left corner
text(ax_inset, 0.05, 0.05, rate_str, 'Units', 'normalized', ...
    'Interpreter', 'latex', 'FontSize', FS_text, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
    'BackgroundColor', [1 1 1 0.8], 'EdgeColor', 'k', 'Margin', 4);

% Export
if export_to_pdf_figure
    export_fig('fig_bkw_relaxation', '-pdf', '-painters', '-nocrop');
end

% ========================================================================
% --- FIGURE 2: MACHINE-PRECISION CONSERVATION ---
% ========================================================================
fig_cons = figure('Name', 'BKW Conservation', 'Position', [100, 100, 700, 400], 'Color', 'w');
ax_sec = axes('Position', [0.12, 0.12, 0.83, 0.8]); % Define main axis bounds
hold on; grid on;

% Calculate drift relative to initial state, floring at 1e-17 to prevent axis glitches
mass_err = max(abs(c_ode(:, 1) - c_ode(1, 1)), eps());
energy_err = max(abs(c_ode(:, Basis.N_Q + 1) - c_ode(1, Basis.N_Q + 1)), eps());

% Subsample markers so they don't overlap (plot exactly 10 markers across the line)
sub = round(linspace(1, length(t_out), 10));

% Plot with -o and -x markers
semilogy(t_out * mu, mass_err, 'k-o', 'LineWidth', 2, 'MarkerSize', 15, ...
    'MarkerIndices', sub, 'DisplayName', 'Mass Error $|\Delta c_0|$');
semilogy(t_out * mu, energy_err, 'k-x', 'LineWidth', 2, 'MarkerSize', 15, ...
    'MarkerIndices', sub, 'DisplayName', 'Energy Error $|\Delta c_1|$');

% title('\textbf{Numerical Conservation}', 'FontSize', FS_title);
xlabel('Dimensionless Time $\tau = \mu t$', 'FontSize', FS_labels);
ylabel('Absolute Error $|c(t) - c(0)|$', 'FontSize', FS_labels);

% Force strict log scaling and explicit ticks
set(gca, 'YScale', 'log');
ylim([1e-17, 1e-12]); 
yticks(10.^(-17:-12));
legend('Location', 'northeast', 'FontSize', FS_legend);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.2);

if export_to_pdf_figure
    export_fig('fig_bkw_conservation', '-pdf', '-painters', '-nocrop');
end

%% --- HELPER: EXACT CONTINUOUS BKW PROJECTOR ---
function c_state = project_bkw(alpha, K_max, Basis)
    % Integrates the continuous analytical BKW function exactly onto the spectral basis.
    % This uses highly resolved Gauss-Laguerre quadrature to prevent aliasing.
    
    N_rad = 80;
    qr = Gauss.generalized_laguerre(N_rad, 0.5);
    x = qr.x; 
    w = qr.w;
    
    c_state = zeros(Basis.N_terms, 1);
    
    for i = 1:length(x)
        v_sq = x(i);
        
        % F(v) = f_bkw(v) / e^-v^2 
        F_val = pi^(-1.5) * (1-alpha)^(-1.5) * exp(-v_sq * alpha / (1-alpha)) * ...
                (1 + (alpha/(1-alpha)) * (v_sq/(1-alpha) - 1.5));
                
        % Evaluate basis at an arbitrary point on the sphere (x-axis)
        v_vec = [sqrt(v_sq), 0, 0];
        Psi = Basis.evaluate(v_vec); % [1 x N_terms]
        
        % Integrate assuming isotropy: 4*pi*v^2 dv = 2*pi*sqrt(x) dx 
        c_step = (2 * pi * w(i) * F_val) .* Psi';
        
        % CRITICAL FIX: The BKW solution is purely isotropic!
        % Evaluating Psi at a single point is mathematically invalid for L > 0.
        % We strictly retain only the isotropic L = 0 components.
        for k = 0:K_max
            idx = k * Basis.N_Q + 1; % Index of the isotropic mode
            c_state(idx) = c_state(idx) + c_step(idx);
        end
    end
end