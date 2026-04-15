% TUTORIAL_KERNEL_CONVERGENCE
% Visualizes the Spherical Harmonic Addition Theorem and the spectral 
% convergence of a highly anisotropic Boltzmann collision kernel.
clear; clc; close all;

%% 1. Configuration & Physics Parameters
omega      = 0.5;     % Viscosity index
kappa      = 5.0;     % Anisotropy factor (higher = sharper forward peak)
N_max      = 16;      % Maximum Legendre truncation degree to test

% Solid Harmonic space
K_max      = 4;    
L_max      = 4;
tol        = 0;       % Set to 0 to force it to compute all the way to N_max

fprintf('--- Scattering Kernel Convergence Tutorial ---\n');
fprintf('Simulating Anisotropy (kappa) : %.1f\n', kappa);
fprintf('Evaluating degrees 0 through %d...\n', N_max);

%% 2. Define the Kernel and Initialize
% Initialize the Scattering Kernel (Generates the dense grid up to N_max since tol=0)
alpha = 2.0 * (1.0 - omega);
ScatKernel = ScatteringKernel(K_max, L_max, N_max, alpha, tol);

% FIX 1: Overwrite the internal isotropic kernel with the anisotropic tutorial kernel
ScatKernel.exact_kernel = @(u, cos_chi) (u.^alpha) .* exp(kappa * cos_chi);

% FIX 2: Define the target relative speed (u_val) missing from the original script
u_val = 1.0; 

% Define the incoming relative velocity direction (pointing along X-axis)
u_hat = [1, 0, 0]; 
u_theta = acos(u_hat(3));
u_phi = atan2(u_hat(2), u_hat(1));
u_phi(u_phi < 0) = u_phi(u_phi < 0) + 2*pi;

%% 3. Evaluate the Ground Truth
% Evaluate the exact continuous kernel on every point of the product grid
cos_chi_grid = ScatKernel.Omega * u_hat'; 

% FIX 3: Use the class's built-in evaluate method instead of undefined exact_kernel
exact_B_vals = ScatKernel.evaluate(u_val, cos_chi_grid);

% Compute the exact L2 norm of the kernel for relative error scaling
norm_exact = sqrt(sum(ScatKernel.W_ang .* exact_B_vals.^2));

%% 4. Convergence Loop
% Pre-evaluate the incoming direction Spherical Harmonics at N_max
W_u_full = ScatKernel.get_expansion_weights(u_val);  
SH = SphericalHarmonics(ScatKernel.N_kernel);
Y_u_hat_full = SH.evaluate(u_theta, u_phi);          

rel_L2_errors = zeros(ScatKernel.N_kernel + 1, 1);
degrees = 0:ScatKernel.N_kernel;

for N = 0:ScatKernel.N_kernel
    % Truncate the spectral arrays to degree N
    N_spec_trunc = (N + 1)^2;
    W_u_trunc = W_u_full;
    W_u_trunc(N_spec_trunc + 1 : end) = 0; % Zero out higher degrees
    
    % Reconstruct the kernel using the truncated Addition Theorem
    % B_approx(sigma) = sum_k W_k * Y_k(u_hat) * Y_k(sigma)
    approx_B_vals = ScatKernel.Y_leb * (W_u_trunc .* Y_u_hat_full)'; 
    
    % Compute Relative L2 Error
    error_squared = sum(ScatKernel.W_ang .* (exact_B_vals - approx_B_vals).^2);
    rel_L2_errors(N + 1) = sqrt(error_squared) / norm_exact;
end

% Extract a specific approximation (e.g., N=6) to display the "ringing" artifact
N_display = 6;
W_u_disp = W_u_full; 
W_u_disp((N_display+1)^2 + 1 : end) = 0;
disp_B_vals = ScatKernel.Y_leb * (W_u_disp .* Y_u_hat_full)'; 

%% 5. Visualization
figure('Name', 'Kernel Convergence Tutorial', 'Position', [100, 100, 1400, 450]);
c_limits = [min(disp_B_vals), max(exact_B_vals)];

% Subplot 1: Exact Kernel
subplot(1,3,1);
marker_size = 40 + 60 * (exact_B_vals / max(exact_B_vals));
scatter3(ScatKernel.Omega(:,1), ScatKernel.Omega(:,2), ScatKernel.Omega(:,3), ...
         marker_size, exact_B_vals, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.2);
axis equal; grid on; view(30, 20);
title('Exact Kernel', 'FontSize', 14); colormap(jet); clim(c_limits);

% Subplot 2: Truncated Approximation
subplot(1,3,2);
marker_size = 40 + 60 * (disp_B_vals / max(exact_B_vals));
scatter3(ScatKernel.Omega(:,1), ScatKernel.Omega(:,2), ScatKernel.Omega(:,3), ...
         marker_size, disp_B_vals, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.2);
axis equal; grid on; view(30, 20);
title(sprintf('Approximated (N = %d)', N_display), 'FontSize', 14); colormap(jet); clim(c_limits);

% Subplot 3: Spectral Convergence Curve
subplot(1,3,3);
semilogy(degrees, rel_L2_errors, '-ko', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
grid on;
title('Relative L2 Error Convergence', 'FontSize', 14);
xlabel('Truncation Degree (N)');
ylabel('Relative L2 Error');
xlim([0, N_max]);
ylim([1e-6, 1]);