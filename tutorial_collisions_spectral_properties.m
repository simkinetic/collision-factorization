%% TUTORIAL: Core Physical Properties of the Spectral Boltzmann Operator
% This script validates the three most important physical properties of the 
% fully nonlinear spectral collision tensor:
%   1. Stability & The H-Theorem (Spectrum non-positivity)
%   2. Exact theoretical matching (Wang Chang-Uhlenbeck spectrum)
%   3. Galilean Invariance (Exact conservation despite spectral truncation)
clear; clc; close all;

fprintf('==============================================================\n');
fprintf('  TUTORIAL: Validating the Spectral Collision Tensor\n');
fprintf('==============================================================\n\n');

%% 1. Initialize Phase Space & Generate the Tensor (Done Once)
% We use K_max = 2, L_max = 2 (27 total basis functions).
K_max = 2;  
L_max = 2;  
Basis = SpectralBasis(K_max, L_max);
N_terms = Basis.N_terms;

fprintf('--- PHASE 1: TENSOR GENERATION ---\n');
fprintf('Building the Exact Numerical Collision Tensor (K=%d, L=%d)...\n', K_max, L_max);

% We use Maxwellian molecules (omega = 1.0 -> alpha = 0.0) because they possess 
% an exact analytical spectrum we can compare against.
alpha = 0.0;
N_max = 5;
tol   = 1e-6;

% Initialize using the new interface
Kernel = ScatteringKernel(K_max, L_max, N_max, alpha, tol);

TensorObj = GeneralCollisionTensor(Basis, Kernel);
TensorObj.generate_R_tensor(true);

C_assembled = TensorObj.assemble_full_tensor();
C_flat = reshape(C_assembled, N_terms, N_terms^2);
fprintf('Tensor generation complete.\n\n');

%% 2. Stability Analysis & H-Theorem
fprintf('--- PHASE 2: STABILITY & H-THEOREM ---\n');

% The Jacobian J_ij = dQ_i / dc_j evaluated at equilibrium evaluates to:
J_num = squeeze(C_assembled(:,:,1)) + squeeze(C_assembled(:,1,:));

lambda_num = eig(J_num);
lambda_num = sort(real(lambda_num), 'descend');

num_zeros = sum(abs(lambda_num) < 1e-12);
num_positive = sum(lambda_num > 1e-12);
num_negative = sum(lambda_num < -1e-12);

fprintf('Jacobian Spectrum Summary: %d Zeros, %d Stable (Decaying), %d Unstable (Growing)\n', ...
    num_zeros, num_negative, num_positive);

if num_zeros == 5 && num_positive == 0
    fprintf('SUCCESS: Operator is unconditionally stable and satisfies the H-Theorem!\n\n');
else
    fprintf('WARNING: Operator violates stability requirements.\n\n');
end

% Visualizing the Spectrum (FIGURE 1)
figure('Name', 'Eigenvalue Spectrum', 'Position', [100, 100, 800, 400], 'Color', 'w');
hold on; grid on;

% Plot and explicitly capture handles to ensure the legend formats correctly
h_inv = plot(lambda_num(abs(lambda_num) < 1e-12), 0, 'ko', 'MarkerSize', 10, 'LineWidth', 2);
h_stb = plot(lambda_num(lambda_num < -1e-12), 0, 'b.', 'MarkerSize', 20);
handles = [h_inv(1), h_stb(1)];
labels = {'Invariants (Zero)', 'Relaxation Modes (Stable)'};

if num_positive > 0
    h_ust = plot(lambda_num(lambda_num > 1e-12), 0, 'r*', 'MarkerSize', 10, 'LineWidth', 2);
    handles(end+1) = h_ust(1);
    labels{end+1} = 'Unstable Modes (Growing)';
end

xline(0, 'k-', 'LineWidth', 1.5);
title('Spectrum of the Linearized Collision Operator', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Real Eigenvalue (Growth/Decay Rate)', 'FontSize', 12, 'FontWeight', 'bold');
yticks([]);
legend(handles, labels, 'Location', 'northwest', 'FontSize', 11);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
xlim([min(lambda_num)-0.5, 0.5]);

%% 3. Wang Chang-Uhlenbeck (WCU) Exact Spectrum
fprintf('--- PHASE 3: WANG CHANG-UHLENBECK ANALYTICAL SPECTRUM ---\n');

% Pass the Kernel object instead of a raw function handle
lambda_wcu = compute_wcu_spectrum(K_max, L_max, Kernel);

% Normalize by the first physical relaxation mode (index 6)
norm_num = abs(lambda_num(6));
norm_wcu = abs(lambda_wcu(6));

lambda_num_norm = lambda_num / norm_num;
lambda_wcu_norm = lambda_wcu / norm_wcu;

fprintf('Mode |   Numerical Ratio   |  Analytical Ratio   |  Absolute Diff\n');
fprintf('-----------------------------------------------------------------\n');
for i = 1:N_terms
    diff = abs(lambda_num_norm(i) - lambda_wcu_norm(i));
    lbl = '';
    if abs(lambda_wcu_norm(i)) < 1e-12
        lbl = '(Invariant)';
    end
    fprintf('%4d | %19.12f | %19.12f | %12.2e %s\n', ...
        i, lambda_num_norm(i), lambda_wcu_norm(i), diff, lbl);
end
fprintf('-----------------------------------------------------------------\n');

max_spec_diff = max(abs(lambda_num_norm - lambda_wcu_norm));
if max_spec_diff < 1e-10
    fprintf('SUCCESS: The numerical operator perfectly matches WCU theory!\n\n');
else
    fprintf('WARNING: Discrepancy found in the spectrum.\n\n');
end

% Visualizing the WCU Match (FIGURE 2)
figure('Name', 'WCU Spectrum Match', 'Position', [150, 150, 800, 500], 'Color', 'w');
hold on; grid on;

% 1. Exact Analytical Theory (Thick, semi-transparent grey line)
plot(1:N_terms, lambda_wcu_norm, '-', 'Color', [0.6, 0.6, 0.6, 0.6], 'LineWidth', 10, 'DisplayName', 'Analytical WCU Theory');

% 2. Numerical Tensor Eigenvalues (Crisp black dots)
plot(1:N_terms, lambda_num_norm, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k', 'DisplayName', 'Numerical Tensor');

title('Wang Chang-Uhlenbeck Eigenvalue Spectrum Match', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Sorted Mode Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Eigenvalue \lambda_k / |\lambda_6|', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);

%% 4. Galilean Invariance & Truncation Independence
fprintf('--- PHASE 4: GALILEAN INVARIANCE ---\n');
fprintf('Generating 3D Quadrature Grid for Projection...\n');

N_rad = 60; 
qr = Gauss.generalized_laguerre(N_rad, 0.5);
r_nodes = sqrt(qr.x);
w_rad = qr.w / 2;

N_leb = Kernel.N_leb;
Omega = Kernel.Omega;
W_ang = Kernel.W_ang;

N_points = N_rad * N_leb;
v_vec = zeros(N_points, 3);
W_tot = zeros(N_points, 1);

idx = 1;
for a = 1:N_rad
    for p = 1:N_leb
        v_vec(idx, :) = r_nodes(a) * Omega(p, :);
        W_tot(idx) = w_rad(a) * W_ang(p); 
        idx = idx + 1;
    end
end

Psi_eval = Basis.evaluate(v_vec);

% Identify the 5 invariant indices in our global basis
idx_mass = 1;
idx_momY = 2;
idx_momZ = 3;
idx_momX = 4;
idx_energy = 1 * Basis.N_Q + 1; % K=1, L=0, m=0
inv_indices = [idx_mass, idx_momY, idx_momZ, idx_momX, idx_energy];

fprintf(' Bulk Velocity U  |  Truncation Error Norm (L2)  |  Invariant Conservation Error \n');
fprintf('---------------------------------------------------------------------------------\n');

% Test a stationary Maxwellian, and three increasingly fast Shifted Maxwellians
U_tests = [0.0, 0.1, 0.3, 0.6];
max_inv_error = 0;

for U_mag = U_tests
    U_vec = [U_mag, 0, 0];
    
    % F_v = f(v) / e^-v^2 = pi^(-3/2) * exp(2*v.U - U^2)
    F_v = pi^(-1.5) * exp(2 * (v_vec * U_vec') - U_mag^2);
    c_shift = Psi_eval' * (W_tot .* F_v);
    
    % Evaluate Q = C * (c x c)
    c_outer = c_shift * c_shift';
    Q = C_flat * c_outer(:);
    
    Q_invariants = Q(inv_indices);
    current_inv_error = max(abs(Q_invariants));
    max_inv_error = max(max_inv_error, current_inv_error);
    total_error_norm = norm(Q);
    
    lbl = '(Shifted - Truncated)';
    if U_mag == 0
        lbl = '(Stationary - Exact)';
    end
    
    fprintf(' U = [%.1f, 0, 0]  |          %8.2e          |           %8.2e %s\n', ...
        U_mag, total_error_norm, current_inv_error, lbl);
end
fprintf('---------------------------------------------------------------------------------\n');

if max_inv_error < 1e-12
    fprintf('SUCCESS: Invariants are perfectly conserved despite massive spectral truncation!\n\n');
else
    fprintf('WARNING: Invariants were broken by the coordinate shift.\n\n');
end

%% --- HELPER: WCU ANALYTICAL EIGENVALUE GENERATOR ---
function lambda_exact = compute_wcu_spectrum(K_max, L_max, kernel_obj)
    qr = Gauss.legendre(200, -1, 1);
    mu = qr.x;
    w = qr.w;
    
    % Evaluate using the Kernel object's internal method instead of a raw handle
    B_vals = kernel_obj.evaluate(1.0, mu); 
    
    c = sqrt((1 + mu) / 2);
    s = sqrt((1 - mu) / 2);
    lambda_list = [];
    
    for l = 0:L_max
        P_all_c = legendre(l, c'); P_l_c = P_all_c(1, :)';
        P_all_s = legendre(l, s'); P_l_s = P_all_s(1, :)';
        
        for k = 0:K_max
            term1 = (c.^(2*k + l)) .* P_l_c;
            term2 = (s.^(2*k + l)) .* P_l_s;
            integrand = B_vals .* (term1 + term2 - 1);
            val = 2 * pi * sum(w .* integrand);
            
            if (k == 0 && l == 0) || (k == 0 && l == 1) || (k == 1 && l == 0)
                val = 0;
            end
            
            for m = 1:(2*l + 1)
                lambda_list(end+1, 1) = val;
            end
        end
    end
    lambda_exact = sort(lambda_list, 'descend');
end