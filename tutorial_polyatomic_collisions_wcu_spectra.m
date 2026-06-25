%% TUTORIAL: Exact Analytical Ratios for Linear and Non-Linear Polyatomic Gases
% This script validates the mathematical properties of the fully nonlinear 
% spectral collision tensor for both linear (nu=0.0) and non-linear (nu=0.5) 
% polyatomic Maxwell gases.

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL');

fprintf('==============================================================\n');
fprintf('  TUTORIAL: Polyatomic Collision Tensor Validation\n');
fprintf('==============================================================\n\n');

% Set global spectral resolution
K_max = 2;
L_max = 2;  
I_max = 2;

% Define the two test cases
nu_cases = [0.0, 0.5];
molecule_types = {'Linear Polyatomic (D=5)', 'Non-Linear Polyatomic (D=6)'};

% Storage for final summary comparison
results_shear = zeros(2, 1);
results_bulk  = zeros(2, 1);
results_ratio = zeros(2, 1);
exact_ratios  = zeros(2, 1);
error_ratios  = zeros(2, 1);

for case_idx = 1:length(nu_cases)
    nu_val = nu_cases(case_idx);
    mol_type = molecule_types{case_idx};
    
    fprintf('==============================================================\n');
    fprintf('  CASE %d: %s | nu = %.1f\n', case_idx, mol_type, nu_val);
    fprintf('==============================================================\n\n');
    
    %% 1. Initialize Phase Space & Generate the Tensor
    Basis = SpectralBasis(K_max, L_max, I_max, nu_val);
    N_I = I_max + 1;
    N_Q = (L_max + 1)^2;
    
    fprintf('--- PHASE 1: TENSOR GENERATION ---\n');
    % Maxwellian molecules (gamma = 0.0)
    gamma = 0.0;
    Kernel = ScatteringKernel('Polyatomic', gamma);
    TensorObj = GeneralCollisionTensor(Basis, Kernel);
    
    % PADDING: (Spatial Radial, Spatial Angular, Internal)
    % Internal pad = 0 is sufficient for exact integration of polynomials
    TensorObj.generate_R_tensor_sumfac(16, 16, 0); 
    C_assembled = TensorObj.assemble_full_tensor();
    fprintf('Tensor generation complete.\n\n');
    
    %% 2. Stability Analysis & Jacobian Construction
    fprintf('--- PHASE 2: JACOBIAN & NORMALIZATION ---\n');
    J_num = squeeze(C_assembled(:,:,1)) + squeeze(C_assembled(:,1,:));
    lambda_num = eig(J_num);
    lambda_num = sort(real(lambda_num), 'descend');
    
    % Normalize globally against the 6th mode (first physical relaxation mode)
    norm_factor = abs(lambda_num(6));
    J_num = J_num / norm_factor;
    
    num_zeros = sum(abs(lambda_num) < 1e-10);
    if num_zeros == 5
        fprintf('SUCCESS: Operator is stable and possesses exactly 5 null-space invariants.\n\n');
    else
        fprintf('WARNING: Operator violates conservation requirements (%d zeros found).\n\n', num_zeros);
    end
    
    %% 3. Exact WCU Eigenvalue Spectrum Extraction
    fprintf('--- PHASE 3: NAVIER-STOKES EIGENVALUE EXTRACTION ---\n');
    
    lambda_shear = 0;
    lambda_bulk = 0;
    
    for l = [0, 2] % We only need L=0 (Bulk) and L=2 (Shear) for this test
        m = 0; 
        for n = 0:(K_max + I_max)
            idx_list = [];
            
            for k = 0:K_max
                for i = 0:I_max
                    if (k + i) == n
                        idx = (k * N_I + i) * N_Q + (l^2 + l + m) + 1;
                        idx_list = [idx_list, idx];
                    end
                end
            end
            
            if isempty(idx_list), continue; end
            
            J_block = J_num(idx_list, idx_list);
            evals = sort(real(eig(J_block)), 'descend');
            
            % Capture the physical Navier-Stokes modes
            if l == 2 && n == 0
                lambda_shear = evals(1);
            elseif l == 0 && n == 1
                lambda_bulk = evals(2); % 1st is 0 (Energy invariant), 2nd is Bulk
            end
        end
    end
    
    %% 4. Analytical Comparison
    % Compute the theoretical degrees of freedom
    d_t = 3; 
    d_i = 2 * nu_val + 2; 
    
    % Exact Theoretical Ratio = (2 * (dt + di)) / (dt + 2*di)
    exact_ratio = (2 * (d_t + d_i)) / (d_t + 2 * d_i);
    numerical_ratio = lambda_bulk / lambda_shear;
    
    % Compute Absolute Error
    error_ratio = abs(numerical_ratio - exact_ratio);
    
    % Store for summary
    results_shear(case_idx) = lambda_shear;
    results_bulk(case_idx)  = lambda_bulk;
    results_ratio(case_idx) = numerical_ratio;
    exact_ratios(case_idx)  = exact_ratio;
    error_ratios(case_idx)  = error_ratio;
    
    fprintf('  Extracted Shear Viscosity Eigenvalue (L=2, n=0) : %16.8f\n', lambda_shear);
    fprintf('  Extracted Bulk Viscosity Eigenvalue  (L=0, n=1) : %16.8f\n', lambda_bulk);
    fprintf('  ------------------------------------------------------------------\n');
    fprintf('  Numerical Ratio (Bulk / Shear)                  : %16.12f\n', numerical_ratio);
    fprintf('  Exact Analytical Theoretical Ratio              : %16.12f\n', exact_ratio);
    fprintf('  Absolute Ratio Error                            : %16.4e\n\n', error_ratio);
    
end

%% 5. FINAL SUMMARY TABLE
fprintf('=================================================================================\n');
fprintf('  FINAL SUMMARY: POLYATOMIC CONTINUUM VALIDATION\n');
fprintf('=================================================================================\n');
fprintf('  %-25s | %-20s | %-20s\n', 'Metric', 'nu = 0.0', 'nu = 0.5');
fprintf('---------------------------------------------------------------------------------\n');
fprintf('  %-25s | %-20s | %-20s\n', 'Molecule Geometry', 'Linear', 'Non-Linear');
fprintf('  %-25s | %-20d | %-20d\n', 'Total D.O.F (D)', 5, 6);
fprintf('  %-25s | %-20.8f | %-20.8f\n', 'Numerical Shear Mode', results_shear(1), results_shear(2));
fprintf('  %-25s | %-20.8f | %-20.8f\n', 'Numerical Bulk Mode', results_bulk(1), results_bulk(2));
fprintf('---------------------------------------------------------------------------------\n');
fprintf('  %-25s | %-20.12f | %-20.12f\n', 'Numerical Ratio', results_ratio(1), results_ratio(2));
fprintf('  %-25s | %-20.12f | %-20.12f\n', 'Theoretical Ratio', exact_ratios(1), exact_ratios(2));
fprintf('  %-25s | %-20.4e | %-20.4e\n', 'Absolute Ratio Error', error_ratios(1), error_ratios(2));
fprintf('=================================================================================\n');

if max(error_ratios) < 1e-12
    fprintf('  >>> SUCCESS: Tensor flawlessly captures continuum limits at machine precision! <<<\n');
else
    fprintf('  >>> WARNING: Error exceeds standard machine precision limits. <<<\n');
end
fprintf('=================================================================================\n');