classdef TestWCUEigenvalues < matlab.unittest.TestCase
    % TESTWCUEIGENVALUES - Unit test validating the spectral collision operator
    % against the exact analytical Wang Chang-Uhlenbeck (WCU) eigenvalues.
    
    properties
        K_max = 2;  
        L_max = 2;  
        tol = 1e-10; % Strict tolerance for mathematical proof
    end
    
    methods (Test)
        
        function testSpectrumMatchesWCU(testCase)
            fprintf('\n==============================================================\n');
            fprintf('  PROOF: Wang Chang-Uhlenbeck Analytical Eigenvalues\n');
            fprintf('==============================================================\n');
            
            %% 1. Initialize Phase Space & Physics
            Basis = SpectralBasis(testCase.K_max, testCase.L_max);
            N_terms = Basis.N_terms;
            
            % Maxwellian molecules (gamma = 0.0) guarantee exact theoretical matching
            gamma = 0.0;
            
            fprintf('1. Building the Exact Numerical Collision Tensor...\n');
            Kernel = ScatteringKernel(gamma);
            TensorObj = GeneralCollisionTensor(Basis, Kernel);
            
            % Generate using the sum-factorized routine with high precision padding
            TensorObj.generate_R_tensor_sumfac(16, 16);
            C_assembled = TensorObj.assemble_full_tensor();
            
            %% 2. Extract Numerical Spectrum
            fprintf('2. Computing Numerical Jacobian...\n');
            % Linearize around the base equilibrium mode (c_1 = 1.0)
            % J_ij = dQ_i / dc_j = C_ij1 + C_i1j
            J_num = squeeze(C_assembled(:, :, 1)) + squeeze(C_assembled(:, 1, :));
            
            lambda_num = eig(J_num);
            lambda_num = sort(real(lambda_num), 'descend');
            
            %% 3. Generate Analytical WCU Spectrum
            fprintf('3. Computing Analytical WCU Spectrum...\n');
            lambda_wcu = testCase.compute_wcu_spectrum(testCase.K_max, testCase.L_max);
            
            %% 4. Normalize
            % The first 5 modes are invariants (0). The 6th mode is the first 
            % true decaying relaxation mode (usually L=2 Stress or K=1 Heat Flux).
            % We divide by its absolute magnitude so the first physical mode is -1.0.
            norm_num = abs(lambda_num(6));
            norm_wcu = abs(lambda_wcu(6));
            
            lambda_num_norm = lambda_num / norm_num;
            lambda_wcu_norm = lambda_wcu / norm_wcu;
            
            %% 5. Console Output (Optional, but good for diagnostics)
            fprintf('\n--- SPECTRUM COMPARISON (Normalized by 1st Relaxation Mode) ---\n');
            fprintf('Mode |   Numerical Ratio   |  Analytical Ratio   |  Absolute Diff\n');
            fprintf('-----------------------------------------------------------------\n');
            
            for i = 1:N_terms
                diff = abs(lambda_num_norm(i) - lambda_wcu_norm(i));
                
                if abs(lambda_wcu_norm(i)) < 1e-12
                    lbl = '(Invariant)';
                else
                    lbl = '';
                end
                
                fprintf('%4d | %19.12f | %19.12f | %12.2e %s\n', ...
                    i, lambda_num_norm(i), lambda_wcu_norm(i), diff, lbl);
            end
            fprintf('-----------------------------------------------------------------\n');
            
            %% 6. Mathematical Assertion
            % Verify invariants are natively 0
            testCase.verifyLessThan(abs(lambda_num_norm(1:5)), testCase.tol, ...
                'The first 5 invariant modes are not exactly zero.');
            
            % Verify the entire spectrum matches the analytical WCU theory
            testCase.verifyEqual(lambda_num_norm, lambda_wcu_norm, 'AbsTol', testCase.tol, ...
                'The numerical operator does not perfectly match WCU theory!');
        end
        
    end
    
    methods (Access = private)
        
        function lambda_exact = compute_wcu_spectrum(~, K_max, L_max)
            % High-order quadrature for exact theoretical integration
            qr = Gauss.legendre(200, -1, 1);
            mu = qr.x;
            w = qr.w;
            
            % For VHS (Variable Hard Sphere) models like Hard Spheres and Maxwell 
            % Molecules, the angular scattering is purely isotropic. Therefore, 
            % the angular dependence B(cos X) is simply a constant.
            B_vals = ones(size(mu)); 
            
            % Half-angle trigonometric identities
            c = sqrt((1 + mu) / 2);
            s = sqrt((1 - mu) / 2);
            
            lambda_list = [];
            
            for l = 0:L_max
                % Legendre polynomials P_l
                P_all_c = legendre(l, c'); P_l_c = P_all_c(1, :)';
                P_all_s = legendre(l, s'); P_l_s = P_all_s(1, :)';
                
                for k = 0:K_max
                    % WCU Integrand: B * (c^(2k+l)*P_l(c) + s^(2k+l)*P_l(s) - 1)
                    term1 = (c.^(2*k + l)) .* P_l_c;
                    term2 = (s.^(2*k + l)) .* P_l_s;
                    
                    integrand = B_vals .* (term1 + term2 - 1);
                    val = 2 * pi * sum(w .* integrand);
                    
                    % The 5 invariants are exactly 0 natively, but we clamp them 
                    % to remove floating point integration noise.
                    if (k == 0 && l == 0) || (k == 0 && l == 1) || (k == 1 && l == 0)
                        val = 0;
                    end
                    
                    % Each eigenvalue has a geometric degeneracy of (2l + 1)
                    for m = 1:(2*l + 1)
                        lambda_list(end+1, 1) = val;
                    end
                end
            end
            
            % Sort descending (Invariants first, then the fastest decaying modes)
            lambda_exact = sort(lambda_list, 'descend');
        end
        
    end
end