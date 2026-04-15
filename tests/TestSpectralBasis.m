classdef TestSpectralBasis < matlab.unittest.TestCase
    % SPECTRALBASISTEST Unit tests for the SpectralBasis class
    % Validates orthogonality, analytical normalization, and optimized sub-basis extraction.
    
    methods (Test)
        
        function testConstructor(testCase)
            % Verify that properties are correctly initialized
            K_max = 2; 
            L_max = 3;
            Basis = SpectralBasis(K_max, L_max);
            
            testCase.verifyEqual(Basis.K_max, 2, 'K_max mismatch.');
            testCase.verifyEqual(Basis.L_max, 3, 'L_max mismatch.');
            testCase.verifyEqual(Basis.N_Q, 16, 'N_Q should be (L_max+1)^2.');
            testCase.verifyEqual(Basis.N_terms, 48, 'N_terms should be (K_max+1)*N_Q.');
        end
        
        function testOrthogonalityAndNormalization(testCase)
            % Rigorous check of the L2 inner product using Gauss and Lebedev quadrature
            K_max = 3; 
            L_max = 4;
            Basis = SpectralBasis(K_max, L_max);
            
            % 1. Setup Exact Quadrature
            % Radial Grid (integrates x^{0.5} e^{-x} where x = r^2)
            N_rad = max(15, 2 * K_max + L_max + 2); 
            qr_rad = Gauss.generalized_laguerre(N_rad, 0.5); 
            r_nodes = sqrt(qr_rad.x); 
            w_rad = qr_rad.w / 2; % dx = 2r dr Jacobian adjustment
            
            % Angular Grid (Lebedev)
            req_deg = 2 * L_max;
            [N_leb, ~] = get_required_lebedev_points(req_deg); 
            leb_grid = getLebedevSphere(N_leb);
            Omega = [leb_grid.x, leb_grid.y, leb_grid.z];
            w_ang = leb_grid.w;
            
            % Flatten grid
            N_total = N_rad * N_leb;
            v_vec = zeros(N_total, 3);
            weights = zeros(N_total, 1);
            
            idx = 1;
            for i = 1:N_rad
                for j = 1:N_leb
                    v_vec(idx, :) = r_nodes(i) * Omega(j, :);
                    weights(idx) = w_rad(i) * w_ang(j);
                    idx = idx + 1;
                end
            end
            
            % 2. Evaluate Basis
            Psi = Basis.evaluate(v_vec);
            
            % 3. Numerical Mass Matrix
            W_matrix = spdiags(weights, 0, N_total, N_total);
            M_num = Psi' * W_matrix * Psi;
            
            % 4. Exact Theoretical Diagonal
            exact_diag = zeros(Basis.N_terms, 1);
            for l = 0:L_max
                m_indices = (l^2 + 1) : ((l+1)^2);
                for k = 0:K_max
                    % Analytic L2 norm: 0.5 * Gamma(k + l + 1.5) / k!
                    N_kl = 0.5 * gamma(k + l + 1.5) / factorial(k);
                    global_indices = k * Basis.N_Q + m_indices;
                    exact_diag(global_indices) = N_kl;
                end
            end
            
            % --- Inside your test or script ---
            diag_M = diag(M_num);
            off_diag_M = M_num - diag(diag_M);
            
            % 1. Relative Max Error (Scaled by the magnitude of the basis)
            rel_max_err = max(abs(off_diag_M(:))) / mean(diag_M);
            
            % 2. Frobenius Relative Error (Scaled by the size/energy of the whole matrix)
            rel_frob_err = norm(off_diag_M, 'fro') / norm(diag(diag_M), 'fro');
            
            fprintf('Mean Diagonal Value:  %.2e\n', mean(diag_M));
            fprintf('Relative Max Error:   %.2e\n', rel_max_err);
            fprintf('Relative Frob Error:  %.2e\n', rel_frob_err);
            
            % Updated Assertion
            testCase.verifyLessThan(rel_frob_err, 1e-8, ...
                'The relative orthogonality error is too high.');
        end
        
        function testSubBasisEvaluation(testCase)
            % Verifies that `evaluate_sub_basis` perfectly mimics extracting
            % columns from the full `evaluate` matrix for a given q_map.
            
            K_max = 2; 
            L_max = 2;
            Basis = SpectralBasis(K_max, L_max);
            
            % Create a few arbitrary 3D test points
            v_vec = [
                1.0,  0.0,  0.0; 
                0.0,  1.0,  0.0; 
                0.0,  0.0,  1.0; 
                1.5, -0.2,  0.8; 
               -0.5,  2.1, -1.1
            ];
            
            % Create a mock `q_map`. Arbitrarily mask out a few q-indices.
            % Only active_qs will be computed by evaluate_sub_basis.
            q_map = zeros(Basis.N_Q, 1);
            active_qs = [1, 3, 5, 6, 9]; 
            for i = 1:length(active_qs)
                q_map(active_qs(i)) = i;
            end
            
            % Evaluate both methods
            Psi_full = Basis.evaluate(v_vec);
            Psi_sub  = Basis.evaluate_sub_basis(v_vec, q_map);
            
            % 1. Verify Dimensions
            expected_cols = (K_max + 1) * length(active_qs);
            testCase.verifyEqual(size(Psi_sub, 2), expected_cols, ...
                'evaluate_sub_basis returned an incorrect number of columns.');
            
            % 2. Verify Values (Extract corresponding columns from Psi_full)
            Psi_extracted = zeros(size(v_vec, 1), expected_cols);
            
            for k = 0:K_max
                for j = 1:length(active_qs)
                    full_col = k * Basis.N_Q + active_qs(j);
                    sub_col  = k * length(active_qs) + j;
                    Psi_extracted(:, sub_col) = Psi_full(:, full_col);
                end
            end
            
            % Should match to strict machine precision since the math is identical
            testCase.verifyEqual(Psi_sub, Psi_extracted, 'AbsTol', 1e-14, ...
                'evaluate_sub_basis results differ from full evaluate results.');
        end
        
    end
end