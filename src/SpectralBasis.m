classdef SpectralBasis < handle
    % SPECTRALBASIS Exact evaluator for the 3D Spectral Boltzmann Basis
    % Evaluates Orthonormal phi_{k,l,m}(v) = N_{kl} * L_k^{(l+1/2)}(|v|^2) * |v|^l * Y_{l,m}(v_hat)
    % Natively vectorized for rapid quadrature point evaluation.
    
    properties
        K_max
        L_max
        N_Q       % Total angular modes: (L_max + 1)^2
        N_terms   % Total basis terms: (K_max + 1) * N_Q
        SH        % Instance of SphericalHarmonics evaluator
    end
    
    methods
        function obj = SpectralBasis(K_max, L_max)
            obj.K_max = K_max;
            obj.L_max = L_max;
            obj.N_Q = (L_max + 1)^2;
            obj.N_terms = (K_max + 1) * obj.N_Q;
            
            % Initialize the Spherical Harmonic evaluator
            obj.SH = SphericalHarmonics(L_max);
        end
        
        function Psi = evaluate(obj, v_vec)
            % EVALUATE Computes the full spectral basis at given velocities
            N_points = size(v_vec, 1);
            Psi = zeros(N_points, obj.N_terms);
            
            % 1. Convert Cartesian to Spherical (r, theta, phi)
            r2 = sum(v_vec.^2, 2);
            r = sqrt(r2) + eps; 
            theta = acos(v_vec(:,3) ./ r);
            phi = atan2(v_vec(:,2), v_vec(:,1));
            
            % 2. Evaluate all angular components Y_{l,m}
            Y_eval = obj.SH.evaluate(theta, phi); 
            
            % 3. Assemble the full basis
            for l = 0:obj.L_max
                m_indices = (l^2 + 1) : ((l+1)^2);
                SolidHarm = (r.^l) .* Y_eval(:, m_indices);
                alpha = l + 0.5;
                
                for k = 0:obj.K_max
                    L_k = obj.eval_generalized_laguerre(k, alpha, r2);
                    
                    % --- ORTHONORMALIZATION ---
                    % M_ii = 0.5 * Gamma(k + alpha + 1) / k!
                    % Using log-gamma to prevent factorial overflow at high degrees
                    ln_M_ii = -log(2) + gammaln(k + alpha + 1) - gammaln(k + 1);
                    norm_factor = exp(-0.5 * ln_M_ii);
                    
                    global_indices = k * obj.N_Q + m_indices;
                    Psi(:, global_indices) = norm_factor .* L_k .* SolidHarm;
                end
            end
        end
        
        function Psi_sub = evaluate_sub_basis(obj, v_vec, q_map)
            % EVALUATE_SUB_BASIS (Optimized for Pivot Extraction)
            N_points = size(v_vec, 1);
            N_sub_q = max(q_map);
            N_sub = (obj.K_max + 1) * N_sub_q;
            
            Psi_sub = zeros(N_points, N_sub);
            
            r2 = sum(v_vec.^2, 2);
            r = sqrt(r2) + eps; 
            theta = acos(v_vec(:,3) ./ r);
            phi = atan2(v_vec(:,2), v_vec(:,1));
            
            Y_eval = obj.SH.evaluate(theta, phi);
            
            for l = 0:obj.L_max
                m_indices = (l^2 + 1) : ((l+1)^2);
                
                active_mask = q_map(m_indices) > 0;
                if ~any(active_mask)
                    continue; 
                end
                
                active_q = m_indices(active_mask);
                sub_q_indices = q_map(active_q);
                
                SolidHarm = (r.^l) .* Y_eval(:, active_q);
                alpha = l + 0.5;
                
                for k = 0:obj.K_max
                    L_k = obj.eval_generalized_laguerre(k, alpha, r2);
                    
                    % --- ORTHONORMALIZATION ---
                    ln_M_ii = -log(2) + gammaln(k + alpha + 1) - gammaln(k + 1);
                    norm_factor = exp(-0.5 * ln_M_ii);
                    
                    sub_indices = k * N_sub_q + sub_q_indices';
                    Psi_sub(:, sub_indices) = norm_factor .* L_k .* SolidHarm;
                end
            end
        end

        function R_val = evaluate_radial(obj, n_idx, l, v)
            % EVALUATE_RADIAL Computes the purely radial part of the basis:
            % R_{kl}(v) = N_{kl} * L_k^{(l+1/2)}(v^2) * v^l
            %
            % Note: n_idx is 1-based from the loop (1 to K_max + 1), 
            % so we shift it to get the 0-based Laguerre degree 'k'.
            
            k = n_idx - 1; 
            alpha = l + 0.5;
            
            % 1. Evaluate generalized Laguerre polynomial
            v2 = v.^2;
            L_k = obj.eval_generalized_laguerre(k, alpha, v2);
            
            % 2. Compute orthonormalization constant matching the 3D evaluate()
            ln_M_ii = -log(2) + gammaln(k + alpha + 1) - gammaln(k + 1);
            norm_factor = exp(-0.5 * ln_M_ii);
            
            % 3. Assemble the full radial component
            R_val = norm_factor .* L_k .* (v.^l);
        end
        
        function P_val = evaluate_legendre(obj, l, mu)
            % EVALUATE_LEGENDRE Computes the standard Legendre polynomial P_l(mu)
            % Natively handles N-dimensional arrays (like our 3D tensor grid)
            
            orig_shape = size(mu);
            mu_flat = mu(:)'; % Flatten to a row vector for MATLAB's legendre()
            
            % MATLAB's legendre(l, x) returns an (l+1) x N matrix. 
            % The 1st row corresponds to m=0, which is the standard P_l(x).
            P_all = legendre(l, mu_flat);
            P_val_flat = P_all(1, :)'; 
            
            % Reshape seamlessly back into the original ND array shape
            P_val = reshape(P_val_flat, orig_shape);
        end
        
    end
    
    methods (Static)
        function L_val = eval_generalized_laguerre(k, alpha, x)
            N = length(x);
            if k == 0
                L_val = ones(N, 1);
                return;
            elseif k == 1
                L_val = 1 + alpha - x;
                return;
            end
            
            L_prev2 = ones(N, 1);
            L_prev1 = 1 + alpha - x;
            L_val = zeros(N, 1);
            
            for i = 1:(k-1)
                L_val = ((2*i + 1 + alpha - x) .* L_prev1 - (i + alpha) .* L_prev2) / (i + 1);
                L_prev2 = L_prev1;
                L_prev1 = L_val;
            end
        end
    end
end