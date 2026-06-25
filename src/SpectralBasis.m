classdef SpectralBasis < handle
    % SPECTRALBASIS Exact evaluator for the 4D Polyatomic Spectral Boltzmann Basis
    % Natively vectorized for rapid quadrature point evaluation.
    
    properties
        K_max
        L_max
        I_max     % Maximum internal energy polynomial degree
        nu        % Internal degrees of freedom parameter
        N_Q       % Total angular modes: (L_max + 1)^2
        N_I       % Total internal energy modes: I_max + 1
        N_terms   % Total basis terms: (K_max + 1) * N_I * N_Q
        SH        % Instance of SphericalHarmonics evaluator
    end
    
    methods
        function obj = SpectralBasis(K_max, L_max, I_max, nu)
            if nargin < 3 || isempty(I_max), I_max = 0; end
            if nargin < 4 || isempty(nu), nu = 0; end
            
            obj.K_max = K_max;
            obj.L_max = L_max;
            obj.I_max = I_max;
            obj.nu    = nu;
            
            obj.N_Q = (L_max + 1)^2;
            obj.N_I = I_max + 1;
            obj.N_terms = (K_max + 1) * obj.N_I * obj.N_Q;
            obj.SH = SphericalHarmonics(L_max);
        end
        
        function Psi = evaluate(obj, v_vec, I_vec)
            N_points = size(v_vec, 1);
            if nargin < 3 || isempty(I_vec), I_vec = zeros(N_points, 1); end
            
            Psi = zeros(N_points, obj.N_terms);
            r2 = sum(v_vec.^2, 2);
            r = sqrt(r2) + eps; 
            theta = acos(v_vec(:,3) ./ r);
            phi = atan2(v_vec(:,2), v_vec(:,1));
            
            Y_eval = obj.SH.evaluate(theta, phi); 
            
            H_eval = zeros(N_points, obj.N_I);
            for i = 0:obj.I_max
                L_i = obj.eval_generalized_laguerre(i, obj.nu, I_vec);
                ln_N_ii = gammaln(i + 1) - gammaln(i + obj.nu + 1);
                H_eval(:, i+1) = exp(0.5 * ln_N_ii) .* L_i;
            end
            
            for l = 0:obj.L_max
                m_indices = (l^2 + 1) : ((l+1)^2);
                SolidHarm = (r.^l) .* Y_eval(:, m_indices);
                alpha = l + 0.5;
                
                for k = 0:obj.K_max
                    L_k = obj.eval_generalized_laguerre(k, alpha, r2);
                    ln_M_ii = -log(2) + gammaln(k + alpha + 1) - gammaln(k + 1);
                    radial_part = exp(-0.5 * ln_M_ii) .* L_k;
                    
                    for i = 0:obj.I_max
                        global_indices = (k * obj.N_I + i) * obj.N_Q + m_indices;
                        Psi(:, global_indices) = (radial_part .* H_eval(:, i+1)) .* SolidHarm;
                    end
                end
            end
        end
        
        function Psi_sub = evaluate_sub_basis(obj, v_vec, varargin)
            % BUG FIX: Uses varargin to safely handle legacy 3-argument calls from GeneralCollisionTensor
            if length(varargin) == 1
                q_map = varargin{1};
                I_vec = zeros(size(v_vec, 1), 1);
            else
                I_vec = varargin{1};
                q_map = varargin{2};
            end
            
            N_points = size(v_vec, 1);
            N_sub_q = max(q_map);
            N_sub = (obj.K_max + 1) * obj.N_I * N_sub_q;
            
            Psi_sub = zeros(N_points, N_sub);
            r2 = sum(v_vec.^2, 2);
            r = sqrt(r2) + eps; 
            theta = acos(v_vec(:,3) ./ r);
            phi = atan2(v_vec(:,2), v_vec(:,1));
            
            Y_eval = obj.SH.evaluate(theta, phi);
            
            H_eval = zeros(N_points, obj.N_I);
            for i = 0:obj.I_max
                L_i = obj.eval_generalized_laguerre(i, obj.nu, I_vec);
                ln_N_ii = gammaln(i + 1) - gammaln(i + obj.nu + 1);
                H_eval(:, i+1) = exp(0.5 * ln_N_ii) .* L_i;
            end
            
            for l = 0:obj.L_max
                m_indices = (l^2 + 1) : ((l+1)^2);
                active_mask = q_map(m_indices) > 0;
                if ~any(active_mask), continue; end
                
                active_q = m_indices(active_mask);
                sub_q_indices = q_map(active_q);
                SolidHarm = (r.^l) .* Y_eval(:, active_q);
                alpha = l + 0.5;
                
                for k = 0:obj.K_max
                    L_k = obj.eval_generalized_laguerre(k, alpha, r2);
                    ln_M_ii = -log(2) + gammaln(k + alpha + 1) - gammaln(k + 1);
                    radial_part = exp(-0.5 * ln_M_ii) .* L_k;
                    
                    for i = 0:obj.I_max
                        sub_indices = (k * obj.N_I + i) * N_sub_q + sub_q_indices';
                        Psi_sub(:, sub_indices) = (radial_part .* H_eval(:, i+1)) .* SolidHarm;
                    end
                end
            end
        end
        
        function R_val = evaluate_radial(obj, n_idx, l, v)
            k = n_idx - 1; 
            alpha = l + 0.5;
            L_k = obj.eval_generalized_laguerre(k, alpha, v.^2);
            ln_M_ii = -log(2) + gammaln(k + alpha + 1) - gammaln(k + 1);
            R_val = exp(-0.5 * ln_M_ii) .* L_k .* (v.^l);
        end
        
        function H_val = evaluate_internal(obj, i_idx, I)
            i = i_idx - 1; 
            L_i = obj.eval_generalized_laguerre(i, obj.nu, I);
            ln_N_ii = gammaln(i + 1) - gammaln(i + obj.nu + 1);
            H_val = exp(0.5 * ln_N_ii) .* L_i;
        end
        
        function P_val = evaluate_legendre(obj, l, mu)
            orig_shape = size(mu);
            P_all = legendre(l, mu(:)');
            P_val = reshape(P_all(1, :)', orig_shape);
        end
    end
    
    methods (Static)
        function L_val = eval_generalized_laguerre(k, alpha, x)
            N = length(x);
            if k == 0, L_val = ones(N, 1); return;
            elseif k == 1, L_val = 1 + alpha - x; return; end
            
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