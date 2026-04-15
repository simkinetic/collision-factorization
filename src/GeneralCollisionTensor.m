classdef GeneralCollisionTensor < handle
% GENERALCOLLISIONTENSOR - Exact Spectral Boltzmann Collision Operator
    
    properties
        K_max
        L_max
        D_max
        
        R_tensor       % [K x K x K x N_L] The Reduced Physical Tensor
        gaunt_labels   % [N_G x 3] Non-zero Real Gaunt transitions
        gaunt_vals     % [N_G x 1] Real Gaunt values
        ic_map         % [N_G x 1] Mapping from transitions to L-triplets
        
        Basis          % SpectralBasis object
        Kernel         % ScatteringKernel object
        
        use_mex        % Boolean flag to toggle C++ MEX execution
    end
    
    methods

        function obj = GeneralCollisionTensor(Basis, Kernel)
            obj.Basis = Basis;
            obj.Kernel = Kernel;
            
            obj.K_max = Basis.K_max;
            obj.L_max = Basis.L_max;
            obj.D_max = 2 * obj.K_max + obj.L_max;
            
            % Generate geometry maps (assuming this method exists in your class)
            obj.generate_geometry(); 
        end
       
        function generate_R_tensor(obj, use_mex)
            if nargin < 2
                obj.use_mex = false;
            else
                obj.use_mex = use_mex;
            end
            obj.generate_R_tensor_imp();
        end

        function generate_R_tensor_imp(obj)
            % GENERATE_R_TENSOR
            % Computes the fully reduced physical collision tensor R.
            % Branches between highly optimized native MATLAB and C++ MEX.
            
            N_K = obj.K_max + 1; 
            N_Q = obj.Basis.N_Q;
            N_L = max(obj.ic_map);
            
            % =============================================================
            % PHASE 1: Establish Dynamic Quadrature Grids
            % =============================================================
            N_rad = max(8, ceil((3 * obj.K_max + 2 * obj.L_max + 1) / 2) + 3);
            qr = Gauss.generalized_laguerre(N_rad, 0.5);
            v_nodes = sqrt(qr.x);
            w_nodes = v_nodes;
            W_rad   = 0.5 * qr.w; 
            
            N_theta = max(6, ceil((3 * obj.L_max + 2 * obj.K_max + 1) / 2) + 2);
            qr_theta = Gauss.legendre(N_theta, -1, 1);
            nodes_theta_1D = qr_theta.x;
            W_theta_1D = qr_theta.w;
            
            N_chi = max(6, ceil((3 * obj.L_max + 2 * obj.K_max + 1) / 2) + 2);
            qr_chi = Gauss.legendre(N_chi, -1, 1);
            nodes_chi = qr_chi.x;
            W_chi = qr_chi.w;
            
            N_eps = max(6, 2 * obj.K_max + obj.L_max + 2);
            eps_nodes = linspace(0, 2*pi, N_eps + 1);
            eps_nodes(end) = []; 
            W_eps_1D = (2*pi) / N_eps * ones(N_eps, 1); 
            
            % Grids & Weights
            mu_theta = reshape(nodes_theta_1D, [], 1, 1); 
            mu_chi   = reshape(nodes_chi,   1, [], 1); 
            eps_vec  = reshape(eps_nodes,   1, 1, []); 
            
            W_sphere = reshape(W_chi, 1, [], 1) .* reshape(W_eps_1D, 1, 1, []);
            W_theta_3D  = reshape(W_theta_1D, [], 1);
            W_Total = W_sphere .* W_theta_3D; 
            
            % 2D meshgrid of radial weights for easy flattening
            [W_v_grid, W_w_grid] = meshgrid(W_rad, W_rad);
            W_vw = W_v_grid .* W_w_grid;
            
            % Pre-evaluate Radial Basis
            R_table = zeros(N_K, obj.L_max + 1, N_rad);
            for a = 1:N_rad
                for l = 0:obj.L_max
                    for n_idx = 1:N_K
                        R_table(n_idx, l+1, a) = obj.Basis.evaluate_radial(n_idx, l, v_nodes(a));
                    end
                end
            end
            
            theta_w_flat = acos(nodes_theta_1D);
            phi_w_flat = zeros(N_theta, 1);
            Y_all_w = obj.Basis.SH.evaluate(theta_w_flat, phi_w_flat);
            
            % -------------------------------------------------------------
            % SETUP: Pre-Collapse Geometry
            % -------------------------------------------------------------
            L_triplets = zeros(N_L, 3);
            P_loss_pre = zeros(N_theta, N_L);
            P_gain_weights = zeros(N_theta, N_Q, N_L);
            
            % Padded qi_valid matrix for C++ compatibility (padded with -1)
            % Size [N_Q x N_L] so memory is contiguous down the columns for each 't'
            qi_valid_matrix = -1 * ones(N_Q, N_L);
            
            for t = 1:N_L
                g_idx_list = find(obj.ic_map == t);
                q_triplet = obj.gaunt_labels(g_idx_list(1), :);
                L_triplets(t, :) = floor(sqrt(q_triplet - 1));
                
                l_i = L_triplets(t, 1);
                l_j = L_triplets(t, 2);
                q_i0 = l_i^2 + l_i + 1;
                q_j0 = l_j^2 + l_j + 1;
                
                Y_i0_val = sqrt((2*l_i + 1) / (4*pi));
                Y_j0_val = sqrt((2*l_j + 1) / (4*pi));
                
                D_full = sum(obj.gaunt_vals(g_idx_list).^2);
                SCALE = (8 * pi^2 * Y_j0_val) / D_full;
                
                qi_list = [];
                for idx = 1:length(g_idx_list)
                    g = g_idx_list(idx);
                    q_i = obj.gaunt_labels(g, 1);
                    q_j = obj.gaunt_labels(g, 2);
                    q_k = obj.gaunt_labels(g, 3);
                    
                    if q_j == q_j0
                        G_val = obj.gaunt_vals(g);
                        P_gain_weights(:, q_i, t) = P_gain_weights(:, q_i, t) + G_val .* Y_all_w(:, q_k);
                        
                        if ~ismember(q_i, qi_list)
                            qi_list = [qi_list, q_i];
                        end
                        
                        if q_i == q_i0
                            P_loss_pre(:, t) = P_loss_pre(:, t) + G_val .* Y_i0_val .* Y_all_w(:, q_k);
                        end
                    end
                end
                P_gain_weights(:,:,t) = P_gain_weights(:,:,t) * SCALE;
                P_loss_pre(:, t) = P_loss_pre(:, t) * SCALE;
                
                % Store padded matrix
                qi_valid_matrix(1:length(qi_list), t) = qi_list;
            end
            
            % =============================================================
            % PHASE 2: Execution Branch (MATLAB vs MEX)
            % =============================================================
            if obj.use_mex
                % --- C++ MEX ENGINE ---
                % Calculate alpha based on the VHS viscosity index omega.
                % For Hard Spheres, alpha = 1.0.
                % For Maxwell Molecules, alpha = 0.0.
                alpha_val = obj.Kernel.alpha;

                % Pass all pre-allocated grids PLUS the new 22nd argument: alpha_val
                obj.R_tensor = compute_rtensor_mex(N_K, N_L, N_rad, N_theta, N_chi, N_eps, N_Q, ...
                    v_nodes, w_nodes, W_vw, mu_theta, mu_chi, eps_vec, W_Total, W_sphere, W_theta_3D, ...
                    R_table, L_triplets, P_loss_pre, P_gain_weights, qi_valid_matrix, alpha_val);
            else
                % --- NATIVE MATLAB ENGINE ---
                obj.R_tensor = zeros(N_K, N_K, N_K, N_L);
                
                sin_theta = sqrt(max(1 - mu_theta.^2, 0));
                sin_chi = sqrt(max(1 - mu_chi.^2, 0));
                cos_eps = cos(eps_vec);
                sin_eps = sin(eps_vec);
                
                u_prime_x_scat = sin_chi .* cos_eps;
                u_prime_y_scat = sin_chi .* sin_eps;
                u_prime_z_scat = mu_chi;
                
                N_pts = N_theta * N_chi * N_eps;
                R_gain_eval = zeros(N_K, obj.L_max + 1, N_pts);
                P_gain = zeros(N_theta, N_chi, N_eps);
                
                for a = 1:N_rad
                    v = v_nodes(a);
                    for b = 1:N_rad
                        w = w_nodes(b);
                        W_vw_scalar = W_rad(a) * W_rad(b);
                        
                        % --- Kinematics (Factored scalars) ---
                        U_x = w .* sin_theta;
                        U_z = v + w .* mu_theta;
                        
                        u_x = -w .* sin_theta;
                        u_z = v - w .* mu_theta;
                        u_mag = sqrt(max(u_x.^2 + u_z.^2, eps));
                        
                        z_scat_x = u_x ./ u_mag;
                        z_scat_z = u_z ./ u_mag;
                        x_scat_x = z_scat_z;
                        x_scat_z = -z_scat_x;
                        
                        u_prime_x = u_prime_x_scat .* x_scat_x + u_prime_z_scat .* z_scat_x;
                        u_prime_y = u_prime_y_scat; 
                        u_prime_z = u_prime_x_scat .* x_scat_z + u_prime_z_scat .* z_scat_z;
                        
                        v_prime_x = 0.5 .* (U_x + u_mag .* u_prime_x);
                        v_prime_y = 0.5 .* (      u_mag .* u_prime_y);
                        v_prime_z = 0.5 .* (U_z + u_mag .* u_prime_z);
                        
                        v_prime_mag = sqrt(max(v_prime_x.^2 + v_prime_y.^2 + v_prime_z.^2, eps));
                        theta_vp = acos(max(min(v_prime_z ./ v_prime_mag, 1.0), -1.0));
                        phi_vp   = atan2(v_prime_y, v_prime_x);
                        
                        B_val = obj.Kernel.exact_kernel(u_mag, mu_chi); 
                        I_loss_inner = sum(sum(B_val .* W_sphere, 3), 2); 
                        B_W_Total = B_val .* W_Total; 
                        
                        Y_all_vp = obj.Basis.SH.evaluate(theta_vp(:), phi_vp(:));
                        Y_all_vp_4D = reshape(Y_all_vp, [N_theta, N_chi, N_eps, N_Q]); 
                        
                        v_prime_flat = v_prime_mag(:);
                        
                        for l_idx = 0:obj.L_max
                            for n_idx = 1:N_K
                                R_gain_eval(n_idx, l_idx+1, :) = obj.Basis.evaluate_radial(n_idx, l_idx, v_prime_flat);
                            end
                        end
                        
                        for t = 1:N_L
                            l_i = L_triplets(t, 1);
                            l_j = L_triplets(t, 2);
                            l_k = L_triplets(t, 3);
                            
                            R_loss_all = R_table(:, l_i + 1, a);
                            R_j_all    = R_table(:, l_j + 1, a);
                            R_k_all    = R_table(:, l_k + 1, b);
                            R_gain_matrix = reshape(R_gain_eval(:, l_i + 1, :), N_K, N_pts);
                            
                            P_gain(:) = 0; 
                            
                            % Fast iteration over padded matrix
                            for idx = 1:N_Q
                                q_i = qi_valid_matrix(idx, t);
                                if q_i == -1
                                    break; 
                                end
                                P_gain = P_gain + Y_all_vp_4D(:,:,:,q_i) .* reshape(P_gain_weights(:, q_i, t), [N_theta, 1, 1]);
                            end
                            
                            Weighted_Integrand = B_W_Total .* P_gain;
                            S_gain_all = R_gain_matrix * Weighted_Integrand(:);
                            
                            S_loss_angular = sum(I_loss_inner .* P_loss_pre(:, t) .* W_theta_3D);
                            
                            Net_S = (S_gain_all - S_loss_angular .* R_loss_all) .* W_vw_scalar;
                            
                            obj.R_tensor(:, :, :, t) = obj.R_tensor(:, :, :, t) + ...
                                Net_S .* reshape(R_j_all, 1, N_K) .* reshape(R_k_all, 1, 1, N_K);
                                
                        end % t loop
                    end % w loop
                end % v loop
            end % End branch
            
            % =============================================================
            % PHASE 3: Enforce Exact Macroscopic Conservation Laws
            % =============================================================
            for t = 1:N_L
                l_i = L_triplets(t, 1);
                if l_i == 0
                    obj.R_tensor(1, :, :, t) = 0; % Mass
                    obj.R_tensor(2, :, :, t) = 0; % Energy
                elseif l_i == 1
                    obj.R_tensor(1, :, :, t) = 0; % Momentum
                end
            end
            
        end

 
        
        %% --- 2. PIVOT EXTRACTION HELPERS ---
        function [pivot_q, pivot_g, q_map, N_L, N_sub_q] = setup_pivots(obj)
            q2l = @(q) floor(sqrt(double(q)-1));
            all_l = q2l(obj.gaunt_labels);
            [unique_l, ~, ic] = unique(all_l, 'rows');
            N_L = size(unique_l, 1);
            
            pivot_q = zeros(N_L, 3);
            pivot_g = zeros(N_L, 1);
            
            for t = 1:N_L
                idx = find(ic == t);
                [~, max_idx] = max(abs(obj.gaunt_vals(idx)));
                best_idx = idx(max_idx);
                pivot_q(t, :) = obj.gaunt_labels(best_idx, :);
                pivot_g(t) = obj.gaunt_vals(best_idx);
            end
            
            N_Q = (obj.L_max + 1)^2;
            unique_pivot_qs = unique(pivot_q(:));
            N_sub_q = length(unique_pivot_qs);
            
            q_map = zeros(N_Q, 1);
            for j = 1:N_sub_q
                q_map(unique_pivot_qs(j)) = j;
            end
        end
        
        function extract_R_tensor(obj, M_sub, pivot_q, pivot_g, q_map, N_L, N_sub_q)
            obj.R_tensor = zeros(obj.K_max+1, obj.K_max+1, obj.K_max+1, N_L);
            
            for t = 1:N_L
                q1 = pivot_q(t, 1);
                q2 = pivot_q(t, 2);
                q3 = pivot_q(t, 3);
                g_val = pivot_g(t);
                
                sq1 = q_map(q1);
                sq2 = q_map(q2);
                sq3 = q_map(q3);
                
                for k1 = 0:obj.K_max
                    for k2 = 0:obj.K_max
                        for k3 = 0:obj.K_max
                            idx1 = k1 * N_sub_q + sq1;
                            idx2 = k2 * N_sub_q + sq2;
                            idx3 = k3 * N_sub_q + sq3;
                            
                            % Division by mass matrix is no longer needed!
                            obj.R_tensor(k1+1, k2+1, k3+1, t) = M_sub(idx1, idx2, idx3) / g_val;
                        end
                    end
                end
            end
        end
        
        %% --- 3. GEOMETRY & ASSEMBLY ---
        function generate_geometry(obj)
            fprintf('Generating Real Gaunt Geometry...\n');
            [c_labels, c_vals] = gaunt_compute_values(obj.L_max);
            N_Q = (obj.L_max + 1)^2;
            
            G_complex = zeros(N_Q, N_Q, N_Q);
            for i = 1:length(c_vals)
                G_complex(c_labels(i,1), c_labels(i,2), c_labels(i,3)) = c_vals(i);
            end
            
            U = complex2real(obj.L_max);
            G_real = G_complex;
            
            G_real = reshape(U * reshape(G_real, N_Q, []), N_Q, N_Q, N_Q);
            G_real = permute(G_real, [2 1 3]);
            G_real = reshape(U * reshape(G_real, N_Q, []), N_Q, N_Q, N_Q);
            G_real = permute(G_real, [2 1 3]); 
            G_real = permute(G_real, [3 2 1]);
            G_real = reshape(conj(U) * reshape(G_real, N_Q, []), N_Q, N_Q, N_Q);
            G_real = permute(G_real, [3 2 1]); 
            
            G_real = real(G_real); 
            [q1, q2, q3] = ind2sub(size(G_real), find(abs(G_real) > 1e-12));
            obj.gaunt_labels = [q1, q2, q3];
            obj.gaunt_vals = G_real(sub2ind(size(G_real), q1, q2, q3));
            
            q2l = @(q) floor(sqrt(double(q)-1));
            all_l = q2l(obj.gaunt_labels);
            [~, ~, obj.ic_map] = unique(all_l, 'rows');
            fprintf('  Found %d strictly non-zero Real Gaunt transitions.\n', length(obj.gaunt_vals));
        end
        
        function C_assembled = assemble_full_tensor(obj)
            fprintf('Assembling FULL Tensor from R-Tensor and Gaunt Values...\n');
            N_terms = obj.Basis.N_terms;
            N_Q = obj.Basis.N_Q;
            C_assembled = zeros(N_terms, N_terms, N_terms);
            
            master_tic = tic;
            for g_idx = 1:length(obj.gaunt_vals)
                q1 = obj.gaunt_labels(g_idx, 1);
                q2 = obj.gaunt_labels(g_idx, 2);
                q3 = obj.gaunt_labels(g_idx, 3);
                g_val = obj.gaunt_vals(g_idx);
                t = obj.ic_map(g_idx);
                
                R_block = obj.R_tensor(:, :, :, t); 
                
                for k1 = 0:obj.K_max
                    for k2 = 0:obj.K_max
                        for k3 = 0:obj.K_max
                            idx1 = k1 * N_Q + q1;
                            idx2 = k2 * N_Q + q2;
                            idx3 = k3 * N_Q + q3;
                            C_assembled(idx1, idx2, idx3) = R_block(k1+1, k2+1, k3+1) * g_val;
                        end
                    end
                end
            end
            fprintf('  Assembly complete in %.4f seconds.\n', toc(master_tic));
        end
        
        function C_naive = generate_full_tensor_naive(obj)
            fprintf('Extracting FULL Tensor via Naive Exact Quadrature...\n');
            N_terms = obj.Basis.N_terms;
            C_naive = zeros(N_terms, N_terms, N_terms);
            
            N_rad = max(30, ceil((obj.D_max + obj.Kernel.N_kernel)/2) + 5); 
            qr = Gauss.generalized_laguerre(N_rad, 0.5); 
            x_nodes = qr.x; 
            
            U_nodes = sqrt(0.5 * x_nodes); 
            u_nodes = sqrt(2.0 * x_nodes); 
            W_rad = qr.w / 2; 
            
            Omega = obj.Kernel.Omega;
            W_ang = obj.Kernel.W_ang;
            N_leb = obj.Kernel.N_leb;
            
            master_tic = tic;
            
            for a = 1:N_rad  
                u_val = u_nodes(a);
                W_u = obj.Kernel.get_expansion_weights(u_val);
                b_0 = obj.Kernel.bn_func(0, u_val);
                loss_coeff = 4 * pi * b_0;
                
                for b = 1:N_rad
                    U_val = U_nodes(b);
                    W_rad_pair = W_rad(a) * W_rad(b);
                    
                    for p = 1:N_leb
                        U_vec = U_val * Omega(p, :);
                        
                        % PHASE 1: Test Functions (Orthonormal)
                        v_prime_all = U_vec + 0.5 * u_val * Omega; 
                        psi_v_all = obj.Basis.evaluate(v_prime_all); 
                        
                        Psi_test = (W_ang .* obj.Kernel.Y_leb)' * psi_v_all; 
                        
                        % PHASE 2: Trial Functions (Orthonormal)
                        U_vec_plus  = U_vec + 0.5 * u_val * Omega;
                        U_vec_minus = U_vec - 0.5 * u_val * Omega;
                        
                        phi_v_all = obj.Basis.evaluate(U_vec_plus);  
                        phi_w_all = obj.Basis.evaluate(U_vec_minus); 
                        
                        Q_all = (obj.Kernel.Y_leb .* W_u) * Psi_test; 
                        Q_all = Q_all - (loss_coeff * phi_v_all);
                        
                        W_tot_all = W_rad_pair * W_ang(p) * W_ang; 
                        Weighted_Q = Q_all .* W_tot_all; 
                        
                        for i = 1:N_terms
                            scaled_phi_v = phi_v_all .* Weighted_Q(:, i); 
                            C_naive(i,:,:) = C_naive(i,:,:) + reshape(scaled_phi_v' * phi_w_all, [1, N_terms, N_terms]);
                        end
                    end
                end
            end
            fprintf('  Naive tensor generation complete in %.2f seconds.\n', toc(master_tic));
        end
    end

    methods (Static)
        function [obj, Basis, Kernel] = load_precalc(K, L, omega)
            % Construct the standardized path
            dir_path = fullfile('src', 'precalc');
            filename = sprintf('collisiontensor_k%d_l%d_vhs_w%.2f.mat', K, L, omega);
            filepath = fullfile(dir_path, filename);
            
            if exist(filepath, 'file')
                fprintf('Loading precalculated tensor: %s\n', filename);
                data = load(filepath);
                obj = data.TensorObj;
                Basis = data.Basis;
                Kernel = data.Kernel;
            else
                error('Precalculated tensor not found at: %s', filepath);
            end
        end
    end

end