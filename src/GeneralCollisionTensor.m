classdef GeneralCollisionTensor < handle
% GENERALCOLLISIONTENSOR - Exact Spectral Boltzmann Collision Operator
    
    properties
        K_max
        L_max
        I_max          % Maximum internal energy polynomial degree
        D_max
        K_test         % Test space radial degree (for VMS projection)
        I_test         % Test space internal degree
        nu             % Internal degrees of freedom parameter: (D-5)/2
        
        % [K_test+1 x K_max+1 x K_max+1 x I_test+1 x I_max+1 x I_max+1 x N_L]
        R_tensor       
        
        gaunt_labels   % [N_G x 3] Non-zero Real Gaunt transitions
        gaunt_vals     % [N_G x 1] Real Gaunt values
        ic_map         % [N_G x 1] Mapping from transitions to L-triplets
        L_triplets     % [N_L x 3] Valid polar degree interaction channels
        
        Basis          % SpectralBasis object
        Kernel         % ScatteringKernel object
        
        use_mex        % Boolean flag to toggle C++ MEX execution
    end
    
    methods
        function obj = GeneralCollisionTensor(Basis, Kernel, K_test, I_test)
            obj.Basis = Basis;
            obj.Kernel = Kernel;
            
            obj.K_max = Basis.K_max;
            obj.L_max = Basis.L_max;
            
            % Polyatomic extensions assumed to exist in Basis
            if isprop(Basis, 'I_max')
                obj.I_max = Basis.I_max;
                obj.nu = Basis.nu;
            else
                obj.I_max = 0; % Fallback to monatomic
                obj.nu = 0;
            end
            
            obj.D_max = 2 * obj.K_max + obj.L_max;
            
            % Handle optional VMS test space bounds
            if nargin < 3 || isempty(K_test)
                obj.K_test = obj.K_max;
            else
                obj.K_test = K_test;
            end
            
            if nargin < 4 || isempty(I_test)
                obj.I_test = obj.I_max;
            else
                obj.I_test = I_test;
            end
            
            % Generate geometry maps
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
        
        function generate_R_tensor_sumfac(obj, radial_pad, angular_pad, internal_pad)
            if nargin < 2, radial_pad = 10; end
            if nargin < 3, angular_pad = 10; end
            if nargin < 4, internal_pad = 10; end
            
            fprintf('Initializing Sum-Factorized 9D Polyatomic Quadrature Grids...\n');
            obj.R_tensor(:) = 0;
            
            N_K = obj.K_max + 1;
            N_I = obj.I_max + 1;
            N_Q = obj.Basis.N_Q;
            alpha_kernel = obj.Kernel.alpha;
            nu_val = obj.nu;
            
            % 1. EXACT SPATIAL GRID SIZING (with padding)
            N_x = radial_pad + ceil((3 * obj.K_max + 1.5 * obj.L_max + 3) / 2.0);
            N_u1 = radial_pad + 4 * obj.K_max + 3 * obj.L_max + 4;
            N_t1 = angular_pad + obj.K_max + ceil(1.5 * obj.L_max) + 1;
            N_y2 = angular_pad + 4 * obj.K_max + 3 * obj.L_max + 4;
            N_t2 = radial_pad + 3 * obj.K_max + floor(1.5 * obj.L_max) + 3;
            N_chi = obj.K_max + ceil(0.5 * obj.L_max) + 1;
            N_eps = 2 * obj.K_max + obj.L_max + 1;
            
            % 2. EXACT INTERNAL GRID SIZING (from Table 1)
            N_I_nodes = ceil((obj.K_max + 2 * obj.I_max + 1) / 2.0) + internal_pad; 
            N_J_nodes = ceil((obj.K_max + 2 * obj.I_max + 1) / 2.0) + internal_pad;
            N_r_nodes = obj.I_max + 1 + internal_pad;
            N_R_nodes = ceil((obj.K_max + 2 * obj.I_max + 1) / 2.0) + internal_pad;

            % Spatial 1D Grids
            qr_x = Gauss.generalized_laguerre(N_x, alpha_kernel / 2.0);
            x_nodes = qr_x.x; W_x = qr_x.w;
            
            qr_u1 = Gauss.legendre(N_u1, 0, 1); u1_nodes = qr_u1.x; W_u1 = qr_u1.w;
            qr_t1 = Gauss.legendre(N_t1, 0, 1); t1_nodes = qr_t1.x; W_t1 = qr_t1.w;
            
            qr_y2 = Gauss.legendre(N_y2, 0, 1); y2_nodes = qr_y2.x; W_y2 = qr_y2.w;
            qr_t2 = Gauss.legendre(N_t2, 0, 1); t2_nodes = qr_t2.x; W_t2 = qr_t2.w;
            
            qr_chi = Gauss.legendre(N_chi, -1, 1); mu_chi = qr_chi.x; W_chi = qr_chi.w;
            
            eps_vec = (2*pi*(0:(N_eps-1))/N_eps)';
            W_eps = (2*pi/N_eps) * ones(N_eps, 1);
            
            % Polyatomic 1D Grids (Laguerre and Jacobi)
            qr_I = Gauss.generalized_laguerre(N_I_nodes, nu_val);
            I_nodes = qr_I.x; W_I = qr_I.w;
            
            qr_J = Gauss.generalized_laguerre(N_J_nodes, nu_val);
            J_nodes = qr_J.x; W_J = qr_J.w;
            
            qr_r = Gauss.jacobi(N_r_nodes, nu_val, nu_val, 0, 1);
            r_nodes = qr_r.x; W_r = qr_r.w;
            
            qr_R = Gauss.jacobi(N_R_nodes, 2*nu_val + 1, 0.5, 0, 1);
            R_nodes = qr_R.x; W_R = qr_R.w;
            
            K_test_val = obj.K_test;
            N_K_rad = max(N_K, K_test_val + 1);
            
            I_test_val = obj.I_test;
            N_I_rad = max(N_I, I_test_val + 1);
            
            % Radial Normalization Cache
            RadialNorm = zeros(N_K_rad, obj.L_max + 1);
            for n_idx = 1:N_K_rad
                k = n_idx - 1;
                for l = 0:obj.L_max
                    al = l + 0.5;
                    ln_M_ii = -log(2) + gammaln(k + al + 1) - gammaln(k + 1);
                    RadialNorm(n_idx, l+1) = exp(-0.5 * ln_M_ii);
                end
            end
            
            % Internal Energy Normalization Cache (Generalized Laguerre)
            InternalNorm = zeros(N_I_rad, 1);
            for i_idx = 1:N_I_rad
                i_poly = i_idx - 1;
                ln_N_ii = gammaln(i_poly + 1) - gammaln(i_poly + nu_val + 1);
                InternalNorm(i_idx) = exp(0.5 * ln_N_ii);
            end
            
            % Spherical Harmonic Normalization
            SH_Norm = zeros(N_Q, 1);
            for l = 0:obj.L_max
                for m = -l:l
                    abs_m = abs(m);
                    base_norm = sqrt( ((2*l + 1) / (4*pi)) * exp(gammaln(l - abs_m + 1) - gammaln(l + abs_m + 1)) );
                    q_idx = l^2 + l + m + 1;
                    if m == 0
                        SH_Norm(q_idx) = base_norm;
                    else
                        SH_Norm(q_idx) = sqrt(2) * base_norm;
                    end
                end
            end
            
            % Precompute Geometries
            N_L = max(obj.ic_map);
            L_triplets = zeros(N_L, 3);
            qi_valid_mat = -1 * ones(N_Q, N_L); % Padded with -1 for C++
            
            P_loss_p1 = zeros(N_t1, N_u1, N_L);
            P_gain_p1 = zeros(N_t1, N_u1, N_Q, N_L);
            P_loss_p2 = zeros(N_y2, N_L);
            P_gain_p2 = zeros(N_y2, N_Q, N_L);
            
            for t_chan = 1:N_L
                g_indices = find(obj.ic_map == t_chan);
                q_trip_first = obj.gaunt_labels(g_indices(1), :);
                
                q2l = @(q) floor(sqrt(double(q)-1));
                l_i = q2l(q_trip_first(1)); l_j = q2l(q_trip_first(2)); l_k = q2l(q_trip_first(3));
                L_triplets(t_chan, :) = [l_i, l_j, l_k];
                
                q_i0 = l_i^2 + l_i + 1; q_j0 = l_j^2 + l_j + 1;
                Y_i0_val = sqrt((2*l_i + 1) / (4*pi));
                Y_j0_val = sqrt((2*l_j + 1) / (4*pi));
                
                D_full = sum(obj.gaunt_vals(g_indices).^2);
                SCALE = (8 * pi^2 * Y_j0_val) / D_full;
                
                qi_list = [];
                
                for g = g_indices'
                    q_i = obj.gaunt_labels(g, 1); q_j = obj.gaunt_labels(g, 2); q_k = obj.gaunt_labels(g, 3);
                    if q_j == q_j0
                        G_val = obj.gaunt_vals(g);
                        if ~ismember(q_i, qi_list), qi_list = [qi_list, q_i]; end
                        
                        % Patch 1
                        for u_idx = 1:N_u1
                            u = u1_nodes(u_idx);
                            for t_idx = 1:N_t1
                                y = u * t1_nodes(t_idx);
                                mu_val = max(min(1 - 2*y^2, 1), -1);
                                Y_eval = obj.Basis.SH.evaluate(acos(mu_val), 0);
                                
                                P_gain_p1(t_idx, u_idx, q_i, t_chan) = P_gain_p1(t_idx, u_idx, q_i, t_chan) + G_val * Y_eval(q_k);
                                if q_i == q_i0
                                    P_loss_p1(t_idx, u_idx, t_chan) = P_loss_p1(t_idx, u_idx, t_chan) + G_val * Y_i0_val * Y_eval(q_k);
                                end
                            end
                        end
                        
                        % Patch 2
                        for y_idx = 1:N_y2
                            y = y2_nodes(y_idx);
                            mu_val = max(min(1 - 2*y^2, 1), -1);
                            Y_eval = obj.Basis.SH.evaluate(acos(mu_val), 0);
                            
                            P_gain_p2(y_idx, q_i, t_chan) = P_gain_p2(y_idx, q_i, t_chan) + G_val * Y_eval(q_k);
                            if q_i == q_i0
                                P_loss_p2(y_idx, t_chan) = P_loss_p2(y_idx, t_chan) + G_val * Y_i0_val * Y_eval(q_k);
                            end
                        end
                    end
                end
                
                P_gain_p1(:,:,:,t_chan) = P_gain_p1(:,:,:,t_chan) * SCALE;
                P_loss_p1(:,:,t_chan) = P_loss_p1(:,:,t_chan) * SCALE;
                P_gain_p2(:,:,t_chan) = P_gain_p2(:,:,t_chan) * SCALE;
                P_loss_p2(:,t_chan) = P_loss_p2(:,t_chan) * SCALE;
                
                qi_valid_mat(1:length(qi_list), t_chan) = qi_list;
            end
            
            obj.L_triplets = L_triplets;
            
            % NEW: Call Polyatomic MEX (Requires passing all 4 new grids + nu + InternalNorm)
            obj.R_tensor = compute_rtensor_polyatomic_sumfac_mex(obj.K_max, obj.I_max, N_L, N_Q, alpha_kernel, nu_val, ...
                x_nodes, W_x, u1_nodes, W_u1, t1_nodes, W_t1, ...
                y2_nodes, W_y2, t2_nodes, W_t2, mu_chi, W_chi, eps_vec, W_eps, ...
                I_nodes, W_I, J_nodes, W_J, r_nodes, W_r, R_nodes, W_R, ...
                RadialNorm, InternalNorm, SH_Norm, L_triplets, qi_valid_mat, ...
                P_loss_p1, P_gain_p1, P_loss_p2, P_gain_p2, ...
                K_test_val, I_test_val);
                
            % Enforce Conservation
            % (Update conservation logic for polyatomic cases as needed)
        end
        
        %% --- 2. PIVOT EXTRACTION HELPERS ---
        % (Setup pivots remains identical)
        
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
            fprintf('Assembling FULL Tensor from Polyatomic R-Tensor and Gaunt Values...\n');
            
            % Update for polyatomic indices (k, i, q)
            N_Q = obj.Basis.N_Q;
            N_I_trial = obj.I_max + 1;
            N_K_trial = obj.K_max + 1;
            N_trial_terms = N_K_trial * N_I_trial * N_Q;
            
            N_I_test = obj.I_test + 1;
            N_K_test = obj.K_test + 1;
            N_test_terms = N_K_test * N_I_test * N_Q;
            
            C_assembled = zeros(N_test_terms, N_trial_terms, N_trial_terms);
            
            master_tic = tic;
            for g_idx = 1:length(obj.gaunt_vals)
                q1 = obj.gaunt_labels(g_idx, 1);
                q2 = obj.gaunt_labels(g_idx, 2);
                q3 = obj.gaunt_labels(g_idx, 3);
                g_val = obj.gaunt_vals(g_idx);
                t = obj.ic_map(g_idx);
                
                R_block = obj.R_tensor(:, :, :, :, :, :, t); 
                
                % Inverted loop order: Trailing dimensions (k3, i3) on the outside,
                % leading dimensions (k1, i1) on the inside for column-major efficiency.
                for k3 = 0:obj.K_max
                    for i3 = 0:obj.I_max
                        idx3 = (k3 * N_I_trial + i3) * N_Q + q3;
                        
                        for k2 = 0:obj.K_max
                            for i2 = 0:obj.I_max
                                idx2 = (k2 * N_I_trial + i2) * N_Q + q2;
                                
                                for k1 = 0:obj.K_test
                                    for i1 = 0:obj.I_test
                                        idx1 = (k1 * N_I_test + i1) * N_Q + q1;
                                        
                                        C_assembled(idx1, idx2, idx3) = C_assembled(idx1, idx2, idx3) + ...
                                            R_block(k1+1, k2+1, k3+1, i1+1, i2+1, i3+1) * g_val;
                                    end
                                end
                            end
                        end
                    end
                end
            end
            fprintf('  Assembly complete in %.4f seconds.\n', toc(master_tic));
        end
    end
end