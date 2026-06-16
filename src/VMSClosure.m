classdef VMSClosure < handle
% VMSCLOSURE - Computes the Hierarchical Variational Multiscale Closure
% Analytically evaluates the collision frequency convolution and the 
% subgrid mass triple-products.

    properties
        K_max      % Trial space radial degree
        L_max      % Trial space angular degree
        K_test     % Test space radial degree (e.g., 2*K_max or K_max + buffer)
        K_N        % Order of the frequency approximation
        L_N        % Angular order of the frequency approximation
        
        K_mat      % Cell array of size (L_max+1) containing [K_N+1 x K_max+1] frequency maps
        T_tensor   % [K_test+1 x K_test+1 x K_N+1 x N_L] Radial Mass Triple Product
        
        Basis      % Reference to SpectralBasis
        Kernel     % Reference to ScatteringKernel
        Tensor     % Reference to GeneralCollisionTensor (for Gaunt maps)
    end
    
    methods
        function obj = VMSClosure(CollisionTensor, K_test, K_N, L_N)
            obj.Tensor = CollisionTensor;
            obj.Basis = CollisionTensor.Basis;
            obj.Kernel = CollisionTensor.Kernel;
            
            obj.K_max = obj.Basis.K_max;
            obj.L_max = obj.Basis.L_max;
            obj.K_test = K_test;
            
            obj.K_N = K_N;
            obj.L_N = L_N;
            
            fprintf('Initializing VMS Subgrid Closure...\n');
            obj.precompute_frequency_matrices();
            obj.precompute_mass_triple_product();
        end
        
        function precompute_frequency_matrices(obj)
            % STEP 1: Computes the 1D convolution mapping f(w) -> \nu(v)
            % By the Funk-Hecke theorem, the convolution of |v-w|^alpha 
            % with Y_lm(w) isolates to the exact same Y_lm(v) mode.
            fprintf('  -> Assembling Collision Frequency Matrices K^(l)...\n');
            
            obj.K_mat = cell(obj.L_max + 1, 1);
            
            % High-precision 2D grid for the offline radial/angular integral
            N_v = 60; N_w = 60; N_mu = 60;
            qr_v = Gauss.halfrange_hermite(N_v, 1.0); v_nodes = qr_v.x;
            qr_w = Gauss.halfrange_hermite(N_w, 1.0); w_nodes = qr_w.x; W_w = qr_w.w;
            qr_mu = Gauss.legendre(N_mu, -1, 1);      mu_nodes = qr_mu.x; W_mu = qr_mu.w;
            
            for l = 0:obj.L_max
                K_l = zeros(obj.K_N + 1, obj.K_max + 1);
                
                % P_l(mu) for the angular integration
                P_l = obj.Basis.evaluate_legendre(l, mu_nodes); 
                
                for k_trial = 0:obj.K_max
                    for k_gamma = 0:obj.K_N
                        
                        integral_val = 0;
                        % 2D Integration over w (incident speed) and mu (deflection angle)
                        for w_idx = 1:N_w
                            w = w_nodes(w_idx);
                            R_trial = obj.Basis.evaluate_radial(k_trial + 1, l, w);
                            
                            mu_int = 0;
                            for mu_idx = 1:N_mu
                                mu = mu_nodes(mu_idx);
                                
                                % Relative speed: u = |v - w|
                                % We evaluate this for a specific v to build the function \nu(v)
                                % Wait, we need to project \nu(v) onto R_{k_\gamma}(v)
                                % Let's move v outside!
                                
                                % (Inside v-loop below, see optimized version)
                            end
                        end
                    end
                end
            end
            
            % --- OPTIMIZED FAST EVALUATION ---
            % Since it's offline, we loop over v, w, and mu.
            for l = 0:obj.L_max
                K_l = zeros(obj.K_N + 1, obj.K_max + 1);
                P_l = obj.Basis.evaluate_legendre(l, mu_nodes); 
                
                for k_trial = 0:obj.K_max
                    for k_gamma = 0:obj.K_N
                        total_int = 0;
                        
                        for v_idx = 1:N_v
                            v = v_nodes(v_idx);
                            R_gamma = obj.Basis.evaluate_radial(k_gamma + 1, l, v);
                            
                            for w_idx = 1:N_w
                                w = w_nodes(w_idx);
                                R_trial = obj.Basis.evaluate_radial(k_trial + 1, l, w);
                                
                                mu_int = 0;
                                for mu_idx = 1:N_mu
                                    mu = mu_nodes(mu_idx);
                                    u_mag = sqrt(max(v^2 + w^2 - 2*v*w*mu, 0));
                                    B_val = obj.Kernel.evaluate(u_mag);
                                    mu_int = mu_int + W_mu(mu_idx) * B_val * P_l(mu_idx);
                                end
                                % 2 * pi from azimuthal symmetry of the target
                                total_int = total_int + qr_v.w(v_idx) * W_w(w_idx) * R_gamma * R_trial * (2 * pi * mu_int);
                            end
                        end
                        K_l(k_gamma + 1, k_trial + 1) = total_int;
                    end
                end
                obj.K_mat{l+1} = K_l;
            end
        end
        
        function precompute_mass_triple_product(obj)
            % STEP 2: Computes T_\gamma (Triple Radial Product)
            % Integrates: R_{k1,l1}(v) * R_{k2,l2}(v) * R_{k_\gamma, l_\gamma}(v) * v^2 e^{-v^2}
            fprintf('  -> Assembling Radial Mass Triple-Products T_gamma...\n');
            
            N_L = max(obj.Tensor.ic_map);
            obj.T_tensor = zeros(obj.K_test + 1, obj.K_test + 1, obj.K_N + 1, N_L);
            
            % Half-range Hermite grid natively absorbs the v^2 * exp(-v^2) measure!
            N_v = ceil((2 * obj.K_test + obj.K_N + 1.5 * obj.L_max) / 2.0) + 20;
            qr_v = Gauss.halfrange_hermite(N_v, 1.0);
            v_nodes = qr_v.x;
            W_v = qr_v.w;
            
            for t_chan = 1:N_L
                % Extract the polar degrees involved in this geometric transition
                % L_triplets stores [l_test, l_trial, l_gamma]
                l_1 = obj.Tensor.L_triplets(t_chan, 1); 
                l_2 = obj.Tensor.L_triplets(t_chan, 2); 
                l_g = obj.Tensor.L_triplets(t_chan, 3);
                
                % If the frequency order isn't high enough to trigger this, skip
                if l_g > obj.L_N
                    continue;
                end
                
                for k1 = 0:obj.K_test
                    for k2 = 0:obj.K_test
                        for k_g = 0:obj.K_N
                            
                            val = 0;
                            for i = 1:N_v
                                v = v_nodes(i);
                                R1 = obj.Basis.evaluate_radial(k1 + 1, l_1, v);
                                R2 = obj.Basis.evaluate_radial(k2 + 1, l_2, v);
                                Rg = obj.Basis.evaluate_radial(k_g + 1, l_g, v);
                                
                                val = val + W_v(i) * R1 * R2 * Rg;
                            end
                            
                            obj.T_tensor(k1 + 1, k2 + 1, k_g + 1, t_chan) = val;
                        end
                    end
                end
            end
        end
    end
end