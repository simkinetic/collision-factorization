classdef ScatteringKernel < handle
    % SCATTERINGKERNEL - Dynamic Physics and Quadrature Manager
    
    properties
        exact_kernel % Function handle: B(u, cos_chi)
        alpha        % Viscosity index for MEX C++ evaluation
        N_kernel     % Dynamically found maximum Legendre degree
        req_deg      % Exact polynomial degree required to prevent aliasing
        
        N_theta      % Number of polar Gauss-Legendre nodes
        N_phi        % Number of azimuthal Uniform nodes
        N_leb        % Total angular points
        
        Omega        % [N_leb x 3] Flattened Cartesian grid
        W_ang        % [N_leb x 1] Flattened Quadrature weights
        Y_leb        % [N_leb x (N_kernel+1)^2] Evaluated Spherical Harmonics
        
        bn_func      % Function handle for b_n extraction
    end
    
    methods
        function obj = ScatteringKernel(K_max, L_max, N_max, alpha, tol)
            % Set default tolerance if not provided
            if nargin < 5
                tol = 1e-6;
            end
            
            % Standard reference relative speed for tolerance checking
            u_ref = 1.0; 
            
            % Store physical index
            obj.alpha = alpha;
            
            % RECONSTRUCT THE KERNEL INTERNALLY
            obj.exact_kernel = @(u, cos_chi) u.^alpha + 0 * cos_chi;
            
            % Assign function handle for external classes
            obj.bn_func = @(n, u) obj.compute_bn(n, u);
            
            % Compute maximum degree D_max needed for aliasing bounds
            D_max = 2 * K_max + L_max;
            
            % 1. Dynamically find optimal N_kernel
            obj.N_kernel = obj.find_optimal_N_kernel(N_max, tol, u_ref);
            fprintf('ScatteringKernel: Dynamically truncated at N_kernel = %d.\n', obj.N_kernel);
            
            % 2. Calculate rigorous alias-free degree requirement
            obj.req_deg = max(3 * D_max, 2 * D_max + obj.N_kernel);
            
            % 3. Generate the Product Grid (Gauss-Legendre x Uniform)
            obj.generate_product_grid();
            
            % 4. Pre-evaluate Real Spherical Harmonics on the grid
            obj.evaluate_sh_basis();
        end

        function B_val = evaluate(obj, u, mu_chi)
            % EVALUATE - Wraps the exact kernel function handle
            % u      : relative speed
            % mu_chi : cosine of the deflection angle
            B_val = obj.exact_kernel(u, mu_chi);
        end
        
        function W_expansion = get_expansion_weights(obj, u)
            % Returns the (N_kernel+1)^2 expansion weights for the addition theorem
            W_expansion = zeros(1, (obj.N_kernel + 1)^2);
            
            for n = 0:obj.N_kernel
                % Dynamically extract the Legendre coefficient for this 'u'
                bn = obj.compute_bn(n, u);
                
                % Wigner addition theorem scaling factor: 4pi / (2n+1)
                const = bn * (4 * pi) / (2 * n + 1);
                
                % Assign to all m in [-n, n]
                idx = (n^2 + 1) : ((n + 1)^2);
                W_expansion(idx) = const;
            end
        end
        
        function bn = compute_bn(obj, n, u)
            % High-precision extraction of b_n(u) using internal Gauss-Legendre
            qr_leg = Gauss.legendre(60, -1, 1);
            B_vals = obj.exact_kernel(u, qr_leg.x);
            
            P_all = legendre(n, qr_leg.x'); 
            P_n = P_all(1, :)'; 
            
            bn = ((2*n + 1) / 2) * sum(qr_leg.w .* B_vals .* P_n);
        end
    end
    
    methods (Access = private)
        function N_opt = find_optimal_N_kernel(obj, N_max, tol, u_ref)
            qr_fine = Gauss.legendre(100, -1, 1);
            B_vals = obj.exact_kernel(u_ref, qr_fine.x);
            true_energy = sum(qr_fine.w .* (B_vals.^2));
            
            cum_energy = 0;
            for n = 0:N_max
                P_all = legendre(n, qr_fine.x'); 
                P_n = P_all(1, :)'; 
                
                b_n = ((2*n + 1) / 2) * sum(qr_fine.w .* B_vals .* P_n);
                term_energy = (2 / (2*n + 1)) * b_n^2;
                cum_energy = cum_energy + term_energy;
                
                rel_err = sqrt(max(0, true_energy - cum_energy) / true_energy);
                
                if rel_err < tol
                    N_opt = n;
                    return;
                end
            end
            N_opt = N_max;
            fprintf('Warning: Kernel did not reach tolerance. Truncated at N_max=%d\n', N_max);
        end
        
        function generate_product_grid(obj)
            % Gauss-Legendre for polar (mu), Uniform for azimuthal (phi)
            obj.N_theta = ceil((obj.req_deg + 1) / 2);
            obj.N_phi = obj.req_deg + 1;
            obj.N_leb = obj.N_theta * obj.N_phi;
            
            fprintf('ScatteringKernel: Generated %dx%d Product Grid (%d points) for Degree %d.\n', ...
                obj.N_theta, obj.N_phi, obj.N_leb, obj.req_deg);
            
            % Generate 1D Grids
            qr_theta = Gauss.legendre(obj.N_theta, -1, 1);
            mu = qr_theta.x;
            w_mu = qr_theta.w;
            
            phi = linspace(0, 2*pi, obj.N_phi + 1)'; 
            phi(end) = []; 
            w_phi = (2*pi / obj.N_phi) * ones(obj.N_phi, 1);
            
            % Tensor Product (Meshgrid)
            [PHI, MU] = meshgrid(phi, mu);
            W_ANG = w_mu * w_phi'; 
            
            % Flatten matrices to allow BLAS DGEMM tensor contractions
            obj.Omega = [sqrt(1-MU(:).^2).*cos(PHI(:)), sqrt(1-MU(:).^2).*sin(PHI(:)), MU(:)];
            obj.W_ang = W_ANG(:);
        end
        
        function evaluate_sh_basis(obj)
            % Recover theta from mu
            mu_flat = obj.Omega(:, 3);
            theta_flat = acos(mu_flat);
            
            % Recover phi (atan2 returns [-pi, pi], map to [0, 2pi])
            phi_flat = atan2(obj.Omega(:, 2), obj.Omega(:, 1));
            phi_flat(phi_flat < 0) = phi_flat(phi_flat < 0) + 2*pi;
            
            % Instantiate and evaluate using your SphericalHarmonics class
            SH = SphericalHarmonics(obj.N_kernel);
            obj.Y_leb = SH.evaluate(theta_flat, phi_flat);
        end
    end
end