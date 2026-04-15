classdef TestScatteringKernel < matlab.unittest.TestCase
    % TESTSCATTERINGKERNEL - Unit tests for the ScatteringKernel class
    % Validates isotropic truncation and anisotropic spectral convergence.
    
    properties
        K_max = 2;      % Standard radial basis degree
        L_max = 2;      % Standard angular basis degree
        u_ref = 1.0;    % Reference relative speed
    end
    
    methods (Test)
        
        function testIsotropicMaxwellian(testCase)
            % TEST 1: Pure Isotropic Maxwellian Molecule
            % B(u, cos_chi) = 1.0 (No dependence on u or chi)
            % Expected: N_kernel should perfectly truncate to 0.
            
            alpha = 0.0; % Maxwellian physics
            N_max = 10;
            tol = 1e-10; % Very tight tolerance
            
            Kernel = ScatteringKernel(testCase.K_max, testCase.L_max, N_max, alpha, tol);
            
            % Verify that it truncated to exactly 1 term (Degree 0)
            testCase.verifyEqual(Kernel.N_kernel, 0, ...
                'Isotropic Maxwellian should truncate to N_kernel = 0.');
        end
        
        function testIsotropicHardSpheres(testCase)
            % TEST 2: Isotropic Hard Spheres
            % B(u, cos_chi) = u (Dependence on u, but completely isotropic)
            % Expected: N_kernel should perfectly truncate to 0.
            
            alpha = 1.0; % Hard Sphere physics
            N_max = 10;
            tol = 1e-10; 
            
            Kernel = ScatteringKernel(testCase.K_max, testCase.L_max, N_max, alpha, tol);
            
            % Verify that it truncated to exactly 1 term (Degree 0)
            testCase.verifyEqual(Kernel.N_kernel, 0, ...
                'Isotropic Hard Spheres should truncate to N_kernel = 0.');
        end
        
        function testAnisotropicConvergence(testCase)
            % TEST 3: Spectral Convergence of Anisotropic Kernel
            % Replicates the tutorial script and asserts the error at N=12
            % is near 1e-6 as shown in the visual plot.
            
            omega = 0.5;
            alpha = 2.0 * (1.0 - omega);
            kappa = 5.0;
            
            N_max = 16;
            tol = 0; % Force evaluation up to N_max
            
            % To replicate D_max = 15 for high-precision quadrature, 
            % we set K_max=0 and L_max=15 (since D_max = 2*K_max + L_max)
            K_high = 0;
            L_high = 15;
            
            % Initialize Kernel
            Kernel = ScatteringKernel(K_high, L_high, N_max, alpha, tol);
            
            % Overwrite the kernel to inject the exponential anisotropy
            Kernel.exact_kernel = @(u, cos_chi) (u.^alpha) .* exp(kappa .* cos_chi);
            
            % Setup Ground Truth (Incoming velocity along X-axis)
            u_hat = [1, 0, 0];
            u_theta = pi / 2; % acos(0)
            u_phi = 0;        % atan2(0, 1)
            
            cos_chi_grid = Kernel.Omega * u_hat';
            exact_B_vals = Kernel.evaluate(testCase.u_ref, cos_chi_grid);
            norm_exact = sqrt(sum(Kernel.W_ang .* exact_B_vals.^2));
            
            % Extract expansion weights and Spherical Harmonics
            W_u_full = Kernel.get_expansion_weights(testCase.u_ref);
            SH = SphericalHarmonics(N_max);
            Y_u_hat_full = SH.evaluate(u_theta, u_phi);
            
            % We want to test the relative L2 error specifically at N=12
            N_test = 12;
            N_spec_trunc = (N_test + 1)^2;
            
            W_u_trunc = W_u_full;
            W_u_trunc(N_spec_trunc + 1 : end) = 0;
            
            % Reconstruct approximation
            approx_B_vals = Kernel.Y_leb * (W_u_trunc .* Y_u_hat_full)';
            
            % Compute Relative L2 Error
            error_squared = sum(Kernel.W_ang .* (exact_B_vals - approx_B_vals).^2);
            rel_L2_error = sqrt(error_squared) / norm_exact;
            
            % Assert that at N=12, the error has dropped below 2e-6
            % (Allowing a tiny bit of buffer around the 1e-6 mark)
            testCase.verifyLessThan(rel_L2_error, 2e-6, ...
                sprintf('Error at N=12 (%.2e) did not match expected convergence graph.', rel_L2_error));
        end
        
    end
end