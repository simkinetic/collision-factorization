classdef ScatteringKernel < handle
    % SCATTERINGKERNEL - Evaluates the collision kernel physics.
    % Utilizing the exact Wigner-Eckart factorization, the scattering
    % physics rely purely on the relative speed magnitude.
    
    properties
        alpha        % Relative speed exponent (gamma in the paper)
        exact_kernel % Function handle: B(u)
    end
    
    methods
        function obj = ScatteringKernel(alpha)
            % For Hard Spheres, alpha = 1.0
            % For Maxwell Molecules, alpha = 0.0
            if nargin < 1
                alpha = 1.0; % Default to Hard Sphere
            end
            
            obj.alpha = double(alpha);
            
            % Lambda expression mapping exactly to the Julia implementation
            obj.exact_kernel = @(u_mag) (u_mag .* sqrt(2.0)).^obj.alpha;
        end
        
        function B_val = evaluate(obj, u_mag)
            B_val = obj.exact_kernel(u_mag);
        end
    end
end