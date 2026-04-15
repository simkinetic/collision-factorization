classdef SphericalHarmonics
    % Evaluates Real Spherical Harmonics up to degree L.
    
    properties
        L       % Maximum degree
        N_spec  % Total number of basis functions: (L+1)^2
    end
    
    methods
        function obj = SphericalHarmonics(L_max)
            obj.L = L_max;
            obj.N_spec = (L_max + 1)^2;
        end
        
        function Y = evaluate(obj, theta, phi)
            % theta: [N x 1] array of polar angles in [0, pi]
            % phi:   [N x 1] array of azimuthal angles in [0, 2*pi]
            % Returns Y: [N x N_spec] matrix of evaluated harmonics.
            
            theta = theta(:);
            phi = phi(:);
            N_points = length(theta);
            
            Y = zeros(N_points, obj.N_spec);
            x = cos(theta);
            
            for l = 0:obj.L
                % Evaluate Associated Legendre Polynomials for all m=0..l
                Plm = legendre(l, x'); 
                
                for m = -l:l
                    abs_m = abs(m);
                    
                    % Normalization factor with gammaln for stability
                    norm_factor = sqrt( ((2*l + 1) / (4*pi)) * ...
                                        exp(gammaln(l - abs_m + 1) - gammaln(l + abs_m + 1)) );
                    
                    P = Plm(abs_m + 1, :)'; 
                    idx = obj.get_mode_index(l, m);
                    
                    if m > 0
                        Y(:, idx) = sqrt(2) * norm_factor .* P .* cos(abs_m * phi);
                    elseif m == 0
                        Y(:, idx) = norm_factor .* P;
                    else % m < 0
                        Y(:, idx) = sqrt(2) * norm_factor .* P .* sin(abs_m * phi);
                    end
                end
            end
        end
        
        function idx = get_mode_index(obj, l, m)
            % Maps the degree l and order m to a linear 1D index
            if l < 0 || l > obj.L || abs(m) > l
                error('Invalid mode (l,m). Must satisfy 0 <= l <= L and -l <= m <= l.');
            end
            idx = l^2 + l + m + 1;
        end
    end
end