classdef ScatteringKernel < handle
    % SCATTERINGKERNEL - Evaluates the collision kernel physics.
    % Supports both monatomic VHS (Variable Hard Sphere) models 
    % and extended Polyatomic hard potential models.
    
    properties
        model_type   % String: 'VHS', 'Polyatomic'
        alpha        % Relative speed exponent (gamma in the paper)
        mass         % Molecular mass (default: 1.0)
        exact_kernel % Function handle: B(u, I, J)
    end
    
    methods
        function obj = ScatteringKernel(varargin)
            % Handle backwards compatibility and new string-based signatures
            if nargin == 1 && isnumeric(varargin{1})
                % Legacy call: ScatteringKernel(alpha)
                obj.model_type = 'VHS';
                obj.alpha = double(varargin{1});
                obj.mass = 1.0;
            elseif nargin >= 2
                % Standard call: ScatteringKernel('VHS', gamma)
                obj.model_type = varargin{1};
                obj.alpha = double(varargin{2});
                if nargin >= 3
                    obj.mass = double(varargin{3});
                else
                    obj.mass = 1.0;
                end
            else
                % Default
                obj.model_type = 'VHS';
                obj.alpha = 1.0;
                obj.mass = 1.0;
            end
            
            % Lambda expression mapping to the physical models
            switch upper(obj.model_type)
                case {'VHS', 'HARDSPHERES', 'MAXWELL'}
                    % Pure translational dependence (Standard VHS / Maxwell)
                    obj.exact_kernel = @(u_mag, I, J) (u_mag .* sqrt(2.0)).^obj.alpha;
                    
                case 'POLYATOMIC'
                    % Extended Grad assumption for polyatomic hard potentials
                    % B = u^gamma + ((I + J) / m)^(gamma / 2)
                    obj.exact_kernel = @(u_mag, I, J) (u_mag .* sqrt(2.0)).^obj.alpha + ...
                                                      ((I + J) ./ obj.mass).^(obj.alpha / 2.0);
                                                      
                otherwise
                    error('Unknown scattering model type: %s', obj.model_type);
            end
        end
        
        function B_val = evaluate(obj, u_mag, I, J)
            % EVALUATE Computes the collision frequency.
            % I and J are optional (default to 0 for monatomic/VHS calls)
            if nargin < 3
                I = zeros(size(u_mag));
                J = zeros(size(u_mag));
            end
            B_val = obj.exact_kernel(u_mag, I, J);
        end
    end
end 