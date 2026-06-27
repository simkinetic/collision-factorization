classdef ScatteringKernel < handle
    % SCATTERINGKERNEL - Evaluates the collision kernel physics.
    % Supports both monatomic VHS (Variable Hard Sphere) models 
    % and extended Polyatomic hard potential models.
    
    properties
        model_type   % String: 'VHS', 'Polyatomic', 'DSMC'
        alpha        % Relative speed exponent (gamma in the paper; = zeta for DSMC)
        mass         % Molecular mass (default: 1.0)
        exact_kernel % Function handle: B(u, I, J)

        % --- DSMC frozen/non-frozen convex-split model (Djordjic et al. 2023, eqs 53-55) ---
        omega        % Non-frozen weight p_int in [0,1] (frozen weight = 1 - omega)
        zeta         % Translational exponent = 2*(1 - s_visc); kernel ~ |u|^zeta
        delta        % Internal degrees of freedom (nu = delta/2 - 1)
        nu           % Internal-energy Laguerre parameter = delta/2 - 1
        K_delta      % Monatomic-consistency constant 2*Gamma(delta+3/2)/(sqrt(pi)*Gamma(delta/2)^2)
        C_vhs        % VHS constant of the non-frozen channel
        C_vhs_frozen % VHS constant of the frozen (elastic) channel

        % --- Extended model internal-energy modulation (Djordjic et al. 2023, eq 43) ---
        % Non-frozen factor  1 + eta_hat *( (r(1-R)i)^(zeta_hat/2) + ((1-r)(1-R)i*)^(zeta_hat/2) )
        % Frozen     factor  1 + eta_hat_f*( i^zeta_hat_f + i*^zeta_hat_f ),   i = I/E, i* = J/E.
        % All default to 0 -> reduces exactly to the base DSMC (eq 54) kernel.
        eta_hat      % Non-frozen internal-energy coefficient (>= -1/2)
        zeta_hat     % Non-frozen internal-energy exponent    (>= 0)
        eta_hat_f    % Frozen internal-energy coefficient      (>= -1/2)
        zeta_hat_f   % Frozen internal-energy exponent         (>= 0)
    end
    
    methods
        function obj = ScatteringKernel(varargin)
            % DSMC convex-split model: ScatteringKernel('DSMC', params_struct)
            % params fields: omega, zeta (or s_visc), delta, and optionally
            %                C_vhs, C_vhs_frozen, mass.
            if nargin >= 2 && ischar(varargin{1}) && strcmpi(varargin{1}, 'DSMC')
                obj.init_dsmc(varargin{2});
                return;
            end

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

        function init_dsmc(obj, p)
            % INIT_DSMC  Configure the DSMC frozen/non-frozen convex-split kernel
            % (Djordjic, Oblapenko, Pavic-Colic, Torrilhon 2023, eqs 53-55).
            %   B = omega * K_delta * |u|^zeta * R^(zeta/2)        (non-frozen, BL)
            %     + (1-omega) * C_vhs_frozen * |u|^zeta            (frozen, elastic)
            % p is a struct; supply either zeta or s_visc (zeta = 2*(1-s_visc)).
            if ~isstruct(p)
                error('DSMC model requires a parameter struct.');
            end

            obj.model_type = 'DSMC';

            if isfield(p, 'zeta')
                obj.zeta = double(p.zeta);
            elseif isfield(p, 's_visc')
                obj.zeta = 2.0 * (1.0 - double(p.s_visc));
            else
                error('DSMC params need ''zeta'' or ''s_visc''.');
            end

            if ~isfield(p, 'delta'), error('DSMC params need ''delta''.'); end
            obj.delta = double(p.delta);
            obj.nu    = obj.delta / 2.0 - 1.0;

            if isfield(p, 'omega'), obj.omega = double(p.omega); else, obj.omega = 1.0; end
            if isfield(p, 'mass'),  obj.mass  = double(p.mass);  else, obj.mass  = 1.0; end

            % Monatomic-consistency constant K_delta (eq 40)
            obj.K_delta = 2.0 * gamma(obj.delta + 1.5) / (sqrt(pi) * gamma(obj.delta / 2.0)^2);

            % VHS rate constants (overall scale + frozen/non-frozen split).
            % Default both to K_delta so the convex split is a pure omega blend.
            if isfield(p, 'C_vhs'), obj.C_vhs = double(p.C_vhs); else, obj.C_vhs = obj.K_delta; end
            if isfield(p, 'C_vhs_frozen')
                obj.C_vhs_frozen = double(p.C_vhs_frozen);
            else
                obj.C_vhs_frozen = obj.K_delta;
            end

            % Extended-model internal-energy modulation (eq 43). Default 0 ->
            % the base DSMC (eq 54) kernel. eta_hat, eta_hat_f >= -1/2 keep B >= 0.
            if isfield(p, 'eta_hat'),    obj.eta_hat    = double(p.eta_hat);    else, obj.eta_hat    = 0.0; end
            if isfield(p, 'zeta_hat'),   obj.zeta_hat   = double(p.zeta_hat);   else, obj.zeta_hat   = 0.0; end
            if isfield(p, 'eta_hat_f'),  obj.eta_hat_f  = double(p.eta_hat_f);  else, obj.eta_hat_f  = 0.0; end
            if isfield(p, 'zeta_hat_f'), obj.zeta_hat_f = double(p.zeta_hat_f); else, obj.zeta_hat_f = 0.0; end
            if obj.eta_hat < -0.5 || obj.eta_hat_f < -0.5
                error('DSMC extended model requires eta_hat, eta_hat_f >= -1/2 (kernel positivity).');
            end
            if obj.zeta_hat < 0 || obj.zeta_hat_f < 0
                error('DSMC extended model requires zeta_hat, zeta_hat_f >= 0.');
            end

            % alpha mirrors zeta so the existing alpha-keyed energy quadrature
            % (Laguerre(alpha/2) + desingularization /z^(alpha/2)) is reused.
            obj.alpha = obj.zeta;

            % Scalar evaluator (translational part only; for diagnostics/tests).
            obj.exact_kernel = @(u_mag, I, J) (u_mag .* sqrt(2.0)).^obj.zeta;
        end
    end
end 