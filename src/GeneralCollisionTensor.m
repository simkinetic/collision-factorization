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

        % Spectral extended-model (eq 43) path. When true and the kernel is DSMC with
        % nonzero eta_hat/eta_hat_f, the singular factor (I/E)^zhat (E=1/2|u|^2+I+J) is
        % resolved by the auxiliary Laplace s-integral E^{-zhat}=1/Gamma(zhat) int s^{zhat-1}
        % e^{-sE} ds (compute_rtensor_polyatomic_aux_mex), restoring spectral convergence.
        % false -> legacy pointwise (algebraic) extended build via the base MEX.
        % Default true: the auxiliary-Laplace s-integral gives spectral accuracy
        % for energy-coupled/extended kernels (eta_hat ~= 0), where the pointwise
        % algebraic (I/E)^zhat evaluation converges only algebraically. For pure
        % base kernels (eta_hat = eta_hat_f = 0) the correction term vanishes and
        % the two paths agree to machine precision. Set false to opt out for
        % comparison studies (the algebraic pointwise path).
        laplace_extended = true;
        laplace_Ns = 32;          % Gauss-Jacobi s-node count for the Laplace integral

        % Exact enforcement of the 5 collision invariants (mass, 3x momentum, energy) on
        % the test rows of R_tensor. Mass and momentum each occupy a single row and are
        % zeroed outright; the total energy |v|^2 + I spans the (k=1,i=0,l=0)/(k=0,i=1,l=0)
        % pair and is removed by orthogonal projection, which leaves the physical
        % translational<->internal exchange mode untouched. See enforce_conservation.
        % Set false to inspect the raw quadrature residual (benchmark_*quadrature_error).
        conserve_invariants = true;

        % Hard structural constraint (Rene, 2026-08-17): on the SPECTRAL /
        % production path (laplace_extended && DSMC) the four internal-sector
        % directions (I, J, r, R) are EXACT by polynomial degree at the
        % Table-1 node counts: base branches against the native weights,
        % modulated branches via the shifted-Jacobi prefactor rules, frozen
        % branches via the frozen-spectral auxiliary treatment. Requested
        % internal padding is dead weight there and is clamped to zero (one
        % warning per session; the effective values are recorded in
        % effective_padding). The algebraic/pointwise diagnostic paths keep
        % the padding knob untouched -- there internal padding IS the
        % convergence mechanism. N_lambda (laplace_Ns) is a separate,
        % geometric knob and is never clamped.
        % Default true since 2026-08-18: the overnight confirmation sweeps
        % measured int_pad 0 == int_pad 8 at <= 2e-14 on all four spectral
        % branches (nf base, nf modulated post-shift, frozen base, frozen
        % modulated), and the full int_pad sweeps are flat at the roundoff
        % floor from the minimum Table-1 counts. Set false only to reproduce
        % pre-constraint builds in diagnostics.
        clamp_internal_pad = true;

        % Effective (post-clamp) paddings of the last generate_R_tensor_sumfac
        % call: struct with fields radial, angular, internal_requested,
        % internal_effective. Cache writers and logs must record
        % internal_effective, never the request, so filenames and contents
        % cannot diverge.
        effective_padding = struct();

        % DIAGNOSTIC OPT-OUT -- leave false for all production work.
        % Production (false) builds the DSMC frozen channel of Djordjic's
        % EXTENDED model, eq. (43), whose frozen term carries the weight
        % (e_tr)^{zeta/2}. The deltas fix e_tr = R' = R = (1/2)|u|^2/E, so
        % eq. (43) is uniformly |u|^zeta R^{zeta/2}[...] on both channels: the
        % non-frozen one folds R^{zeta/2} into the Jacobi R-weight, the frozen
        % one evaluates it at the collapsed point. Eq. (43) is the model whose
        % parameters Djordjic's Sect. 9 (Table 4) transport fits use.
        % Set true to recover the eq. (53) DSMC-comparison form of Sect. 8,
        % whose frozen term is |u|^zeta C^f_VHS delta delta with no e_tr weight.
        % Dropping the weight over-weights the frozen channel by 1/<R^{zeta/2}>
        % (~1.3 at these zeta), which biases the bulk-to-shear ratio high; the
        % opt-out exists only so that comparison stays reproducible.
        % Exactly a no-op at zeta = 0 and at I_max = 0 (there E = E_tr, e_tr = 1).
        frozen_no_etr = false;

        % DIAGNOSTIC BUILD MODE -- leave false for all production work.
        % When true, the gain (post-collision redistribution) kernels are zeroed
        % before the R-tensor MEX contractions, so the build returns the LOSS-ONLY
        % operator -Q^-(f, f) instead of the full Q = Q^+ - Q^-. This exists so
        % that the equilibrium collision frequency nu_0 can be measured
        % INDEPENDENTLY of the shear eigenvalue, rather than being defined from it
        % (as the normalization in benchmark_polyatomic_wcu_limit.m and
        % benchmark_polyatomic_temperature_relaxation.m does, which is fine for a
        % normalizer but circular as a verification of lambda_shear = -nu_0/2).
        % At zeta = 0 the loss term is exactly nu_0 f, so the linearized loss-only
        % operator is -nu_0 I on every non-density mode and nu_0 can be read off
        % its diagonal. See benchmark_polyatomic_shear_rate.m (paper Sect. 5.2).
        %
        % The flag is consumed at exactly one point (just before the MEX calls in
        % generate_R_tensor_sumfac), so it covers BOTH the spectral/auxiliary
        % Laplace path and the algebraic base path. It has no effect at all when
        % false: the default build path is byte-identical to before this flag existed.
        %
        % NOTE: the loss operator does not conserve momentum or energy on its own,
        % so set conserve_invariants = false alongside it -- otherwise the
        % invariant projection will zero/rotate exactly the rows being measured.
        loss_only = false;
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

            % DSMC convex-split kernel (Djordjic et al. 2023): the non-frozen
            % channel is B^nf = K_delta*|u|^zeta*R^(zeta/2). The R^(zeta/2) folds
            % into the kinetic-partition Jacobi weight; the |u|^zeta is handled by
            % the existing alpha-keyed energy quadrature (alpha = zeta). The
            % additive internal kernel term of the legacy 'Polyatomic' model is
            % dropped via kernel_model = 1.
            is_dsmc = strcmpi(obj.Kernel.model_type, 'DSMC');
            if is_dsmc
                kernel_model = 1;
                R_beta = 0.5 + obj.Kernel.zeta / 2.0;   % R^(1/2) * R^(zeta/2)
                % Extended-model (eq 43) internal-energy modulation params.
                eta_hat    = obj.Kernel.eta_hat;
                zeta_hat   = obj.Kernel.zeta_hat;
                eta_hat_f  = obj.Kernel.eta_hat_f;
                zeta_hat_f = obj.Kernel.zeta_hat_f;
            else
                kernel_model = 0;
                R_beta = 0.5;
                eta_hat = 0.0; zeta_hat = 0.0; eta_hat_f = 0.0; zeta_hat_f = 0.0;
            end
            
            % Internal-sector padding clamp (see the clamp_internal_pad
            % property). On the spectral path -- exactly the builds that the
            % `if obj.laplace_extended && is_dsmc` branch below routes through
            % the auxiliary MEX -- every internal-sector integrand (I, J, r, R)
            % is polynomial against its constructed weight, so the Table-1
            % node counts are EXACT and padding is dead weight: clamp it.
            % Spatial/energy paddings are untouched (still required), as is
            % N_lambda (geometric, separate knob), as is the algebraic path
            % (padding is its convergence mechanism).
            persistent clamp_warned
            internal_pad_req = internal_pad;
            if obj.clamp_internal_pad && obj.laplace_extended && is_dsmc && internal_pad > 0
                if isempty(clamp_warned)
                    warning('GeneralCollisionTensor:InternalPadClamped', ...
                        ['Spectral path: the internal-sector axes (I, J, r, R) ' ...
                         'are exact at the Table-1 node counts; the requested ' ...
                         'internal padding %d is clamped to 0 (dead weight). ' ...
                         'Printed once per session; the effective values are ' ...
                         'in the effective_padding property.'], internal_pad);
                    clamp_warned = true;
                end
                internal_pad = 0;
            end
            obj.effective_padding = struct( ...
                'radial', radial_pad, 'angular', angular_pad, ...
                'internal_requested', internal_pad_req, ...
                'internal_effective', internal_pad);

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
            if obj.I_max == 0
                % ---- Monatomic (Dirac) collapse --------------------------------
                % With no internal resolution the intended semantics is the
                % monatomic limit d_i -> 0, where the internal equilibrium
                % I^nu e^{-I}/Gamma(nu+1) concentrates onto delta(I). Single-node
                % rules reproduce that Dirac distribution exactly:
                %   I = J = 0  with weight Gamma(nu+1) (the full measure mass, so
                %              the i=0 basis norm 1/sqrt(Gamma(nu+1)) still gives
                %              exact orthonormality);
                %   R = 1      (all collision energy stays kinetic: |u'| = |u|,
                %              the elastic monatomic collision);
                %   r = 1/2    (irrelevant: I' = r(1-R)E = 0 at R = 1);
                % with unit partition masses W_r = W_R = 1, so the (r,R) integral
                % is the identity in both the gain and the loss term (the MEX loss
                % kernel is sum_WR*sum_Wr). The polyatomic build then reproduces
                % the monatomic operator exactly. Finite-nu Gauss-Laguerre nodes
                % here would instead put quadrature mass at I,J > 0 and let the
                % partition redistribute kinetic <-> internal energy: the WCU
                % spectrum regression of the strict I_max = 0 truncation that
                % this branch removes.
                % The collapsed weights carry the TOTAL MASS of each measure in
                % the code's own normalization (a 1-node Gauss rule integrates
                % constants exactly, so its weight IS that mass): Gamma(nu+1)
                % for the internal Laguerre axes and the (r,R) Jacobi masses for
                % the partition axes. This is the honest Dirac collapse
                % int f dmu -> f(x*) mu_total, and keeps rates continuous with
                % the I_max > 0 build (for DSMC, C_vhs = K_delta pairs with the
                % partition mass: K_delta * mass(H_delta) = 1 at zeta = 0, so
                % the non-frozen channel reduces to the monatomic VHS rate).
                qm_r = Gauss.jacobi(1, nu_val, nu_val, 0, 1);
                qm_R = Gauss.jacobi(1, 2*nu_val + 1, R_beta, 0, 1);
                I_nodes = 0;   W_I = gamma(nu_val + 1);
                J_nodes = 0;   W_J = gamma(nu_val + 1);
                r_nodes = 0.5; W_r = sum(qm_r.w);
                R_nodes = 1.0; W_R = sum(qm_R.w);
            else
                qr_I = Gauss.generalized_laguerre(N_I_nodes, nu_val);
                I_nodes = qr_I.x; W_I = qr_I.w;

                qr_J = Gauss.generalized_laguerre(N_J_nodes, nu_val);
                J_nodes = qr_J.x; W_J = qr_J.w;

                qr_r = Gauss.jacobi(N_r_nodes, nu_val, nu_val, 0, 1);
                r_nodes = qr_r.x; W_r = qr_r.w;

                qr_R = Gauss.jacobi(N_R_nodes, 2*nu_val + 1, R_beta, 0, 1);
                R_nodes = qr_R.x; W_R = qr_R.w;
            end
            
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

            % Loss-only diagnostic build (see the loss_only property). Zeroing the
            % gain kernels here -- after the Gaunt/SCALE assembly and before every
            % MEX call site below -- removes Q^+ from both the spectral (auxiliary
            % Laplace) and the algebraic base paths with one guard. Inert by default.
            if obj.loss_only
                P_gain_p1(:) = 0;
                P_gain_p2(:) = 0;
                fprintf('  [loss_only] gain kernels zeroed -> LOSS-ONLY operator.\n');
            end

            % ---- Spectral extended-model build via the auxiliary Laplace s-integral ----
            % Resolves the singular factor (I/E)^zhat (E = 1/2|u|^2 + I + J) by
            %   E^{-zhat} = 1/Gamma(zhat) int_0^inf s^{zhat-1} e^{-sE} ds .
            % Per s-node: the velocity part e^{-s|u|^2/2} is carried by the aux MEX
            % (s_laplace); the internal Laguerre nodes are shifted (nu->nu+zhat, holding
            % I^zhat) and scaled (sigma = 1/(1+s)); the (r(1-R))^zhat post weighting
            % (non-frozen) folds into the r/R weights. See laplace_correction.
            if obj.laplace_extended && is_dsmc
                mk_aux = @(xn, Wxn, fspec) @(km, In, WIn, Jn, WJn, rn, Wrn, Rn, WRn, s_lap) ...
                    compute_rtensor_polyatomic_aux_mex( ...
                        obj.K_max, obj.I_max, N_L, N_Q, alpha_kernel, nu_val, ...
                        xn, Wxn, u1_nodes, W_u1, t1_nodes, W_t1, ...
                        y2_nodes, W_y2, t2_nodes, W_t2, mu_chi, W_chi, eps_vec, W_eps, ...
                        In, WIn, Jn, WJn, rn, Wrn, Rn, WRn, ...
                        RadialNorm, InternalNorm, SH_Norm, L_triplets, qi_valid_mat, ...
                        P_loss_p1, P_gain_p1, P_loss_p2, P_gain_p2, ...
                        K_test_val, I_test_val, km, 0.0, 0.0, 0.0, 0.0, s_lap, ...
                        double(obj.frozen_no_etr), double(fspec));
                aux_mex = mk_aux(x_nodes, W_x, 0);

                % Spectral frozen channel (see frozen_laplace_correction). It needs
                % the energy axis rekeyed to weight x^zeta e^{-x}, because carrying
                % E_tr^{zeta/2} in the velocity power takes the frozen rate to
                % |u|^{2 zeta} and the desingularization to /x^zeta.
                % Inert cases where NO auxiliary integral is admissible or needed:
                %   alpha == 0     -> e_tr^0 == 1 identically (and Gamma(0) diverges);
                %   I_max == 0     -> the Dirac collapse gives E = E_tr, so e_tr == 1.
                %     The collapse is not a Gauss rule in I, so the sigma^{2nu+2}
                %     bookkeeping the Jacobi weight encodes does not apply there;
                %     the guard is required for CORRECTNESS, not just for speed.
                use_frozen_lap = ~obj.frozen_no_etr && alpha_kernel > 0 && obj.I_max > 0;
                if use_frozen_lap
                    qx_fr = Gauss.generalized_laguerre(N_x, alpha_kernel);
                    aux_fr = mk_aux(qx_fr.x, qx_fr.w, 1);
                end

                w = obj.Kernel.omega;
                R_nf = []; R_fr = [];
                % Monatomic (I_max == 0) Dirac collapse: the modulation factors
                % (r(1-R) i)^{zhat/2} and i^{zhat_f} vanish identically (i = I/E = 0
                % at the collapsed node I = 0), so the extended-model corrections are
                % exactly zero. laplace_correction must NOT run there: it builds its
                % own finite-nu multi-node Laguerre rules (ignoring the collapsed
                % single-node grids), which would put quadrature mass at I, J > 0 and
                % add a spurious nonzero correction. Same rationale as the I_max > 0
                % requirement in use_frozen_lap above; the pointwise MEX paths get
                % this right automatically (pow(0/E, zhat) = 0).
                use_corr = obj.I_max > 0;
                if w > 0
                    R_nf = aux_mex(1, I_nodes, W_I, J_nodes, W_J, r_nodes, W_r, R_nodes, W_R, 0.0);
                    if eta_hat ~= 0.0 && use_corr
                        R_nf = R_nf + eta_hat * obj.laplace_correction(aux_mex, 1, ...
                            zeta_hat/2.0, nu_val, N_I_nodes, N_J_nodes, r_nodes, W_r, R_nodes, W_R);
                    end
                end
                if w < 1
                    if use_frozen_lap
                        % Base frozen term: e_tr^{zeta/2} alone -> auxiliary exponent zeta/2.
                        R_fr = obj.frozen_laplace_correction(aux_fr, true, 0.0, alpha_kernel, ...
                            nu_val, N_I_nodes, N_J_nodes, r_nodes, W_r, R_nodes, W_R);
                        if eta_hat_f ~= 0.0
                            % Modulated term: e_tr^{zeta/2} (I/E)^{zhat_f} shares ONE
                            % fractional power of E, exponent zeta/2 + zhat_f.
                            R_fr = R_fr + eta_hat_f * obj.frozen_laplace_correction( ...
                                aux_fr, false, zeta_hat_f, alpha_kernel, ...
                                nu_val, N_I_nodes, N_J_nodes, r_nodes, W_r, R_nodes, W_R);
                        end
                    else
                        R_fr = aux_mex(2, I_nodes, W_I, J_nodes, W_J, r_nodes, W_r, R_nodes, W_R, 0.0);
                        if eta_hat_f ~= 0.0 && use_corr   % see the Dirac-collapse note above
                            R_fr = R_fr + eta_hat_f * obj.laplace_correction(aux_mex, 2, ...
                                zeta_hat_f, nu_val, N_I_nodes, N_J_nodes, r_nodes, W_r, R_nodes, W_R);
                        end
                    end
                end
                if w == 0
                    obj.R_tensor = obj.Kernel.C_vhs_frozen * R_fr;
                elseif w == 1
                    obj.R_tensor = obj.Kernel.C_vhs * R_nf;
                else
                    obj.R_tensor = w * obj.Kernel.C_vhs * R_nf + ...
                                   (1 - w) * obj.Kernel.C_vhs_frozen * R_fr;
                end
                obj.enforce_conservation();
                return;
            end

            % Call the Polyatomic MEX. The trailing argument selects the kernel:
            %   0 = legacy polyatomic additive Grad kernel
            %   1 = DSMC non-frozen channel  B^nf = (sqrt2 u)^zeta, R^(zeta/2) in W_R
            %   2 = DSMC frozen (elastic) channel B^f = (sqrt2 u)^zeta e_tr^{zeta/2},
            %       I'=I, J'=J  (eq. 43; e_tr = (1/2)|u|^2/E dropped iff frozen_no_etr)
            % The four trailing args carry the extended-model (eq 43) modulation:
            % non-frozen uses (eta_hat, zeta_hat); frozen uses (eta_hat_f, zeta_hat_f).
            call_mex = @(km) compute_rtensor_polyatomic_sumfac_mex( ...
                obj.K_max, obj.I_max, N_L, N_Q, alpha_kernel, nu_val, ...
                x_nodes, W_x, u1_nodes, W_u1, t1_nodes, W_t1, ...
                y2_nodes, W_y2, t2_nodes, W_t2, mu_chi, W_chi, eps_vec, W_eps, ...
                I_nodes, W_I, J_nodes, W_J, r_nodes, W_r, R_nodes, W_R, ...
                RadialNorm, InternalNorm, SH_Norm, L_triplets, qi_valid_mat, ...
                P_loss_p1, P_gain_p1, P_loss_p2, P_gain_p2, ...
                K_test_val, I_test_val, km, ...
                eta_hat, zeta_hat, eta_hat_f, zeta_hat_f, ...
                double(obj.frozen_no_etr));

            if is_dsmc
                % Convex split: R_total = omega*C_vhs*R_nonfrozen + (1-omega)*C_vhs_frozen*R_frozen.
                % Skip the channel with zero weight at the omega endpoints (the non-frozen
                % channel is the expensive 9D one), which halves endpoint builds.
                w = obj.Kernel.omega;
                if w == 0
                    obj.R_tensor = obj.Kernel.C_vhs_frozen * call_mex(2);
                elseif w == 1
                    obj.R_tensor = obj.Kernel.C_vhs * call_mex(1);
                else
                    R_nf = call_mex(1);
                    R_fr = call_mex(2);
                    obj.R_tensor = w * obj.Kernel.C_vhs * R_nf + ...
                                   (1 - w) * obj.Kernel.C_vhs_frozen * R_fr;
                end
            else
                obj.R_tensor = call_mex(kernel_model);
            end

            obj.enforce_conservation();
        end

        function enforce_conservation(obj)
            % ENFORCE_CONSERVATION  Impose the 5 collision invariants exactly on R_tensor.
            % The test (row) index of R_tensor is (k1, i1); the row's angular degree is
            % l1 = L_triplets(t,1). Since assemble_full_tensor is linear in R_tensor with
            % this same row index, constraining rows of R is exactly equivalent to
            % constraining rows of the assembled operator C, at a fraction of the cost.
            %
            % Mass       -> row (k1=0, i1=0), l1 = 0
            % Momentum   -> row (k1=0, i1=0), l1 = 1
            % Energy     -> the basis is orthonormal w.r.t. e^{-|v|^2} I^nu e^{-I}, and
            %   L_1^(1/2)(v^2) = 3/2 - v^2   =>  v^2 = 3/2   - sqrt(3/2)*psi_(1,0,0)
            %   L_1^(nu)(I)    = nu+1 - I    =>  I   = (nu+1) - sqrt(nu+1)*psi_(0,1,0)
            %   (both normalized by psi_(0,0,0)), so with the weight e^{-|v|^2} -- in which
            %   |v|^2 IS the translational energy in units of kT -- the invariant
            %   |v|^2 + I has row weights (a, b) = (sqrt(3/2), sqrt(nu+1)) on the pair
            %   {(k1=1,i1=0), (k1=0,i1=1)} in the l1 = 0 channels. Conservation is the
            %   single constraint a*row_(1,0) + b*row_(0,1) = 0, imposed by projecting that
            %   2-vector of rows off the unit direction e = (a,b)/|(a,b)|. The orthogonal
            %   complement carries the physical trans<->internal energy exchange
            %   (Landau-Teller) mode and is left exactly as computed.
            if ~obj.conserve_invariants || isempty(obj.R_tensor), return; end

            l1 = obj.L_triplets(:, 1);

            % Mass (l1=0) and momentum (l1=1) share the single row (k1=0, i1=0).
            obj.R_tensor(1, :, :, 1, :, :, l1 == 0 | l1 == 1) = 0;

            % Energy lives in the isotropic channels only.
            if obj.K_test < 1, return; end          % no (k1=1) row in the test space
            sel_e = (l1 == 0);

            if obj.I_max == 0
                % Monatomic: no internal row exists, the invariant is |v|^2 alone.
                obj.R_tensor(2, :, :, 1, :, :, sel_e) = 0;
            elseif obj.I_test >= 1
                a = sqrt(3/2);   b = sqrt(obj.nu + 1);
                n = hypot(a, b); ea = a / n;  eb = b / n;

                E1 = obj.R_tensor(2, :, :, 1, :, :, sel_e);   % (k1=1, i1=0)
                E2 = obj.R_tensor(1, :, :, 2, :, :, sel_e);   % (k1=0, i1=1)
                P  = ea * E1 + eb * E2;                       % component along e
                obj.R_tensor(2, :, :, 1, :, :, sel_e) = E1 - ea * P;
                obj.R_tensor(1, :, :, 2, :, :, sel_e) = E2 - eb * P;
            else
                % I_max>=1 but I_test=0: the VMS test space cannot represent the internal
                % energy row, so the constraint is not expressible there. Zeroing the
                % translational row alone would destroy the physical trans->internal leak.
                warning('GeneralCollisionTensor:EnergyNotEnforced', ...
                    ['I_test=0 with I_max=%d: the energy invariant spans the internal ' ...
                     'row and cannot be enforced in this test space; skipping.'], obj.I_max);
            end
        end

        function Rc = frozen_laplace_correction(obj, aux_fr, is_base, zhat_f, alpha, nu_val, ...
                                                N_I_nodes, N_J_nodes, r_nodes, W_r, R_nodes, W_R)
            % FROZEN_LAPLACE_CORRECTION  Spectral frozen channel of Djordjic eq. (43).
            %
            % The frozen kernel is  B^f = |u|^zeta e_tr^{zeta/2} [1 + ehat_f(i^zf + i*^zf)],
            % e_tr = E_tr/E, E = E_tr + I + I*, E_tr = (1/2)|u|^2.  Both the base and
            % the modulated term are a single fractional power of E:
            %     base       : E_tr^{zeta/2}            E^{-zeta/2}          (zhat_f = 0)
            %     modulated  : E_tr^{zeta/2} I^{zhat_f} E^{-(zeta/2+zhat_f)}
            % so ONE auxiliary integral at exponent p = zeta/2 + zhat_f covers both
            % couplings at once -- they are not nested.
            %     E^{-p} = 1/Gamma(p) int_0^inf lam^{p-1} e^{-lam E} dlam,
            % and e^{-lam E} = e^{-lam E_tr} e^{-lam I} e^{-lam I*} factorizes per axis:
            %   * e^{-lam E_tr} is ENTIRE and applied pointwise on the unchanged
            %     velocity grid by the MEX (s_lap);
            %   * e^{-lam I}, e^{-lam I*} rescale the internal Gauss-Laguerre nodes by
            %     sigma = 1/(1+lam), each contributing sigma^{nu+1} (sigma^{nu+zhat_f+1}
            %     on the zhat_f-shifted axis, which also holds the I^{zhat_f});
            %   * E_tr^{zeta/2} folds into the velocity power (see aux MEX
            %     frozen_spectral), which is why aux_fr carries the rekeyed
            %     x^{zeta} e^{-x} energy rule.
            % Substituting xi = lam/(1+lam) on [0,1] (dlam = dxi/(1-xi)^2) collects
            %     (1-xi)^{-(p-1)} * (1-xi)^{2 nu + zhat_f + 2} * (1-xi)^{-2}
            % and the zhat_f cancels against p, leaving in BOTH cases the Gauss-Jacobi
            % weight
            %     (1 - xi)^{2 nu + 1 - zeta/2}  xi^{p - 1},
            % with total mass B(p, 2 nu + 2 - zeta/2).  Only the xi exponent depends on
            % zhat_f.  Integrability needs 2 nu + 2 > zeta/2, i.e. zeta < 2 delta.
            %
            % The internal axes are left with nothing but polynomials times their own
            % Laguerre weight, so the internal quadrature is EXACT again -- the point of
            % the construction.  Unlike the non-frozen channel there is no (r,R)
            % partition to reweight: the deltas collapsed it.
            Ns = obj.laplace_Ns;
            p  = alpha/2 + zhat_f;

            a_exp = 2*nu_val + 1 - alpha/2;         % (1-xi) exponent
            if a_exp <= -1
                error('GeneralCollisionTensor:FrozenLaplaceNotIntegrable', ...
                    ['Frozen auxiliary rule needs 2*nu+2 > zeta/2 (zeta < 2*delta); ' ...
                     'got nu=%g, zeta=%g.'], nu_val, alpha);
            end
            qj  = Gauss.jacobi(Ns, a_exp, p - 1, 0, 1);
            xik = qj.x;  wxi = qj.w;
            Bmom = exp(gammaln(p) + gammaln(2*nu_val + 2 - alpha/2) ...
                       - gammaln(p + 2*nu_val + 2 - alpha/2));
            wxi = wxi * (Bmom / sum(wxi));          % no-op safeguard, as non-frozen

            % Internal base rules.  Two INDEPENDENT questions, which must not be
            % conflated:
            %   * does the active axis need the shifted rule GL(nu+zhat_f)?  Only
            %     when zhat_f ~= 0 -- and GL(nu+0) = GL(nu), so simply always using
            %     the shifted rule is correct and degenerates cleanly.
            %   * how many channels are summed?  The BASE term (is_base) is a single
            %     integral; the MODULATED term is (i^zhat_f + i*^zhat_f), always TWO
            %     channels -- including at zhat_f = 0, where i^0 + i*^0 = 2 and the
            %     two identical channels must still both be counted.
            % Keying the channel count off zhat_f instead of is_base silently halved
            % the correction for the legal kernel (eta_hat_f ~= 0, zeta_hat_f = 0).
            qP = Gauss.generalized_laguerre(N_J_nodes, nu_val);            tP = qP.x;  wP = qP.w;
            qS = Gauss.generalized_laguerre(N_I_nodes, nu_val + zhat_f);   tS = qS.x;  wS = qS.w;

            Rc = 0;
            for k = 1:Ns
                xi = xik(k);  lam = xi / (1 - xi);  sig = 1 - xi;
                In_S = tS * sig;  In_P = tP * sig;   % scale nodes; weights unscaled
                if is_base
                    Rc = Rc + wxi(k) * ...
                        aux_fr(2, In_S, wS, In_P, wP, r_nodes, W_r, R_nodes, W_R, lam);
                else
                    Rc = Rc + wxi(k) * ( ...
                        aux_fr(2, In_S, wS, In_P, wP, r_nodes, W_r, R_nodes, W_R, lam) + ...
                        aux_fr(2, In_P, wP, In_S, wS, r_nodes, W_r, R_nodes, W_R, lam) );
                end
            end
            Rc = Rc / gamma(p);
        end

        function Rc = laplace_correction(obj, aux_mex, km, zhat, nu_val, ...
                                         N_I_nodes, N_J_nodes, r_nodes, W_r, R_nodes, W_R)
            % LAPLACE_CORRECTION  eta-coefficient correction tensor for the extended model.
            % Returns the (I/E)^zhat + (J/E)^zhat weighted operator (eq 43 bracket, sans the
            % eta prefactor) built spectrally via the auxiliary Laplace s-integral:
            %   J_corr = 1/Gamma(zhat) int_0^inf s^{zhat-1} [ aux operator with e^{-sE} ] ds .
            % Substituting t = s/(1+s) in [0,1] gives a Gauss-Jacobi(2nu+1, zhat-1) rule; the
            % internal Laguerre nodes are scaled by sigma = 1-t = 1/(1+s) (the e^{-sI},e^{-sJ}),
            % the I^zhat (resp. J^zhat) endpoint is held by a shifted GL(nu+zhat) rule, and the
            % velocity e^{-s|u|^2/2} is applied inside the MEX (s_lap). Internal weights are NOT
            % rescaled by sigma-powers -- those are absorbed by the (1-t)^{2nu+1} Jacobi weight.
            % For the non-frozen channel (km==1) the bounded partition prefactors
            % (r(1-R))^zhat / ((1-r)(1-R))^zhat fold into SHIFTED Gauss-Jacobi rules
            % per active channel (see below). I- and J-channels are both summed.
            Ns = obj.laplace_Ns;

            % s-grid: Gauss-Jacobi(2nu+1, zhat-1) on [0,1]. The renormalization to the
            % exact Beta moment B(zhat, 2nu+2) is now a no-op safeguard: Gauss.jacobi's
            % [0,1] mapping carries the properly rescaled weight function.
            qj = Gauss.jacobi(Ns, 2*nu_val + 1, zhat - 1, 0, 1);
            tk = qj.x; wtk = qj.w;
            Bmom = exp(gammaln(zhat) + gammaln(2*nu_val + 2) - gammaln(zhat + 2*nu_val + 2));
            wtk = wtk * (Bmom / sum(wtk));

            % Internal base rules: shifted GL(nu+zhat) holds the ^zhat power; GL(nu) is partner.
            qS = Gauss.generalized_laguerre(N_I_nodes, nu_val + zhat); tS = qS.x; wS = qS.w;
            qP = Gauss.generalized_laguerre(N_J_nodes, nu_val);        tP = qP.x; wP = qP.w;

            is_nf = (km == 1);
            if is_nf
                % Bounded partition prefactors, folded into SHIFTED Gauss-Jacobi
                % rules per active channel (spectral -- in fact exact, since the
                % remaining r/R integrand is polynomial: post-collision internal
                % basis at I' = r(1-R)E, J' = (1-r)(1-R)E; the loss integrand is
                % constant). The former pointwise application of r^zhat,
                % (1-r)^zhat, (1-R)^zhat on the UNSHIFTED rules was only
                % algebraically convergent (fractional endpoint powers,
                % zhat = zeta_hat/2 non-integer). Gauss.jacobi(N, a, b, 0, 1)
                % integrates f(t) (1-t)^a t^b dt on [0,1], so with the native
                % weights r^nu(1-r)^nu and R^{(1+zeta)/2}(1-R)^{2nu+1}:
                %   R axis, both channels : (1-R)^{2nu+1+zhat} R^{(1+zeta)/2}
                %       -> Gauss.jacobi(N_R, 2*nu+1+zhat, (1+zeta)/2)
                %   r axis, I-active chan.: (1-r)^{nu} r^{nu+zhat}
                %       -> Gauss.jacobi(N_r, nu, nu+zhat)
                %   r axis, J-active chan.: (1-r)^{nu+zhat} r^{nu}
                %       -> Gauss.jacobi(N_r, nu+zhat, nu)
                % The MEX computes its loss moments from the passed weights (their
                % sums are now the exact Beta masses), so gain and loss stay
                % consistent. The rules are s-independent: build once.
                N_r = numel(r_nodes);  N_R = numel(R_nodes);
                R_beta = 0.5 + obj.Kernel.zeta / 2.0;
                qR  = Gauss.jacobi(N_R, 2*nu_val + 1 + zhat, R_beta, 0, 1);
                qr1 = Gauss.jacobi(N_r, nu_val, nu_val + zhat, 0, 1);   % I-active
                qr2 = Gauss.jacobi(N_r, nu_val + zhat, nu_val, 0, 1);   % J-active
            end

            Rc = 0;
            for k = 1:Ns
                t = tk(k); s = t / (1 - t); sig = 1 - t;     % sigma = 1/(1+s)
                In_S = tS * sig;  In_P = tP * sig;           % scale nodes; weights unscaled
                if is_nf
                    Rc_I = aux_mex(km, In_S, wS, In_P, wP, qr1.x, qr1.w, qR.x, qR.w, s);
                    Rc_J = aux_mex(km, In_P, wP, In_S, wS, qr2.x, qr2.w, qR.x, qR.w, s);
                else
                    Rc_I = aux_mex(km, In_S, wS, In_P, wP, r_nodes, W_r, R_nodes, W_R, s);
                    Rc_J = aux_mex(km, In_P, wP, In_S, wS, r_nodes, W_r, R_nodes, W_R, s);
                end
                Rc = Rc + wtk(k) * (Rc_I + Rc_J);
            end
            Rc = Rc / gamma(zhat);
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