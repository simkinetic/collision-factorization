classdef TestMonatomicModulationInertness < matlab.unittest.TestCase
    % TESTMONATOMICMODULATIONINERTNESS - The I_max = 0 Dirac collapse must
    % reproduce the monatomic operator EXACTLY, for every kernel branch and on
    % every build path.
    %
    % At the collapse the internal measure concentrates on I = J = 0, so the
    % extended-model (Djordjic eq. 43) modulation factors vanish identically:
    %   non-frozen:  (r(1-R) i)^{zeta_hat/2} = 0   (i = I/E = 0)
    %   frozen:      i^{zeta_hat_f} + i*^{zeta_hat_f} = 0
    % and the frozen e_tr^{zeta/2} weight is identically 1 (E = E_tr). A
    % modulated kernel at I_max = 0 must therefore build the SAME tensor as the
    % base kernel, on the pointwise path AND on the auxiliary-Laplace path.
    %
    % Regression context: the Laplace path's correction terms
    % (laplace_correction) construct their own finite-nu multi-node Laguerre
    % rules, ignoring the collapsed single-node internal grids. Without the
    % I_max > 0 guard at the correction call sites this added a spurious O(1)
    % correction at I_max = 0 (measured rel. L_inf ~0.9). The frozen-spectral
    % branch always had the guard (use_frozen_lap); the fallback did not.

    properties (Constant, Access = private)
        K_max = 2;
        L_max = 2;
        pad_r = 4; pad_t = 4; pad_i = 2;   % cheap; I_max = 0 collapses internals
        zeta  = 0.534;
        delta = 2.01;
        Ns    = 8;
    end

    properties (TestParameter)
        % channel: which convex-split channel carries the modulation
        channel = {'frozen', 'nonfrozen'};
        % path: pointwise (sumfac MEX) vs auxiliary-Laplace (aux MEX, default)
        path = {'pointwise', 'laplace'};
    end

    properties (Access = private)
        base = struct();   % per-channel base-kernel reference (pointwise path)
    end

    methods (TestClassSetup)
        function buildBaseReferences(testCase)
            testCase.base.frozen    = testCase.build(0, 0, 0, 0, 0, false);
            testCase.base.nonfrozen = testCase.build(1, 0, 0, 0, 0, false);
        end
    end

    methods (Test)

        function testModulationIsInert(testCase, channel, path)
            % A modulated kernel at I_max = 0 must equal the base kernel build.
            switch channel
                case 'frozen'
                    omega = 0; eh = 0; zh = 0; ehf = 1; zhf = 0.3;
                case 'nonfrozen'
                    omega = 1; eh = 1; zh = 0.965; ehf = 0; zhf = 0;
            end
            R_mod  = testCase.build(omega, eh, zh, ehf, zhf, strcmp(path, 'laplace'));
            R_base = testCase.base.(channel);
            err = max(abs(R_mod(:) - R_base(:))) / max(abs(R_base(:)));
            testCase.verifyLessThan(err, 1e-13, sprintf( ...
                ['I_max = 0 modulated build (%s channel, %s path) deviates from ' ...
                 'the base kernel by rel. L_inf %.3e; the modulation must be inert.'], ...
                channel, path, err));
        end

        function testPathsAgreeOnBaseKernel(testCase, channel)
            % At I_max = 0 the Laplace path reduces to the same base build as the
            % pointwise path (no admissible auxiliary integral): bitwise-level match.
            omega = double(strcmp(channel, 'nonfrozen'));
            R_lap = testCase.build(omega, 0, 0, 0, 0, true);
            R_pw  = testCase.base.(channel);
            err = max(abs(R_lap(:) - R_pw(:))) / max(abs(R_pw(:)));
            testCase.verifyLessThan(err, 1e-13, sprintf( ...
                'I_max = 0 base build: laplace vs pointwise path differ by %.3e.', err));
        end

    end

    methods (Access = private)
        function R = build(testCase, omega, eh, zh, ehf, zhf, laplace)
            kp = struct('zeta', testCase.zeta, 'delta', testCase.delta, ...
                        'omega', omega, 'eta_hat', eh, 'zeta_hat', zh, ...
                        'eta_hat_f', ehf, 'zeta_hat_f', zhf);
            Kernel = ScatteringKernel('DSMC', kp);
            Basis  = SpectralBasis(testCase.K_max, testCase.L_max, 0, Kernel.nu);
            T = GeneralCollisionTensor(Basis, Kernel);
            T.conserve_invariants = false;
            T.laplace_extended = laplace;
            T.laplace_Ns = testCase.Ns;
            T.generate_R_tensor_sumfac(testCase.pad_r, testCase.pad_t, testCase.pad_i);
            R = T.R_tensor;
        end
    end
end
