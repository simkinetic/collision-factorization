classdef TestPolyatomicConservation < matlab.unittest.TestCase
    % TESTPOLYATOMICCONSERVATION - Unit test for the exact enforcement of the five
    % collision invariants (mass, 3x momentum, energy) on the polyatomic operator.
    %
    % Mass and momentum each occupy a single basis row and are zeroed outright. The
    % total energy |v|^2 + I spans the row pair {(k=1,i=0,l=0), (k=0,i=1,l=0)} with
    % weights (a,b) = (sqrt(3/2), sqrt(nu+1)) -- see GeneralCollisionTensor's
    % enforce_conservation -- and is removed by orthogonal projection. The orthogonal
    % complement of that pair carries the physical translational<->internal energy
    % exchange (Landau-Teller) mode and must survive untouched.
    %
    % Operators are built once in TestClassSetup and shared: each build is the
    % expensive 9D quadrature, so rebuilding per test method would dominate the suite.

    properties (Constant, Access = private)
        K_max = 2;
        L_max = 2;
        I_max = 1;
        pad   = 6;              % Maxwell molecules converge spectrally; 6 is ample
        D_I_LIST = [2 3 5];     % internal DOF cases; mirrors the d_i TestParameter
    end

    properties (TestParameter)
        d_i = {2, 3, 5};
    end

    properties (Access = private)
        cache = struct();       % per-d_i: enforced/raw Jacobians, Basis, tensor
    end

    methods (TestClassSetup)
        function buildOperators(testCase)
            for d = TestPolyatomicConservation.D_I_LIST
                nu = d/2 - 1;
                T_on  = testCase.build_tensor(nu, true);
                T_off = testCase.build_tensor(nu, false);
                testCase.cache.(sprintf('d%d', d)) = struct( ...
                    'nu',    nu, ...
                    'Basis', T_on.Basis, ...
                    'T_on',  T_on, ...
                    'J_on',  testCase.jacobian(T_on), ...
                    'J_off', testCase.jacobian(T_off));
            end
        end
    end

    methods (Test)

        function testInvariantRowsAreExactlyNull(testCase, d_i)
            c = testCase.get(d_i);
            J = c.J_on;
            idx = testCase.indexer(c.Basis);

            % Mass: row (k=0, i=0, l=0)
            testCase.verifyEqual(max(abs(J(idx(0,0,1), :))), 0, ...
                'Mass row is not identically zero.');

            % Momentum: rows (k=0, i=0, l=1), i.e. q = 2,3,4
            for q = 2:4
                testCase.verifyEqual(max(abs(J(idx(0,0,q), :))), 0, ...
                    sprintf('Momentum row q=%d is not identically zero.', q));
            end

            % Energy: the (a,b)-weighted combination of the two energy rows.
            [a, b] = testCase.energy_weights(c.nu);
            r = a*J(idx(1,0,1), :) + b*J(idx(0,1,1), :);
            testCase.verifyLessThan(max(abs(r)) / max(abs(J(:))), 1e-12, ...
                'The energy row combination a*row_(1,0) + b*row_(0,1) is not null.');
        end

        function testNullSpaceIsFiveDimensional(testCase, d_i)
            c = testCase.get(d_i);
            lam = sort(abs(eig(c.J_on))) / max(abs(c.J_on(:)));

            n_null = sum(lam < 1e-12);
            testCase.verifyEqual(n_null, 5, ...
                sprintf('Expected a 5-dimensional null space, found %d.', n_null));

            % The null block must be cleanly separated, or the count above is meaningless.
            testCase.verifyGreaterThan(lam(6), 1e-4, ...
                'The null space is not cleanly separated from the rest of the spectrum.');
        end

        function testEnergyDirectionMatchesTheBasis(testCase, d_i)
            % The measured left-null direction of the 2x2 energy block, in the raw
            % (unenforced) operator, must be the analytic (a,b). This pins the
            % normalization convention: the invariant is |v|^2 + I, NOT 1/2|v|^2 + I,
            % because SpectralBasis uses r2 = |v|^2 (weight e^{-|v|^2}).
            c = testCase.get(d_i);
            idx = testCase.indexer(c.Basis);
            blk = [idx(1,0,1), idx(0,1,1)];

            [W, D] = eig(c.J_off(blk, blk).');       % columns of W = left eigenvectors
            [~, j0] = min(abs(real(diag(D))));
            w = real(W(:, j0));  w = w / w(1);

            [a, b] = testCase.energy_weights(c.nu);
            testCase.verifyEqual(w(2), b/a, 'RelTol', 1e-8, ...
                'The measured energy direction disagrees with (sqrt(3/2), sqrt(nu+1)).');
        end

        function testExchangeModeIsPreserved(testCase, d_i)
            % Enforcement must remove only the component along the energy direction.
            % The orthogonal component -- the trans<->internal exchange rate that sets
            % the Landau-Teller relaxation -- must be unchanged.
            c = testCase.get(d_i);
            idx = testCase.indexer(c.Basis);
            blk = [idx(1,0,1), idx(0,1,1)];

            rate = @(J) min(real(eig(J(blk, blk))));  % the nonzero (negative) eigenvalue
            r_on  = rate(c.J_on);
            r_off = rate(c.J_off);

            testCase.verifyLessThan(r_on, 0, ...
                'The energy-exchange mode was destroyed by the projection.');
            testCase.verifyEqual(r_on, r_off, 'RelTol', 1e-9, ...
                'The energy-exchange rate changed under conservation enforcement.');
        end

        function testEnforcementIsIdempotent(testCase)
            % enforce_conservation is an orthogonal projection, so re-applying it is a
            % no-op up to roundoff. This guards the two call sites in
            % generate_R_tensor_sumfac against a double application changing the answer.
            c = testCase.get(3);
            T  = c.T_on;
            R1 = T.R_tensor;
            T.enforce_conservation();
            drift = max(abs(T.R_tensor(:) - R1(:))) / max(abs(R1(:)));
            testCase.verifyLessThan(drift, 1e-14, ...
                'enforce_conservation is not idempotent.');
        end

        function testDisabledFlagLeavesTheOperatorUntouched(testCase)
            % With conserve_invariants = false the invariant rows must still carry the
            % raw quadrature residual -- small, but not exactly zero. The quadrature
            % error benchmarks rely on this.
            c = testCase.get(3);
            testCase.verifyGreaterThan(max(abs(c.J_off(1, :))), 0, ...
                'conserve_invariants = false still zeroed the mass row.');
        end

    end

    methods (Access = private)

        function c = get(testCase, d)
            c = testCase.cache.(sprintf('d%d', d));
        end

        function T = build_tensor(testCase, nu, enforce)
            % Maxwell molecules (gamma = 0) so the spectrum is exact and cheap.
            Basis  = SpectralBasis(testCase.K_max, testCase.L_max, testCase.I_max, nu);
            Kernel = ScatteringKernel('Polyatomic', 0.0);
            T = GeneralCollisionTensor(Basis, Kernel);
            T.conserve_invariants = enforce;
            T.generate_R_tensor_sumfac(testCase.pad, testCase.pad, testCase.pad);
        end

        function J = jacobian(~, T)
            C = T.assemble_full_tensor();
            % Linearize about equilibrium (c_1 = 1): J_ij = C_ij1 + C_i1j
            J = squeeze(C(:, :, 1)) + squeeze(C(:, 1, :));
        end

        function idx = indexer(~, Basis)
            idx = @(k, i, q) (k*Basis.N_I + i)*Basis.N_Q + q;
        end

        function [a, b] = energy_weights(~, nu)
            a = sqrt(3/2);
            b = sqrt(nu + 1);
        end

    end
end
