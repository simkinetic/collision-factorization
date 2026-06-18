classdef TestQuadraturePolynomialReproduction < matlab.unittest.TestCase
    
    properties (TestParameter)
        % 1. Degrees to test (0, 1, 2, 3, 4)
        m_degree = num2cell(0:4);
        
        % 2. Quadrature types to test
        quad_type = {'legendre', 'lobatto'};
        
        % 3. Number of nodes for the Jacobi test
        N_nodes = {5, 50};
    end
    
    methods (Test)
        
        function testPolyReproduction(testCase, m_degree, quad_type)
            
            % domain
            a = 1.5; 
            b = 4.5;
            p = 4; % Basis Degree (Quartic)
            N = p + 1; % Number of interior quadrature points
            
            % We only want to add the 0-weight boundary nodes if using Legendre
            include_boundaries = strcmpi(quad_type, 'legendre');
            
            % Instantiate the new stateful quadrature object
            switch lower(quad_type)
                case 'lobatto'
                    qr = Gauss.lobatto(N, a, b);
                case 'legendre'
                    qr = Gauss.legendre(N, a, b, include_boundaries);
            end
            x = qr.x; w = qr.w;
            
            % Define target function f(x) = (m+1) * x^m
            f_exact = @(x) (m_degree+1) * x.^m_degree;
            
            % exact integral: \int_a^b (m+1) x^m dx = b^(m+1) - a^(m+1)
            I_exact = b^(m_degree+1) - a^(m_degree+1);
            
            % approximated integral
            I_quad = dot(w, f_exact(x));
            
            % Check error (L_inf norm)
            err = abs(I_exact - I_quad);
            
            % Assertion
            testCase.verifyLessThan(err, 1e-12, ...
                sprintf('Failed x^%d with p=%d using %s nodes', ...
                        m_degree, p, quad_type));
        end
        
        function testHalfRangeHermiteReproduction(testCase, m_degree)
            % TESTHALFRANGEHERMITEREPRODUCTION
            % Verifies exact integration of f(x) = x^(2m) against the 
            % physical Maxwellian weight x^2 e^(-beta x^2) on [0, inf).
            
            beta_scale = 1.25; % Use a non-trivial scaling parameter
            N = 5;       % 5 points can exactly integrate up to m=9
            
            % Generate half-range Hermite quadrature rule
            qr = Gauss.halfrange_hermite(N, beta_scale);
            x = qr.x; w = qr.w;
            
            % Define target function f(x) = x^(2m)
            f_eval = x.^(2 * m_degree);
            
            % Exact analytical integral of x^(2m + 2) * e^(-beta * x^2)
            exact_val = gamma(m_degree + 1.5) / (2 * beta_scale^(m_degree + 1.5));
            
            % Approximated integral
            I_quad = dot(w, f_eval);
            
            % Check error
            err = abs(exact_val - I_quad);
            
            % Assertion
            testCase.verifyLessThan(err, 1e-12, ...
                sprintf('Failed half-range Hermite x^(%d) with beta=%.2f', ...
                        2*m_degree, beta_scale));
        end
        
        function testJacobiReproductionAndStability(testCase, m_degree, N_nodes)
            % TESTJACOBIREPRODUCTIONANDSTABILITY
            % Verifies exact integration of f(x) = x^m against the Jacobi
            % weight (1-x)^alpha * (1+x)^beta on [-1, 1], and ensures the
            % Golub-Welsch eigenvalue solver remains stable and highly accurate
            % (error < 1e-12) even at high node counts.
            
            alpha_val = 1.5;
            beta_val  = 0.5;
            a = -1; b = 1;
            
            qr = Gauss.jacobi(N_nodes, alpha_val, beta_val, a, b);
            x = qr.x; w = qr.w;
            
            % 1. Verify structural integrity of the arrays (NaN/Inf checks)
            testCase.verifyFalse(any(isnan(x)), sprintf('NaN found in nodes at N=%d', N_nodes));
            testCase.verifyFalse(any(isnan(w)), sprintf('NaN found in weights at N=%d', N_nodes));
            testCase.verifyFalse(any(isinf(w)), sprintf('Inf found in weights at N=%d', N_nodes));
            
            % 2. Exact analytical integral using binomial expansion and the Beta function:
            % int_{-1}^1 x^m (1-x)^a (1+x)^b dx
            exact_val = 0;
            for k = 0:m_degree
                % built-in beta(z, w) computes the Beta function
                beta_term = beta(beta_val + k + 1, alpha_val + 1);
                term = nchoosek(m_degree, k) * 2^k * (-1)^(m_degree - k) * beta_term;
                exact_val = exact_val + term;
            end
            exact_val = exact_val * 2^(alpha_val + beta_val + 1);
            
            % 3. Approximated integral
            f_eval = x.^m_degree;
            I_quad = dot(w, f_eval);
            
            err = abs(exact_val - I_quad);
            
            % 4. Assert strict 1e-12 tolerance
            testCase.verifyLessThan(err, 1e-12, ...
                sprintf('Failed Jacobi reproduction for x^%d with alpha=%.1f, beta=%.1f at N=%d (err: %e)', ...
                        m_degree, alpha_val, beta_val, N_nodes, err));
        end
        
    end
end