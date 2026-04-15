classdef TestQuadraturePolynomialReproduction < matlab.unittest.TestCase
    
    properties (TestParameter)
        % 1. Degrees to test (0, 1, 2, 3, 4)
        m_degree = num2cell(0:4);
        
        % 2. Quadrature types to test
        quad_type = {'legendre', 'lobatto'};
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
            
            beta = 1.25; % Use a non-trivial scaling parameter
            N = 5;       % 5 points can exactly integrate up to m=9
            
            % Generate half-range Hermite quadrature rule
            qr = Gauss.halfrange_hermite(N, beta);
            x = qr.x; w = qr.w;
            
            % Define target function f(x) = x^(2m)
            f_eval = x.^(2 * m_degree);
            
            % Exact analytical integral of x^(2m + 2) * e^(-beta * x^2)
            exact_val = gamma(m_degree + 1.5) / (2 * beta^(m_degree + 1.5));
            
            % Approximated integral
            I_quad = dot(w, f_eval);
            
            % Check error
            err = abs(exact_val - I_quad);
            
            % Assertion
            % We use 1e-12 as Hermite rules can occasionally exhibit slightly
            % more floating point drift than compact Legendre rules.
            testCase.verifyLessThan(err, 1e-12, ...
                sprintf('Failed half-range Hermite x^(%d) with beta=%.2f', ...
                        2*m_degree, beta));
        end
        
    end
end