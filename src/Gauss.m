classdef Gauss
    % GAUSSQUADRATURE Static factory for 1D Gauss quadrature rules.
    % Generates points and weights for numerical integration.
    
    methods (Static)
    
        function qr = lobatto(N, a, b)
            % LOBATTO Returns Gauss-Lobatto-Legendre nodes, weights, and Vandermonde
            % N: Polynomial Order (Number of points = N + 1)

            % CONSTRUCTOR
            if nargin < 2 || isempty(a), a = -1; end
            if nargin < 3 || isempty(b), b = 1; end
            
            N1 = N; N = N - 1;
            x = cos(pi*(0:N)/N)'; % Initial guess (Chebyshev-Lobatto)
            
            P = zeros(N1, N1);
            xold = 2;
            
            while max(abs(x - xold)) > eps
                xold = x;
                P(:, 1) = 1;    
                P(:, 2) = x;
                for k = 2:N
                    P(:, k+1) = ((2*k-1)*x.*P(:, k) - (k-1)*P(:, k-1)) / k;
                end
                x = xold - (x.*P(:, N1) - P(:, N)) ./ (N1*P(:, N1));
            end
            
            
            qr.x = flipud((a*(1-x)+b*(1+x))/2);      
            
            % Compute the weights
            qr.w = (b-a) ./ (N*N1*P(:, N1).^2);
        end
        
        function qr = legendre(N, a, b, include_boundaries)

            if nargin < 2 || isempty(a), a = -1; end
            if nargin < 3 || isempty(b), b = 1; end
            if nargin < 4 || isempty(include_boundaries), include_boundaries = false; end
            
            % This script is for computing definite integrals using Legendre-Gauss 
            % Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
            % [a,b] with truncation order N
            %
            % Suppose you have a continuous function f(x) which is defined on [a,b]
            % which you can evaluate at any x in [a,b]. Simply evaluate it at all of
            % the values contained in the x vector to obtain a vector f. Then compute
            % the definite integral using sum(f.*w);
            %
            % Written by Greg von Winckel - 02/25/2004
            N=N-1;
            N1=N+1; N2=N+2;
            
            xu=linspace(-1,1,N1)';
            
            % Initial guess
            y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);
            
            % Legendre-Gauss Vandermonde Matrix
            L=zeros(N1,N2);
            
            % Derivative of LGVM
            Lp=zeros(N1,N2);
            
            % Compute the zeros of the N+1 Legendre Polynomial
            % using the recursion relation and the Newton-Raphson method
            
            y0=2;
            
            % Iterate until new points are uniformly within epsilon of old points
            while max(abs(y-y0))>eps
                
                
                L(:,1)=1;
                Lp(:,1)=0;
                
                L(:,2)=y;
                Lp(:,2)=1;
                
                for k=2:N1
                    L(:,k+1)=( (2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1) )/k;
                end
             
                Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);   
                
                y0=y;
                y=y0-L(:,N2)./Lp;
                
            end
            
            % Linear map from[-1,1] to [a,b]
            qr.x=flipud((a*(1-y)+b*(1+y))/2);      
            
            % Compute the weights
            qr.w=flipud((b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2);

            if include_boundaries
                % Add endpoints with 0 weight. 
                qr.x = [a; qr.x; b];
                qr.w = [0; qr.w; 0];
            end
        end

        function qr = generalized_laguerre_2(N, alpha)
            % GENLAGUERRE Returns Generalized Gauss-Laguerre nodes and weights
            % Weight: x^alpha * e^-x on [0, inf)
            if nargin < 2 || isempty(alpha), alpha = 0; end
            
            % Golub-Welsch Algorithm via the Jacobi Matrix
            i = 1:N;
            a = 2.*i - 1 + alpha;
            b = sqrt( i(1:N-1) .* (i(1:N-1) + alpha) );
            
            J = diag(a) + diag(b, 1) + diag(b, -1);
            [V, D] = eig(J, 'vector'); % Get eigenvalues as vector
            
            [x, idx] = sort(D);
            V = V(:, idx);
            
            qr.x = x;
            % Weight relies on the first element of the normalized eigenvectors
            qr.w = gamma(alpha + 1) .* (V(1, :)').^2;
        end
        
        function qr = generalized_laguerre(N, alpha)
            % GENLAGUERRE Returns Generalized Gauss-Laguerre nodes and weights.
            % Uses Newton-Raphson polishing and derivative-based weight calculation 
            % to achieve machine precision, matching FastGaussQuadrature.jl logic.
            if nargin < 2 || isempty(alpha), alpha = 0; end
            
            % 1. Initial root guesses via Golub-Welsch eigenvalues
            i = 1:N;
            a = 2.*i - 1 + alpha;
            b = sqrt( i(1:N-1) .* (i(1:N-1) + alpha) );
            J = diag(a) + diag(b, 1) + diag(b, -1);
            x = sort(eig(J)); % Eigenvalues are stable, eigenvectors are not!
            
            % 2. Newton-Raphson root polishing
            for iter = 1:15
                L0 = ones(N, 1);
                L1 = 1 + alpha - x;
                Lp0 = zeros(N, 1);
                Lp1 = -ones(N, 1);
                
                for k = 2:N
                    % 3-term recurrence for Laguerre polynomial and its derivative
                    L2 = ((2*k - 1 + alpha - x) .* L1 - (k - 1 + alpha) .* L0) / k;
                    Lp2 = ((2*k - 1 + alpha - x) .* Lp1 - L1 - (k - 1 + alpha) .* Lp0) / k;
                    
                    L0 = L1; L1 = L2;
                    Lp0 = Lp1; Lp1 = Lp2;
                end
                
                % Newton step
                dx = L1 ./ Lp1;
                x = x - dx;
                
                if max(abs(dx)) < 1e-15
                    break;
                end
            end
            
            % 3. High-Precision Weight Calculation via the Derivative
            % w_i = Gamma(N + alpha + 1) / ( N! * x_i * [L_N'(x_i)]^2 )
            ln_const = gammaln(N + alpha + 1) - gammaln(N + 1);
            w = exp(ln_const) ./ (x .* Lp1.^2);
            
            qr.x = x;
            qr.w = w;
        end



        function qr = halfrange_hermite(N, beta)
            % HALFRANGE_HERMITE Nodes and weights for Half-Range Hermite integration.
            % Computes exact quadratures for: \int_0^\infty f(x) x^2 e^{-\beta x^2} dx
            %
            % beta: Scaling parameter in the exponential (default: 1.0). 
            
            if nargin < 2 || isempty(beta)
                beta = 1.0; 
            end
            
            % 1. Get Generalized Laguerre rule with alpha = 1/2
            lag_qr = Gauss.generalized_laguerre(N, 0.5);
            
            % 2. Map the Laguerre nodes back to the half-range Hermite domain
            qr.x = sqrt(lag_qr.x / beta);
            
            % 3. Apply the differential scale factor (1 / 2*beta^(3/2))
            qr.w = lag_qr.w / (2 * beta^(1.5));
        end
    end
end