% =========================================================================
% EXPERIMENT: Isolating the effect of alpha with fixed k and l
% =========================================================================
clear; clc; close all;

% 1. Setup the velocity grid
c = linspace(0, 5, 500); 
x = c.^2;                

% 2. Fix both the radial order and the angular degree
k = 3; 
l = 1; % Fixed angular momentum (forces c^1 behavior at the origin)

% 3. Sweep through arbitrary Laguerre shape parameters (alpha)
alphas = [0.5, 0.6, 0.7, 0.8];
colors = {'k', 'b', 'r', 'g'};

figure('Name', 'Decoupled Alpha Experiment', 'Color', 'w', 'Position', [100, 100, 800, 650]);

% -------------------------------------------------------------------------
% PLOT 1: Pure Associated Laguerre Polynomials (l is irrelevant here)
% -------------------------------------------------------------------------
subplot(2,1,1); hold on;
for i = 1:length(alphas)
    a = alphas(i);
    L = eval_laguerre(k, a, x);
    plot(x, L, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('\\alpha = %.1f', a));
end
title(sprintf('1. Pure Laguerre Polynomials (Fixed k = %d)', k), 'FontSize', 12);
xlabel('Argument x = c^2'); ylabel(sprintf('L_{%d}^{(\\alpha)}(x)', k));
legend('Location', 'best'); grid on;

% -------------------------------------------------------------------------
% PLOT 2: Physical Basis Functions (Fixed l, varying alpha)
% -------------------------------------------------------------------------
subplot(2,1,2); hold on;
for i = 1:length(alphas)
    a = alphas(i);
    
    % Evaluate polynomial with the arbitrary alpha
    L = eval_laguerre(k, a, x);
    
    % Compute standard Laguerre normalization constant
    mu = sqrt((2 * factorial(k)) / gamma(k + a + 1));
    
    % Apply physical weights using the FIXED l
    Phi = mu .* exp(-c.^2 / 2) .* (c.^l) .* L;
    
    plot(c, Phi, 'Color', colors{i}, 'LineWidth', 2, ...
        'DisplayName', sprintf('\\alpha = %.1f  (Fixed l=%d)', a, l));
end
title(sprintf('2. Weighted Functions \\Phi(c) (Fixed k=%d, Fixed l=%d)', k, l), 'FontSize', 12);
xlabel('Relative Speed c'); ylabel('Amplitude \Phi(c)');
legend('Location', 'best'); grid on;

% =========================================================================
% HELPER FUNCTION: Recursive Laguerre Evaluator
% =========================================================================
function L = eval_laguerre(k, alpha, x)
    if k == 0
        L = ones(size(x));
    elseif k == 1
        L = 1 + alpha - x;
    else
        L0 = ones(size(x));
        L1 = 1 + alpha - x;
        for i = 1:(k-1)
            L = ((2*i + 1 + alpha - x) .* L1 - (i + alpha) .* L0) / (i + 1);
            L0 = L1;
            L1 = L;
        end
    end
end