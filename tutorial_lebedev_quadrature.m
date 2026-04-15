% =========================================================================
% Lebedev Quadrature Test and Visualization
% Make sure 'getLebedevSphere.m' is in the same directory as this script.
% =========================================================================

% 1. Generate the quadrature rule
% 74 points correspond to an algebraic order of 13, meaning it integrates 
% polynomials exactly up to degree 13.
degree = 74; 
leb = getLebedevSphere(degree);

% 2. Plot the Lebedev points on the sphere
figure('Name', 'Lebedev Quadrature Points', 'Position', [100, 100, 600, 500]);
plot3(leb.x, leb.y, leb.z, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
axis equal;
grid on;
title(sprintf('Lebedev Quadrature Points (Degree = %d)', degree));
xlabel('X'); ylabel('Y'); zlabel('Z');
view(3);

% 3. Define the monomial test function: f(x,y,z) = x^2 * y^2 * z^2
% This is a polynomial of degree 6, so a Lebedev grid of order 13 
% (74 points) will integrate it exactly.
f = @(x,y,z) (x.^2) .* (y.^2) .* (z.^2);

% 4. Calculate the Numerical Integral
% Since the weights from getLebedevSphere are already normalized to 4*pi,
% the integral is just the sum of (function_values .* weights)
f_eval = f(leb.x, leb.y, leb.z);
numerical_integral = sum(f_eval .* leb.w);

% 5. Calculate the Analytical Integral for comparison
% The exact integral of x^2 * y^2 * z^2 over the unit sphere is 4*pi/105
analytical_integral = (4 * pi) / 105;

% Display the results in the command window
fprintf('--- Integration Results ---\n');
fprintf('Function:            f(x,y,z) = x^2 * y^2 * z^2\n');
fprintf('Analytical Integral: %.15f\n', analytical_integral);
fprintf('Numerical Integral:  %.15f\n', numerical_integral);
fprintf('Absolute Error:      %.4e\n', abs(numerical_integral - analytical_integral));
fprintf('---------------------------\n');

% 6. Visualize the Monomial on the Sphere
% Generate a dense spherical mesh
[X_sphere, Y_sphere, Z_sphere] = sphere(100);

% Evaluate the function on the dense mesh for coloring
C_sphere = f(X_sphere, Y_sphere, Z_sphere);

figure('Name', 'Monomial Visualization', 'Position', [750, 100, 600, 500]);
surf(X_sphere, Y_sphere, Z_sphere, C_sphere, 'EdgeColor', 'none');
axis equal;
colorbar;
colormap('jet');
title('Monomial f(x,y,z) = x^2 y^2 z^2 on the Unit Sphere');
xlabel('X'); ylabel('Y'); zlabel('Z');
view(3);