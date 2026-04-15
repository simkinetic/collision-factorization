%% BENCHMARK: Algorithmic Scaling of the MEX Collision Tensor
% This script measures the execution time of the C++ MEX tensor generator
% across varying angular (L_max) and radial (K_max) resolutions.
clear; clc; close all;

fprintf('==============================================================\n');
fprintf('  BENCHMARK: C++ MEX Spectral Tensor Generation\n');
fprintf('==============================================================\n\n');

% Define the resolution grids
L_vec = [2, 3, 4, 6, 8, 10, 13];
K_vec = [2, 4, 6];

% Pre-allocate timing matrix [Size: length(K_vec) x length(L_vec)]
time_matrix = zeros(length(K_vec), length(L_vec));

fprintf('Running benchmark... (This may take a minute or two)\n');
fprintf(' L_max |');
for k = K_vec
    fprintf(' K=%-2d |', k);
end
fprintf('\n--------------------------------------------------------------\n');

for j = 1:length(L_vec)
    L_max = L_vec(j);
    fprintf('  %-4d |', L_max);
    
    for i = 1:length(K_vec)
        K_max = K_vec(i);
        
        % Suppress the noisy console output from the class constructors
        % using evalc (evaluates expression and captures command window text)
        junk_output = evalc('Basis = SpectralBasis(K_max, L_max);');
        
        % FIXED: Using the new interface with alpha = 0.0 for Maxwellian
        junk_output = evalc('Kernel = ScatteringKernel(K_max, L_max, 5, 1.0, 1e-6);');
        junk_output = evalc('TensorObj = GeneralCollisionTensor(Basis, Kernel);');
        
        % Measure strictly the Tensor Generation time
        tic;
        TensorObj.generate_R_tensor(true); % use mex
        elapsed_time = toc;
        
        time_matrix(i, j) = elapsed_time;
        fprintf(' %5.2fs |', elapsed_time);
    end
    fprintf('\n');
end
fprintf('--------------------------------------------------------------\n');
fprintf('Benchmark Complete!\n\n');

%% --- High-Quality Journal-Style Visualization ---
% Set LaTeX interpreters for beautiful math formatting
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

figure('Name', 'MEX Tensor Scaling', 'Position', [150, 150, 800, 600], 'Color', 'w');
hold on; grid on;

% Use a beautiful perceptually uniform colormap (turbo or parula)
colors = turbo(length(K_vec) + 1); 
markers = {'o', 's', '^', 'd', 'v', 'p'};
leg_entries = {};

% 1. Plot the actual execution times
for i = 1:length(K_vec)
    plot(L_vec, time_matrix(i, :), ['-', markers{i}], ...
        'Color', colors(i, :), ...
        'MarkerFaceColor', colors(i, :), ...
        'LineWidth', 2.5, ...
        'MarkerSize', 8);
        
    leg_entries{end+1} = sprintf('$K_{\\max} = %d$', K_vec(i));
end

% 2. Plot a Theoretical Reference Slope O(L^3)
% We anchor it to the K=6, L=2 point for visual alignment
anchor_idx = 1; % L=2
anchor_time = time_matrix(end, anchor_idx);
anchor_L = L_vec(anchor_idx);

% O(L^3) scaling means: Time = C * L^3
C_const = anchor_time / (anchor_L^3);
theoretical_time = C_const * (L_vec.^3);

plot(L_vec, theoretical_time, 'k--', 'LineWidth', 2);
leg_entries{end+1} = 'Theoretical $\mathcal{O}(L_{\max}^3)$ Scaling';

% 3. Format the Log-Log Axes Natively
set(gca, 'XScale', 'log', 'YScale', 'log');

% Force the X-axis to display exactly our L_max values (no weird fractions)
xticks(L_vec);
xticklabels(string(L_vec));

% Add nice minor grid lines for logarithmic scales
grid minor;
set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.5);

% Labels and Polish
title('\textbf{Performance Scaling of C++ MEX Collision Tensor}', 'FontSize', 15);
xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', 13);
ylabel('\textbf{Wall-clock Execution Time [Seconds]}', 'FontSize', 13);

legend(leg_entries, 'Location', 'northwest', 'FontSize', 12);
set(gca, 'FontSize', 12, 'LineWidth', 1.2);