%% Quadrature Convergence Study: Hard Spheres vs Maxwell Molecules
clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL');

% --- Configuration ---
K_max = 4;
L_max = 4;
paddings = [0, 2, 4, 8, 16, 32];
alphas = [1.0, 0.0]; % [Hard Spheres, Maxwell Molecules]
errors = zeros(length(paddings), 2);

Basis = SpectralBasis(K_max, L_max);

for a_idx = 1:2
    alpha = alphas(a_idx);
    Kernel = ScatteringKernel(alpha);
    
    % Compute Reference (Pad = 64)
    fprintf('--- Alpha = %.1f: Reference (Pad=64) ---\n', alpha);
    T_ref = GeneralCollisionTensor(Basis, Kernel);
    T_ref.generate_R_tensor_sumfac(64, 64);
    norm_ref = max(abs(T_ref.R_tensor(:)));
    
    % Compute Paddings
    for i = 1:length(paddings)
        pad = paddings(i);
        T_obj = GeneralCollisionTensor(Basis, Kernel);
        T_obj.generate_R_tensor_sumfac(pad, pad);
        
        err = max(abs(T_obj.R_tensor(:) - T_ref.R_tensor(:))) / norm_ref;
        errors(i, a_idx) = err;
        fprintf('  Pad = %d | Rel Error = %e\n', pad, err);
    end
end

%% --- Plot Formatting ---

% save figures?
export_to_pdf_figure = true;

FS_labels = 28; FS_ticks = 18; FS_legend = 18;
c_angular = [0.0000, 0.4470, 0.7410]; % Blue (Hard Spheres)
c_dense   = [0.8500, 0.3250, 0.0980]; % Orange-Red (Maxwell)

fig_conv = figure('Name', 'Quadrature Convergence', 'Position', [100, 100, 750, 650], 'Color', 'w');
hold on; 

% Plot Maxwell Molecules first (usually converges slightly faster/cleaner)
h_maxwell = plot(paddings, errors(:, 2), 's--', 'Color', c_dense, ...
    'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_dense);

% Plot Hard Spheres
h_hs = plot(paddings, errors(:, 1), 'o-', 'Color', c_angular, ...
    'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_angular);

% Axis scaling and limits
set(gca, 'YScale', 'log', 'XScale', 'linear');
xticks(paddings); xticklabels(string(paddings));
xlim([min(paddings) - 2, max(paddings) + 2]); 
ylim([1e-16, 1e-2]); yticks(10.^(-16:2:-2));

% Grid formatting
grid on; 
set(gca, 'YMinorGrid', 'off', 'GridLineStyle', '-', 'GridAlpha', 0.2); 

% Labels & Legend
xlabel('\textbf{Quadrature Padding} ($N_{\mathrm{pad}}$)', 'Interpreter', 'latex', 'FontSize', FS_labels);
ylabel('\textbf{Relative $L_\infty$ Error} ($\epsilon$)', 'Interpreter', 'latex', 'FontSize', FS_labels);
legend([h_hs, h_maxwell], {'\textbf{Hard Spheres} ($\gamma=1$)', '\textbf{Maxwell Molecules} ($\gamma=0$)'}, ...
    'Interpreter', 'latex', 'FontSize', FS_legend, 'Location', 'southwest');

set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex');

% Export
if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    exportgraphics(gcf, 'fig_quadrature_convergence.pdf', 'ContentType', 'vector');
end