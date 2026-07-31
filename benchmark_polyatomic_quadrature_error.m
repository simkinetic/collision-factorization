%% Polyatomic Quadrature Convergence Study (Paper Section 5.2)
% Spectral convergence of the 9D polyatomic singular quadrature, isolating the
% NEW internal-energy quadrature rules introduced in Table 1: the Generalized
% Gauss-Laguerre rules for the target/incident internal energies (I, J) and the
% Gauss-Jacobi rules for the partition parameters (r, R).
%
% Method
% ------
% The padding arguments of generate_R_tensor_sumfac(radial_pad, angular_pad,
% internal_pad) count quadrature points ADDED on top of the minimum bounds that
% make the rules exact for the polynomial part of the integrand. To measure the
% internal convergence in isolation we hold the two SPATIAL paddings fixed and
% identical between the reference and every sweep point: the 5D spatial core then
% uses the exact same nodes everywhere and cancels in the difference, leaving a
% pure measurement of the internal (I, J, r, R) quadrature error.
%
% Two collision models expose the two regimes:
%   * Polyatomic hard potential (gamma = 1): the kernel carries the internal
%     term sqrt((I + J) / m), which is NON-polynomial (an algebraic branch point
%     at I + J = 0). The Gauss-Laguerre / Gauss-Jacobi rules are not exact, and
%     the error decays steadily as internal points are added.
%   * Polyatomic Maxwell (gamma = 0): the kernel B is constant, so the internal
%     integrand is purely polynomial and is integrated EXACTLY at internal_pad
%     = 0 (flat at machine precision).
%
% The L_inf error of the dense physical tensor R is measured relative to an
% over-resolved reference (internal_pad = ref_internal_pad).

clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL');

%% --- Configuration ---
K_max = 2;
L_max = 2;
I_max = 2;
nu    = 1.0;                 % internal d.o.f. parameter, (D-5)/2 -> D = 7

spatial_pad       = 16;      % fixed radial/angular padding (cancels in the diff)
internal_paddings = [0, 1, 2, 3, 4, 5];
ref_internal_pad  = 10;      % over-resolved internal reference

gammas = [1.0, 0.0];        % [Polyatomic hard potential, Polyatomic Maxwell]
errors = zeros(length(internal_paddings), 2);

Basis = SpectralBasis(K_max, L_max, I_max, nu);

for g_idx = 1:length(gammas)
    gamma = gammas(g_idx);
    Kernel = ScatteringKernel('Polyatomic', gamma);

    % Over-resolved reference (same spatial nodes, deep internal quadrature)
    fprintf('--- gamma = %.1f: Reference (internal_pad = %d) ---\n', gamma, ref_internal_pad);
    % Conservation enforcement is switched off throughout: it clamps the invariant
    % rows to exactly zero in both the reference and the test builds, which would
    % remove the cleanest quadrature-convergence indicator from the error metric.
    T_ref = GeneralCollisionTensor(Basis, Kernel);
    T_ref.conserve_invariants = false;
    T_ref.generate_R_tensor_sumfac(spatial_pad, spatial_pad, ref_internal_pad);
    norm_ref = max(abs(T_ref.R_tensor(:)));

    for i = 1:length(internal_paddings)
        ip = internal_paddings(i);
        T_obj = GeneralCollisionTensor(Basis, Kernel);
        T_obj.conserve_invariants = false;
        T_obj.generate_R_tensor_sumfac(spatial_pad, spatial_pad, ip);

        err = max(abs(T_obj.R_tensor(:) - T_ref.R_tensor(:))) / norm_ref;
        errors(i, g_idx) = err;
        fprintf('  internal_pad = %2d | Rel L_inf Error = %e\n', ip, err);
    end
end

%% --- Plot Formatting ---

% save figures?
export_to_pdf_figure = true;

FS_labels = 28; FS_ticks = 18; FS_legend = 18;
c_hard    = [0.0000, 0.4470, 0.7410]; % Blue  (Polyatomic hard potential)
c_maxwell = [0.8500, 0.3250, 0.0980]; % Orange (Polyatomic Maxwell)

% Guard against exact zeros for the log axis
errors_plot = max(errors, 1e-17);

fig_conv = figure('Name', 'Polyatomic Quadrature Convergence', ...
                  'Position', [100, 100, 750, 650], 'Color', 'w');
hold on;

h_maxwell = plot(internal_paddings, errors_plot(:, 2), 's--', 'Color', c_maxwell, ...
    'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_maxwell);

h_hard = plot(internal_paddings, errors_plot(:, 1), 'o-', 'Color', c_hard, ...
    'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_hard);

set(gca, 'YScale', 'log', 'XScale', 'linear');
xticks(internal_paddings); xticklabels(string(internal_paddings));
xlim([min(internal_paddings) - 0.5, max(internal_paddings) + 0.5]);
ylim([1e-16, 1e-2]); yticks(10.^(-16:2:-2));

grid on;
set(gca, 'YMinorGrid', 'off', 'GridLineStyle', '-', 'GridAlpha', 0.2);

xlabel('\textbf{Internal Quadrature Padding} ($N_{\mathrm{pad}}^{\mathrm{int}}$)', ...
    'Interpreter', 'latex', 'FontSize', FS_labels);
ylabel('\textbf{Relative $L_\infty$ Error} ($\epsilon$)', ...
    'Interpreter', 'latex', 'FontSize', FS_labels);
legend([h_hard, h_maxwell], ...
    {'\textbf{Polyatomic Hard Potential} ($\gamma=1$)', ...
     '\textbf{Polyatomic Maxwell} ($\gamma=0$)'}, ...
    'Interpreter', 'latex', 'FontSize', FS_legend, 'Location', 'northeast');

set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex');

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    exportgraphics(gcf, 'fig_polyatomic_quadrature_convergence.pdf', 'ContentType', 'vector');
end
