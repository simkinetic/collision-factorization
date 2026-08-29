% benchmark_collision_contraction_scaling.m
% =========================================================================
% DEPRECATED -- MONATOMIC LEGACY BENCHMARK. NOT A POLYATOMIC PAPER ARTIFACT.
%
% WHAT THIS SCRIPT IS
%   The contraction-timing benchmark of the MONATOMIC predecessor study,
%   Hiemstra, Kessler & Abdelmalik, "Wigner-Eckart factorization of the
%   spectral Boltzmann collision operator" (arXiv:2605.28475) -- the source
%   of the 37.2x speedup the polyatomic paper cites. It is kept here,
%   unchanged, as the artifact of that paper, and not because it plays any
%   part in the polyatomic results.
%
% IT CANNOT REPRODUCE SECTION 5.6 OF THE POLYATOMIC PAPER
%   This script is monatomic. It runs at K_max = 4, there is no internal
%   truncation I_max anywhere in it, and it reads the gamma-named monatomic
%   caches collisiontensor_k%d_l%d_gamma%.2f.mat. The polyatomic Section 5.6
%   is at K_max = 2 and sweeps I_max = 0..4. No timing printed or plotted
%   below therefore appears in Section 5.6, in Table 5, or in Figures 9
%   and 10 -- in particular the polyatomic paper's 40.6x / 7.8x / 5.8x
%   dense-baseline speedups at (L_max, I_max) = (10, 2) do not come from
%   here.
%
%   >>> Section 5.6 of the polyatomic paper -- the storage counts, the
%   >>> geometry and slice counts, and the contraction timings, i.e.
%   >>> Table 5 and Figures 9 and 10 -- is produced SOLELY by
%   >>> benchmark_polyatomic_performance.m, which builds everything in
%   >>> memory and reads no cache at all.
%
% THE CACHES IT NEEDS ARE NOT SHIPPED
%   src/precalc/ contains no collisiontensor_k4_l*_gamma1.00.mat files:
%   these are large monatomic hard-sphere tensors and are deliberately not
%   distributed. On a clean checkout every L_max therefore prints
%   "[SKIP] File missing", all four timing curves stay NaN, and the first
%   figure then fails in its ylim() call with
%   MATLAB:rulerFunctions:InvalidNumericLimits because min(t_angular) and
%   max(t_dense) are NaN. That is the expected behaviour of an
%   un-regenerated legacy benchmark, not a bug in the library.
%
%   To run it, regenerate the caches first with
%   precompute_collision_operator.m, edited to the monatomic settings this
%   script asks for:
%
%       K_max      = 4;                        % already its default
%       L_max_list = [2, 4, 6, 8, 10, 12];     % it ships as [2, 4, 6]
%       gamma      = 1.0;                      % it ships as 0.0 (Maxwell)
%
%   gamma = 1.0 is hard spheres, which is what the filename assembled below
%   asks for. Budget hours of quadrature and tens of GB on disk: the upper
%   L_max entries at the script's pad = 20 dominate both. The dense
%   Cartesian baseline assembled below additionally needs many GB of RAM at
%   the upper L_max values, and drops out with a warning where it does not
%   fit.
% =========================================================================

%% SECTION 4: Final Competition - Dense vs Sparse vs Sliced vs Radial
% Benchmarks the execution time of four distinct tensor contraction
% algorithms using precomputed Wigner-Eckart factorized collision tensors.
clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL', 'src/precalc');

K_max = 4;
L_max_list = [2, 4, 6, 8, 10, 12]; % Updated to match the generated precalc files
gamma = 1.0; % Hard spheres (to match precomputed files)

% save figures?
export_to_pdf_figure = false;

t_dense       = NaN(size(L_max_list)); % True Baseline
t_naive       = NaN(size(L_max_list)); % Unoptimized Sparse
t_angular     = NaN(size(L_max_list)); % Angular-First (q1-Sliced)
t_radial      = NaN(size(L_max_list)); % Radial-First

fprintf('==============================================================\n');
fprintf('Final Hardware Benchmark (K_max = %d)...\n', K_max);
fprintf('==============================================================\n');

for i = 1:length(L_max_list)
    L = L_max_list(i);
    
    % Updated to load the new gamma-based filenames
    filename = sprintf('collisiontensor_k%d_l%d_gamma%.2f.mat', K_max, L, gamma);
    filepath = fullfile('src', 'precalc', filename);
    
    if ~exist(filepath, 'file')
        fprintf('  [SKIP] File missing: %s\n', filename);
        continue; 
    end
    
    % Load the object-oriented precomputed data
    data = load(filepath, 'TensorObj', 'Basis');
    TensorObj = data.TensorObj;
    Basis = data.Basis;
    
    K_len = K_max + 1;
    N_Q = Basis.N_Q; % Extract exact angular DOFs from the Basis object
    N_terms = N_Q * K_len;
    
    f_test = rand(N_Q, K_len); 
    f_flat = f_test(:); % Flattened strictly for the dense contraction
    
    % Extract internal arrays for the MEX functions
    g_labels = TensorObj.gaunt_labels;
    g_vals   = TensorObj.gaunt_vals;
    ic_map   = TensorObj.ic_map;
    R_tensor = TensorObj.R_tensor;
    
    % --- Pre-processing for True Dense 3D Tensor ---
    fprintf('  Assembling Dense Cartesian Tensor for L=%d (Size: %d x %d x %d)...\n', L, N_terms, N_terms, N_terms);
    try
        C_dense = zeros(N_terms, N_terms, N_terms);
        N_G = size(g_labels, 1);
        for z = 1:N_G
            q1 = g_labels(z, 1);
            q2 = g_labels(z, 2);
            q3 = g_labels(z, 3);
            g_val = g_vals(z);
            t = ic_map(z);
            
            for k1 = 0:K_max
                for k2 = 0:K_max
                    for k3 = 0:K_max
                        idx1 = k1 * N_Q + q1;
                        idx2 = k2 * N_Q + q2;
                        idx3 = k3 * N_Q + q3;
                        C_dense(idx1, idx2, idx3) = R_tensor(k1+1, k2+1, k3+1, t) * g_val;
                    end
                end
            end
        end
    catch ME
        warning('Out of memory assembling dense tensor for L=%d. Skipping Dense baseline.', L);
        C_dense = [];
    end
    
    % --- Sort for Angular-First and Radial-First (Blocked by t) ---
    [~, sort_idx] = sortrows([ic_map, g_labels(:, 1)]);
    ic_tq1 = ic_map(sort_idx);
    lb_tq1 = g_labels(sort_idx, :);
    vs_tq1 = g_vals(sort_idx);
    
    % --- Benchmarks ---
    
    % 1. True Dense Baseline
    if ~isempty(C_dense)
        func_dense = @() call_mex_out(@dense_tensor_kernel_mex, zeros(N_terms, 1), f_flat, C_dense(:), N_terms);
        t_dense(i) = timeit(func_dense);
    end
    
    % 2. Standard Sparse (Naive)
    func_naive = @() call_mex_out(@naive_collision_kernel_mex, zeros(N_Q, K_len), f_test, ...
                    g_labels, g_vals, ic_map, R_tensor, N_Q, K_len);
    t_naive(i) = timeit(func_naive);
    
    % 3. Angular-First (q1-Sliced Cache Winner)
    func_af = @() call_mex_out(@angular_first_collision_kernel_mex, zeros(N_Q, K_len), f_test, ...
                 lb_tq1, vs_tq1, ic_tq1, R_tensor, N_Q, K_len);
    t_angular(i) = timeit(func_af);
    
    % 4. Radial-First 
    func_rf = @() call_mex_out(@radial_first_collision_kernel_mex, zeros(N_Q, K_len), f_test, ...
                 lb_tq1, vs_tq1, ic_tq1, R_tensor, N_Q, K_len);
    t_radial(i) = timeit(func_rf);
    
    fprintf('  -> Dense: %.4fs | Sparse: %.4fs | Angular-1st: %.4fs | Radial-1st: %.4fs\n\n', ...
            t_dense(i), t_naive(i), t_angular(i), t_radial(i));
end

%% ========================================================================
% GLOBAL LATEX FONT CONFIGURATION
% ========================================================================
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

% ========================================================================
% CONFIGURATION: Resolution & Plot Formatting
% ========================================================================
FS_title  = 24; 
FS_labels = 28; 
FS_ticks  = 18; 
FS_text   = 18; 
FS_legend = 18;

% Clean, distinct journal colors
c_dense   = [0.8500, 0.3250, 0.0980]; % Orange-Red
c_sparse  = [0.4940, 0.1840, 0.5560]; % Purple
c_angular = [0.0000, 0.4470, 0.7410]; % Blue
c_radial  = [0.4660, 0.6740, 0.1880]; % Green

%% ========================================================================
% FIGURE 1: Minimalist Log-Log Execution Time Scaling
% ========================================================================
fig1 = figure('Name', 'Execution Time Profile', 'Position', [100, 100, 750, 650], 'Color', 'w');
hold on; grid on;

h_sparse  = plot(L_max_list, t_naive, '^--', 'Color', c_sparse, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', c_sparse);
h_radial  = plot(L_max_list, t_radial, 'd-.', 'Color', c_radial, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', c_radial);
h_angular = plot(L_max_list, t_angular, 'o-', 'Color', c_angular, 'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_angular);
h_dense   = plot(L_max_list, t_dense, 's-', 'Color', c_dense, 'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_dense);

% Axis formatting
set(gca, 'XScale', 'log', 'YScale', 'log');
xticks(L_max_list); xticklabels(string(L_max_list));
xlim([min(L_max_list) * 0.9, max(L_max_list) * 1.1]); 
ylim([min(t_angular)*0.5, max(t_dense)*3]); 
grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

% Legend
legend([h_dense, h_sparse, h_radial, h_angular], ...
    {'\textbf{Dense Cartesian Baseline}', '\textbf{Standard Sparse Contraction}', '\textbf{Radial-First Contraction}', '\textbf{Angular-First Contraction}'}, ...
    'Location', 'northwest', 'FontSize', FS_legend - 2);

% Theoretical Slope Triangles (Dynamically placed on the last two points)
valid_idx = find(~isnan(t_dense), 2, 'last');
if length(valid_idx) == 2
    L1 = L_max_list(valid_idx(1));
    L2 = L_max_list(valid_idx(2));
    draw_slope_triangle(L1, L2, t_dense(valid_idx(1)) * 0.6, 6, '$\mathcal{O}(L_{\max}^6)$', 1.25, FS_text);
    draw_slope_triangle(L1, L2, t_angular(valid_idx(1)) * 0.6, 5, '$\mathcal{O}(L_{\max}^5)$', 1.25, FS_text);
end

xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Execution Time [s]}', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    exportgraphics(gcf, 'fig_execution_time.pdf', 'ContentType', 'vector');
end

%% ========================================================================
% FIGURE 2: Relative Speedup Factor
% ========================================================================
s_sparse  = t_dense ./ t_naive;
s_radial  = t_dense ./ t_radial;
s_angular = t_dense ./ t_angular;

fig2 = figure('Name', 'Relative Speedup Factor', 'Position', [900, 100, 750, 650], 'Color', 'w');
hold on; grid on;

plot(L_max_list, s_sparse, '^--', 'Color', c_sparse, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', c_sparse);
plot(L_max_list, s_radial, 'd-.', 'Color', c_radial, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', c_radial);
plot(L_max_list, s_angular, 'o-', 'Color', c_angular, 'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_angular);

% Axis formatting 
set(gca, 'XScale', 'log', 'YScale', 'linear');
xticks(L_max_list); xticklabels(string(L_max_list));
xlim([min(L_max_list) * 0.9, max(L_max_list) * 1.1]);
ylim([0, max(s_angular)*1.15]); 
grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

% Baseline at y=1
yline(1.0, 'k--', 'LineWidth', 2.0, 'HandleVisibility', 'off');
text(L_max_list(2), 1.0 + max(s_angular)*0.03, '\textbf{Dense Baseline ($1.0\times$)}', ...
    'Interpreter', 'latex', 'FontSize', FS_text-2, 'HorizontalAlignment', 'center');

% Add bold text annotations
last_valid = find(~isnan(s_angular), 1, 'last');
if ~isempty(last_valid)
    text(L_max_list(last_valid), s_angular(last_valid) + max(s_angular)*0.05, sprintf('\\textbf{%.1f$\\times$}', s_angular(last_valid)), ...
        'Color', c_angular, 'FontSize', FS_text+2, 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
    text(L_max_list(last_valid), s_radial(last_valid) + max(s_angular)*0.05, sprintf('\\textbf{%.1f$\\times$}', s_radial(last_valid)), ...
        'Color', c_radial, 'FontSize', FS_text, 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
    text(L_max_list(last_valid), s_sparse(last_valid) - max(s_angular)*0.04, sprintf('\\textbf{%.1f$\\times$}', s_sparse(last_valid)), ...
        'Color', c_sparse, 'FontSize', FS_text, 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
end

legend({'\textbf{Standard Sparse Contraction}', '\textbf{Radial-First Contraction}', '\textbf{Angular-First Contraction}'}, ...
    'Location', 'northwest', 'FontSize', FS_legend - 2);

xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Speedup Factor} ($T_{\mathrm{dense}} / T_{\mathrm{method}}$)', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    exportgraphics(gcf, 'fig_relative_speedup.pdf', 'ContentType', 'vector');
end

%% ========================================================================
% HELPER FUNCTIONS
% ========================================================================
function draw_slope_triangle(x1, x2, y1, slope_val, label_str, text_offset, font_size)
    y2 = y1 * (x2 / x1)^slope_val;
    plot([x1, x2, x2, x1], [y1, y1, y2, y1], 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    x_mid = sqrt(x1 * x2);
    if slope_val >= 0
        y_text = y1 * text_offset;
    else
        y_text = y2 * text_offset;
    end
    text(x_mid, y_text, label_str, 'HorizontalAlignment', 'center', ...
        'FontSize', font_size, 'Interpreter', 'latex');
end

function out = call_mex_out(f, out, varargin)
    f(out, varargin{:});
end