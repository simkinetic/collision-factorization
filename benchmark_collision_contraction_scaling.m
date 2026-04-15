%% SECTION 4: Final Competition - Dense vs Sparse vs Sliced vs Radial
clear; clc; close all;
K_max = 4;
L_max_list = [4, 6, 8, 10, 13]; 
vhs_omega = 1.0; 

% save figures?
export_to_pdf_figure = false;

t_dense       = zeros(size(L_max_list)); % True Baseline
t_naive       = zeros(size(L_max_list)); % Unoptimized Sparse
t_angular     = zeros(size(L_max_list)); % Angular-First (q1-Sliced)
t_radial      = zeros(size(L_max_list)); % Radial-First

fprintf('Final Hardware Benchmark (K_max = %d)...\n', K_max);
for i = 1:length(L_max_list)
    L = L_max_list(i);
    filename = sprintf('src/precalc/collisiontensor_k%d_l%d_vhs_w%.2f.mat', K_max, L, vhs_omega);
    if ~exist(filename, 'file'), continue; end
    data = load(filename);
    
    K_len = K_max + 1;
    N_Q = max(data.gaunt_labels(:));
    N_terms = N_Q * K_len;
    
    f_test = rand(N_Q, K_len); 
    f_flat = f_test(:); % Flattened strictly for the dense contraction
    
    % --- Pre-processing for True Dense 3D Tensor ---
    fprintf('  Assembling Dense Cartesian Tensor (Size: %d x %d x %d)...\n', N_terms, N_terms, N_terms);
    try
        C_dense = zeros(N_terms, N_terms, N_terms);
        N_G = size(data.gaunt_labels, 1);
        for z = 1:N_G
            q1 = data.gaunt_labels(z, 1);
            q2 = data.gaunt_labels(z, 2);
            q3 = data.gaunt_labels(z, 3);
            g_val = data.gaunt_vals(z);
            t = data.ic_map(z);
            
            for k1 = 0:K_max
                for k2 = 0:K_max
                    for k3 = 0:K_max
                        idx1 = k1 * N_Q + q1;
                        idx2 = k2 * N_Q + q2;
                        idx3 = k3 * N_Q + q3;
                        C_dense(idx1, idx2, idx3) = data.R_tensor(k1+1, k2+1, k3+1, t) * g_val;
                    end
                end
            end
        end
    catch ME
        warning('Out of memory assembling dense tensor for L=%d. Skipping Dense baseline.', L);
        C_dense = [];
    end
    
    % --- Sort for Angular-First and Radial-First (Blocked by t) ---
    [~, sort_idx] = sortrows([data.ic_map, data.gaunt_labels(:, 1)]);
    ic_tq1 = data.ic_map(sort_idx);
    lb_tq1 = data.gaunt_labels(sort_idx, :);
    vs_tq1 = data.gaunt_vals(sort_idx);
    
    % --- Benchmarks ---
    
    % 1. True Dense Baseline
    if ~isempty(C_dense)
        func_dense = @() call_mex_out(@dense_tensor_kernel_mex, zeros(N_terms, 1), f_flat, C_dense(:), N_terms);
        t_dense(i) = timeit(func_dense);
    else
        t_dense(i) = NaN;
    end
    
    % 2. Standard Sparse (Naive)
    func_naive = @() call_mex_out(@naive_collision_kernel_mex, zeros(N_Q, K_len), f_test, ...
                    data.gaunt_labels, data.gaunt_vals, data.ic_map, data.R_tensor, N_Q, K_len);
    t_naive(i) = timeit(func_naive);
    
    % 3. Angular-First (q1-Sliced Cache Winner)
    func_af = @() call_mex_out(@angular_first_collision_kernel_mex, zeros(N_Q, K_len), f_test, ...
                 lb_tq1, vs_tq1, ic_tq1, data.R_tensor, N_Q, K_len);
    t_angular(i) = timeit(func_af);
    
    % 4. Radial-First 
    func_rf = @() call_mex_out(@radial_first_collision_kernel_mex, zeros(N_Q, K_len), f_test, ...
                 lb_tq1, vs_tq1, ic_tq1, data.R_tensor, N_Q, K_len);
    t_radial(i) = timeit(func_rf);
    
    fprintf('  L=%2d | Dense: %.4fs | Sparse: %.4fs | Angular-1st: %.4fs | Radial-1st: %.4fs\n', ...
            L, t_dense(i), t_naive(i), t_angular(i), t_radial(i));
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
xlim([min(L_max_list), max(L_max_list) * 1.1]); 
ylim([min(t_angular)*0.5, max(t_dense)*3]); 

grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

% Legend
legend([h_dense, h_sparse, h_radial, h_angular], ...
    {'\textbf{Dense Cartesian Baseline}', '\textbf{Standard Sparse Contraction}', '\textbf{Radial-First Contraction}', '\textbf{Angular-First Contraction}'}, ...
    'Location', 'northwest', 'FontSize', FS_legend - 2);

% Theoretical Slope Triangles
idx1 = find(L_max_list == 10);
idx2 = find(L_max_list == 13);
L1 = L_max_list(idx1);
L2 = L_max_list(idx2);

draw_slope_triangle(L1, L2, t_dense(idx1) * 0.6, 6, '$\mathcal{O}(L_{\max}^6)$', 1.25, FS_text);
draw_slope_triangle(L1, L2, t_angular(idx1) * 0.6, 5, '$\mathcal{O}(L_{\max}^5)$', 1.25, FS_text);

xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Execution Time [s]}', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    export_fig('fig_execution_time', '-pdf', '-painters');
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
xlim([min(L_max_list), max(L_max_list) * 1.1]);
ylim([0, max(s_angular)*1.15]); 

grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

% Baseline at y=1
yline(1.0, 'k--', 'LineWidth', 2.0, 'HandleVisibility', 'off');
text(L_max_list(2), 1.0 + max(s_angular)*0.03, '\textbf{Dense Baseline ($1.0\times$)}', ...
    'Interpreter', 'latex', 'FontSize', FS_text-2, 'HorizontalAlignment', 'center');

% Add bold text annotations
text(L_max_list(end), s_angular(end) + max(s_angular)*0.05, sprintf('\\textbf{%.1f$\\times$}', s_angular(end)), ...
    'Color', c_angular, 'FontSize', FS_text+2, 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
text(L_max_list(end), s_radial(end) + max(s_angular)*0.05, sprintf('\\textbf{%.1f$\\times$}', s_radial(end)), ...
    'Color', c_radial, 'FontSize', FS_text, 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
text(L_max_list(end), s_sparse(end) - max(s_angular)*0.04, sprintf('\\textbf{%.1f$\\times$}', s_sparse(end)), ...
    'Color', c_sparse, 'FontSize', FS_text, 'HorizontalAlignment', 'center', 'Interpreter', 'latex');

legend({'\textbf{Standard Sparse Contraction}', '\textbf{Radial-First Contraction}', '\textbf{Angular-First Contraction}'}, ...
    'Location', 'northwest', 'FontSize', FS_legend - 2);

xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Speedup Factor} ($T_{\mathrm{dense}} / T_{\mathrm{method}}$)', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    export_fig('fig_relative_speedup', '-pdf', '-painters');
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