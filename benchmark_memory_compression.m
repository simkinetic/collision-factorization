%% BENCHMARK: Extreme Memory Compression & Efficiency (JCP Figures)
% Measures memory footprint using precomputed Wigner-Eckart factorized tensors.
clear; clc; close all;
addpath('src', 'src/mex', 'src/SHL', 'src/precalc');

% Do you want to export to PDF figure?
export_to_pdf_figure = false;

% ========================================================================
% GLOBAL LATEX FONT CONFIGURATION
% ========================================================================
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

% ========================================================================
% CONFIGURATION: Resolution & Plot Formatting
% ========================================================================
K_max = 4; 
L_vec = [2, 4, 6, 8, 10, 12, 14, 16];
gamma = 1.0; % Hard spheres (to match precomputed files)

FS_title  = 32; 
FS_labels = 32; 
FS_ticks  = 18; 
FS_text   = 18; 
FS_legend = 18;

c_naive = [0.8500, 0.3250, 0.0980]; c_total = [0.0000, 0.4470, 0.7410];
c_phys  = [0.3010, 0.7450, 0.9330]; c_gaunt = [0.4940, 0.1840, 0.5560];

% ========================================================================
% DATA EXTRACTION FROM PRECOMPUTED FILES
% ========================================================================
% Pre-allocate with NaNs to cleanly drop missing files in plots
naive_GB      = NaN(size(L_vec));
total_fact_GB = NaN(size(L_vec));
R_GB          = NaN(size(L_vec));
Gaunt_GB      = NaN(size(L_vec));
DOF_vec       = NaN(size(L_vec));
NG_vec        = NaN(size(L_vec));

fprintf('\nComputing memory results for K_max = %d...\n', K_max);

for i = 1:length(L_vec)
    L = L_vec(i);
    
    % The counts need only the Gaunt geometry, which the constructor generates (no quadrature).
    data.Basis = SpectralBasis(K_max, L);
    data.TensorObj = GeneralCollisionTensor(data.Basis, ScatteringKernel(gamma));
    
    DOF_vec(i) = data.Basis.N_terms; % (K_max + 1) * (L_max + 1)^2
    N_K = K_max + 1;
    N_L = max(data.TensorObj.ic_map);          
    NG_vec(i) = length(data.TensorObj.gaunt_vals);   
    
    % Memory in decimal GB: 8 bytes per entry of R, 20 bytes per Gaunt nonzero (3 x uint32 + 1 x double)
    naive_GB(i) = (8 * (DOF_vec(i)^3)) / 1e9;
    R_GB(i)     = (8 * (N_K^3) * N_L) / 1e9;
    Gaunt_GB(i) = (20 * NG_vec(i)) / 1e9; 
    
    total_fact_GB(i) = R_GB(i) + Gaunt_GB(i);
end

% ========================================================================
% OUTPUT TABLE (LaTeX Formatting)
% ========================================================================
fprintf('\nTable: Memory Efficiency and Sparsity (K_max = %d)\n', K_max);
fprintf('%-6s | %-8s | %-14s | %-15s | %-14s | %-12s\n', ...
    'L_max', 'DOFs', 'Gaunt Nonzeros', 'Naive Mem (GB)', 'Fact. Mem (MB)', 'Ratio (F/N)');
fprintf('------------------------------------------------------------------------------------------\n');
for i = 1:length(L_vec)
    if isnan(total_fact_GB(i)), continue; end
    
    ratio = total_fact_GB(i) / naive_GB(i);
    fact_MB = total_fact_GB(i) * 1000;
    
    fprintf('%-6d | %-8d | %-14d | %-15.2e | %-14.2f | %-12.2e\n', ...
        L_vec(i), DOF_vec(i), NG_vec(i), naive_GB(i), fact_MB, ratio);
end
fprintf('------------------------------------------------------------------------------------------\n\n');

%% ========================================================================
% FIGURE 1: Minimalist Log-Log Memory Scaling
% ========================================================================
fig1 = figure('Name', 'Memory Compression Profile', 'Position', [100, 100, 700, 600], 'Color', 'w');
hold on; grid on;

h_R = plot(L_vec, R_GB, 'v:', 'Color', c_phys, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', c_phys);
h_G = plot(L_vec, Gaunt_GB, '^--', 'Color', c_gaunt, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', c_gaunt);
h_naive = plot(L_vec, naive_GB, 's-', 'Color', c_naive, 'LineWidth', 3.0, 'MarkerSize', 10, 'MarkerFaceColor', c_naive);
h_total = plot(L_vec, total_fact_GB, 'o-', 'Color', c_total, 'LineWidth', 3.0, 'MarkerSize', 10, 'MarkerFaceColor', c_total);

set(gca, 'XScale', 'log', 'YScale', 'log');
xticks(L_vec); xticklabels(string(L_vec));
xlim([min(L_vec) * 0.9, max(L_vec) * 1.1]); 
ylim([1e-6, max(naive_GB) * 5]);
grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

legend([h_naive, h_total, h_G, h_R], {'\textbf{Dense Cartesian tensor}', '\textbf{Factorized (total)}', '\textbf{Gaunt tensor (COO)}', '\textbf{Impact tensor} $\mathcal{R}$'}, ...
    'Location', 'northwest', 'FontSize', FS_legend);

% Dynamically place Theoretical Slope Triangles based on valid loaded data
valid_idx = find(~isnan(naive_GB));
if length(valid_idx) >= 2
    % Grab the last two valid data points to draw the triangles
    idx1 = valid_idx(end-1);
    idx2 = valid_idx(end);
    L1 = L_vec(idx1);
    L2 = L_vec(idx2);
    
    draw_slope_triangle(L1, L2, naive_GB(idx1) * 0.5, 6, '$\mathcal{O}(L_{\max}^6)$', 0.4, FS_text);
    draw_slope_triangle(L1, L2, Gaunt_GB(idx1) * 0.6, 5, '$\mathcal{O}(L_{\max}^5)$', 0.4, FS_text);
    
    if length(valid_idx) >= 3
        idx0 = valid_idx(end-2);
        L0 = L_vec(idx0);
        draw_slope_triangle(L0, L1, R_GB(idx0) * 0.5, 3, '$\mathcal{O}(L_{\max}^3)$', 0.4, FS_text);
    end
end

xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Memory [GB]}', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    exportgraphics(gcf, 'fig_memory_scaling.pdf', 'ContentType', 'vector');
end

%% ========================================================================
% FIGURE 2: Relative Memory Efficiency (Normalized Ratio)
% ========================================================================
fig2 = figure('Name', 'Relative Memory Efficiency', 'Position', [850, 100, 700, 600], 'Color', 'w');
hold on; grid on;

memory_ratio = total_fact_GB ./ naive_GB;
plot(L_vec, memory_ratio, 'd-', 'Color', c_total, 'LineWidth', 3.0, 'MarkerSize', 12, 'MarkerFaceColor', c_total);

set(gca, 'XScale', 'log', 'YScale', 'log');
xticks(L_vec); xticklabels(string(L_vec));
xlim([min(L_vec) * 0.9, max(L_vec) * 1.1]); 
ylim([min(memory_ratio)*0.5, 2]);
grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

yline(1.0, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Memory ratio} (factorized / dense)', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    exportgraphics(gcf, 'fig_rel_memory_efficiency.pdf', 'ContentType', 'vector');
end

%% ========================================================================
% HELPER FUNCTIONS
% ========================================================================
function draw_slope_triangle(x1, x2, y1, slope_val, label_str, text_offset, font_size)
    y2 = y1 * (x2 / x1)^slope_val;
    plot([x1, x2, x2, x1], [y1, y1, y2, y1], 'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    
    x_mid = sqrt(x1 * x2);
    y_text = (slope_val >= 0) * (y1 * text_offset) + (slope_val < 0) * (y2 * text_offset);
    text(x_mid, y_text, label_str, 'HorizontalAlignment', 'center', ...
        'FontSize', font_size, 'Interpreter', 'latex');
end