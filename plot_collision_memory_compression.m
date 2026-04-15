%% BENCHMARK: Extreme Memory Compression & Efficiency (JCP Figures)
clear; clc; close all;

% Do you want to export to PDF figure
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
L_vec = [2, 4, 6, 8, 10, 13, 16];

FS_title  = 32; 
FS_labels = 32; 
FS_ticks = 18; 
FS_text = 18; 
FS_legend = 18;

c_naive = [0.8500, 0.3250, 0.0980]; c_total = [0.0000, 0.4470, 0.7410];
c_phys  = [0.3010, 0.7450, 0.9330]; c_gaunt = [0.4940, 0.1840, 0.5560];

% ========================================================================
% DATA GENERATION
% ========================================================================
naive_GB = zeros(size(L_vec));
total_fact_GB = zeros(size(L_vec));
R_GB = zeros(size(L_vec));
Gaunt_GB = zeros(size(L_vec));
DOF_vec = zeros(size(L_vec));
NG_vec = zeros(size(L_vec));

fprintf('\nComputing results for K_max = %d...\n', K_max);

for i = 1:length(L_vec)
    L = L_vec(i);
    
    evalc('Basis = SpectralBasis(K_max, L);');
    
    % FIXED: Updated to new ScatteringKernel interface (alpha = 0.0)
    evalc('Kernel = ScatteringKernel(K_max, L, 5, 0.0, 1e-6);');
    evalc('TensorObj = GeneralCollisionTensor(Basis, Kernel);');
    
    DOF_vec(i) = Basis.N_terms;
    N_K = K_max + 1;
    N_L = max(TensorObj.ic_map);          
    NG_vec(i) = length(TensorObj.gaunt_vals);   
    
    naive_GB(i) = (8 * (DOF_vec(i)^3)) / (1024^3);
    R_GB(i) = (8 * (N_K^3) * N_L) / (1024^3);
    Gaunt_GB(i) = (20 * NG_vec(i)) / (1024^3); 
    
    total_fact_GB(i) = R_GB(i) + Gaunt_GB(i);
end

% ========================================================================
% OUTPUT TABLE (LaTeX Formatting)
% ========================================================================
fprintf('\nTable: Memory Efficiency and Sparsity (K_max = %d)\n', K_max);
fprintf('%-6s | %-8s | %-12s | %-15s | %-12s | %-12s\n', ...
    'L_max', 'DOFs', 'Gaunt Nonzeros', 'Naive Mem (GB)', 'Fact. Mem (MB)', 'Ratio (F/N)');
fprintf('------------------------------------------------------------------------------------------\n');
for i = 1:length(L_vec)
    ratio = total_fact_GB(i) / naive_GB(i);
    fact_MB = total_fact_GB(i) * 1024;
    
    fprintf('%-6d | %-8d | %-12d | %-15.2e | %-12.2f | %-12.2e\n', ...
        L_vec(i), DOF_vec(i), NG_vec(i), naive_GB(i), fact_MB, ratio);
end
fprintf('------------------------------------------------------------------------------------------\n\n');

%% ========================================================================
% FIGURE 1: Minimalist Log-Log Memory Scaling
% ========================================================================
fig1 = figure('Name', 'Memory Compression Profile', 'Position', [100, 100, 700, 600], 'Color', 'w');
hold on; grid on;

plot(L_vec, R_GB, 'v:', 'Color', c_phys, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', c_phys);
plot(L_vec, Gaunt_GB, '^--', 'Color', c_gaunt, 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', c_gaunt);
h_naive = plot(L_vec, naive_GB, 's-', 'Color', c_naive, 'LineWidth', 3.0, 'MarkerSize', 10, 'MarkerFaceColor', c_naive);
h_total = plot(L_vec, total_fact_GB, 'o-', 'Color', c_total, 'LineWidth', 3.0, 'MarkerSize', 10, 'MarkerFaceColor', c_total);

set(gca, 'XScale', 'log', 'YScale', 'log');
xticks(L_vec); xticklabels(string(L_vec));
xlim([min(L_vec), max(L_vec) * 1.1]); ylim([1e-6, 5e2]);
grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

legend([h_naive, h_total], {'\textbf{Naive Tensor}', '\textbf{Total Factorized}'}, ...
    'Location', 'northwest', 'FontSize', FS_legend);

draw_slope_triangle(10, 16, naive_GB(L_vec==10) * 0.5, 6, '$\mathcal{O}(L_{\max}^6)$', 0.4, FS_text);
draw_slope_triangle(10, 16, Gaunt_GB(L_vec==10) * 0.6, 5, '$\mathcal{O}(L_{\max}^5)$', 0.4, FS_text);
draw_slope_triangle(6, 10, R_GB(L_vec==6) * 0.5, 3, '$\mathcal{O}(L_{\max}^3)$', 0.4, FS_text);

% title(sprintf('\\textbf{Tensor Memory Footprint} ($K_{\\max} = %d$)', K_max), 'FontSize', FS_title);
xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Required Memory [GB]}', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); 
    set(gca, 'Color', 'w');
    export_fig('fig_memory_scaling', '-pdf', '-painters');
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
xlim([min(L_vec), max(L_vec)]); ylim([min(memory_ratio)*0.5, 2]);
grid minor; set(gca, 'MinorGridLineStyle', ':', 'MinorGridAlpha', 0.4);

yline(1.0, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% title('\textbf{Relative Memory Efficiency}', 'FontSize', FS_title);
xlabel('\textbf{Angular Resolution} ($L_{\max}$)', 'FontSize', FS_labels);
ylabel('\textbf{Memory Ratio} (Factorized / Naive)', 'FontSize', FS_labels);
set(gca, 'FontSize', FS_ticks, 'LineWidth', 1.5);

if export_to_pdf_figure
    set(gcf, 'Color', 'w'); 
    set(gca, 'Color', 'w');
    export_fig('fig_rel_memory_efficiency', '-pdf', '-painters');
end

%% ========================================================================
% HELPER FUNCTIONS
% ========================================================================
function draw_slope_triangle(x1, x2, y1, slope_val, label_str, text_offset, font_size)
    y2 = y1 * (x2 / x1)^slope_val;
    plot([x1, x2, x2, x1], [y1, y1, y2, y1], 'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    
    x_mid = sqrt(x1 * x2);
    y_text = (slope_val >= 0) * (y1 * text_offset) + (slope_val < 0) * (y2 * text_offset);
    text(x_mid, y_text, label_str, 'HorizontalAlignment', 'center', 'FontSize', font_size);
end