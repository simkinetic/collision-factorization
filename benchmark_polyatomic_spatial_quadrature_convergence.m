%% Generate the data for plot_spatial_quadrature_convergence_paper.m (paper Sec. 5.1)
% Spatial-padding convergence of the 9D polyatomic quadrature for the DSMC
% Borgnakke-Larsen kernel at three relative-speed exponents:
%   Maxwell     zeta = 0
%   Hard sphere zeta = 1
%   N2 fit      zeta = 0.533
% Protocol: K_max = L_max = I_max = 2, delta = 2.01, omega = 1 (non-frozen
% channel), spatial padding swept against a pad-32 reference, internal padding
% fixed at 4 everywhere (identical internal nodes -> the internal quadrature
% error is common-mode and cancels in the tensor difference), conservation
% enforcement off (it would clamp the invariant rows to zero in both builds
% and mask the raw quadrature residual).
%
% Writes spatial_quadrature_convergence_data.mat next to this script.
% Runtime: roughly an hour (the pad-32 references dominate).
clear; clc;

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'), fullfile(root,'src','mex'));
addpath(genpath(fullfile(root,'src','SHL')));

K_max = 2; L_max = 2; I_max = 2;
delta = 2.01; omega = 1;
int_pad = 4; ref_pad = 32;
paddings = [0 1 2 4 6 8 10 12 14 16 20 24];

zetas  = [0.0, 1.0, 0.533];
labels = {'err_maxwell', 'err_hardsphere', 'err_n2'};
out = struct('paddings', paddings, 'K_max', K_max, 'L_max', L_max, ...
             'I_max', I_max, 'delta', delta, 'omega', omega, ...
             'int_pad', int_pad, 'ref_pad', ref_pad, 'zetas', zetas);

for z = 1:numel(zetas)
    Kernel = ScatteringKernel('DSMC', struct('zeta',zetas(z),'delta',delta,'omega',omega));
    Basis  = SpectralBasis(K_max, L_max, I_max, Kernel.nu);

    fprintf('=== zeta = %.3f: reference (spatial pad = %d) ===\n', zetas(z), ref_pad);
    T_ref = GeneralCollisionTensor(Basis, Kernel);
    T_ref.conserve_invariants = false;
    T_ref.generate_R_tensor_sumfac(ref_pad, ref_pad, int_pad);
    norm_ref = max(abs(T_ref.R_tensor(:)));

    errs = zeros(numel(paddings), 1);
    for i = 1:numel(paddings)
        p = paddings(i);
        T = GeneralCollisionTensor(Basis, Kernel);
        T.conserve_invariants = false;
        T.generate_R_tensor_sumfac(p, p, int_pad);
        errs(i) = max(abs(T.R_tensor(:) - T_ref.R_tensor(:))) / norm_ref;
        fprintf('  spatial_pad = %2d | Rel L_inf Error = %e\n', p, errs(i));
    end
    out.(labels{z}) = errs;
end

save(fullfile(paper_output_dir(), 'spatial_quadrature_convergence_data.mat'), '-struct', 'out');
fprintf('Saved %s\n', fullfile(paper_output_dir(), 'spatial_quadrature_convergence_data.mat'));
