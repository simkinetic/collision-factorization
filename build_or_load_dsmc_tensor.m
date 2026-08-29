function [TensorObj, Basis, Kernel, filepath, was_cached] = build_or_load_dsmc_tensor(cfg)
% BUILD_OR_LOAD_DSMC_TENSOR  Cache layer for polyatomic DSMC collision tensors.
%
% Builds (or loads, if already cached) the DSMC Borgnakke-Larsen collision
% tensor for a given configuration and persists it to src/precalc/ using the
% same save convention as precompute_collision_operator.m (TensorObj, Basis,
% Kernel, saved -v7.3), with the configuration encoded in the filename:
%
%   collisiontensor_dsmc_k{K}_l{L}_i{I}_z{zeta}_d{delta}_w{omega}_p{rad}-{tan}-{int}[_lapNs{Ns}][_pfshift][_etr43|_etr43spec|_noetr].mat
%
% {int} is the EFFECTIVE internal padding: on the spectral path (laplace) the
% internal-sector axes are exact at the Table-1 counts, the class clamps the
% request to 0, and the name records 0 (asserted post-build). Legacy spectral
% caches named with the requested padding are still served (read fallback).
%
% cfg fields (defaults in brackets):
%   K_max, L_max, I_max        spectral resolution        [2 2 2]
%   zeta, delta, omega         DSMC kernel parameters     (required)
%   rad_pad, tan_pad, int_pad  quadrature padding         [10 10 10]
%   eta_hat, zeta_hat, eta_hat_f, zeta_hat_f   extended model [0 0 0 0]
%   laplace, laplace_Ns        spectral Laplace path      [false, 32]
%   conserve                   enforce the 5 invariants   [true]
%   frozen_no_etr              drop eq.(43) e_tr^{z/2}    [false]
%   loss_only                  zero the gain kernels      [false]
%
% loss_only builds the diagnostic LOSS-ONLY operator (GeneralCollisionTensor
% .loss_only) and tags the cache filename with _lossonly, so a loss build can
% never be served into a run that asked for the full operator, or vice versa.
% Pair it with conserve = false (the loss operator is not conservative).
%
% NOTE: this is an additive helper; it does not modify any existing script.
% Existing benchmarks do not load these caches (they build in-memory); the
% cache is for reuse by new drivers / interactive sessions.

    root = fileparts(mfilename('fullpath'));
    addpath(fullfile(root,'src'), fullfile(root,'src','mex'));
    addpath(genpath(fullfile(root,'src','SHL')));

    def = struct('K_max',2,'L_max',2,'I_max',2, ...
                 'rad_pad',10,'tan_pad',10,'int_pad',10, ...
                 'eta_hat',0,'zeta_hat',0,'eta_hat_f',0,'zeta_hat_f',0, ...
                 'laplace',[],'laplace_Ns',32,'conserve',true, ...
                 'frozen_no_etr',false,'loss_only',false);
    fn = fieldnames(def);
    for i = 1:numel(fn)
        if ~isfield(cfg, fn{i}), cfg.(fn{i}) = def.(fn{i}); end
    end
    % Default path: Laplace for extended kernels (spectral accuracy; matches the
    % class default), algebraic for pure base kernels (paths agree to machine
    % precision there; keeps existing untagged base-cache filenames valid).
    if isempty(cfg.laplace)
        cfg.laplace = (cfg.eta_hat ~= 0 || cfg.eta_hat_f ~= 0);
    end

    ext = '';
    if cfg.eta_hat ~= 0 || cfg.eta_hat_f ~= 0
        ext = sprintf('_eh%s_zh%s_ehf%s_zhf%s', num2str(cfg.eta_hat), ...
              num2str(cfg.zeta_hat), num2str(cfg.eta_hat_f), num2str(cfg.zeta_hat_f));
    end
    if cfg.laplace, ext = [ext sprintf('_lapNs%d', cfg.laplace_Ns)]; end
    % Rule-version tag: the non-frozen modulated correction now folds the
    % partition prefactors (r(1-R))^{zh/2}, ((1-r)(1-R))^{zh/2} into SHIFTED
    % Gauss-Jacobi rules (spectral) instead of applying them pointwise on the
    % unshifted rules (algebraic). Tensors built on the laplace path with
    % eta_hat ~= 0 before this change carry the pointwise-prefactor error under
    % an otherwise identical filename; the tag makes them unreachable so a
    % stale cache can never be served into a production run. Frozen-only
    % modulation (eta_hat == 0, eta_hat_f ~= 0) is unaffected by the change
    % and keeps its filename.
    if cfg.laplace && cfg.eta_hat ~= 0, ext = [ext '_pfshift']; end
    if ~cfg.conserve, ext = [ext '_nocons']; end
    % Loss-only diagnostic builds get their own key: a loss tensor must never be
    % served to a caller that asked for the full operator (or vice versa).
    if cfg.loss_only, ext = [ext '_lossonly']; end
    % Frozen-channel tag. Only omega < 1 builds the frozen channel at all, so
    % omega == 1 filenames are unaffected (existing caches stay valid). For
    % omega < 1 the tag records WHICH frozen treatment produced the file:
    %   _noetr     eq.-(53) opt-out, no e_tr weight
    %   _etr43     eq.-(43) e_tr^{zeta/2} applied POINTWISE (algebraic path)
    %   _etr43spec eq.-(43) e_tr^{zeta/2} resolved SPECTRALLY by the frozen
    %              auxiliary Laplace integral (the laplace path)
    % The last distinction is load-bearing: tensors built on the laplace path
    % before the spectral frozen treatment landed carry the pointwise weight
    % under an otherwise identical filename, and would otherwise be served as
    % stale cache hits into a production run.
    if cfg.omega < 1
        if cfg.frozen_no_etr
            ext = [ext '_noetr'];
        elseif cfg.laplace
            ext = [ext '_etr43spec'];
        else
            ext = [ext '_etr43'];
        end
    end

    % EFFECTIVE-padding naming: on the spectral path the internal-sector axes
    % are exact at the Table-1 node counts and GeneralCollisionTensor clamps
    % any requested internal padding to zero (clamp_internal_pad, default
    % true). Filenames record the EFFECTIVE internal padding, so names and
    % contents can never diverge; a post-build assertion enforces it.
    int_pad_eff = cfg.int_pad;
    if cfg.laplace, int_pad_eff = 0; end
    mkname = @(ip) sprintf('collisiontensor_dsmc_k%d_l%d_i%d_z%s_d%s_w%s_p%d-%d-%d%s.mat', ...
        cfg.K_max, cfg.L_max, cfg.I_max, num2str(cfg.zeta), num2str(cfg.delta), ...
        num2str(cfg.omega), cfg.rad_pad, cfg.tan_pad, ip, ext);
    filename = mkname(int_pad_eff);
    out_dir = fullfile(root, 'src', 'precalc');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    filepath = fullfile(out_dir, filename);

    if exist(filepath, 'file')
        S = load(filepath, 'TensorObj', 'Basis', 'Kernel');
        TensorObj = S.TensorObj; Basis = S.Basis; Kernel = S.Kernel;
        was_cached = true;
        fprintf('[cache hit ] %s\n', filename);
        return;
    end

    % Legacy fallback: spectral caches written before the effective-padding
    % convention carry the REQUESTED internal padding in the name (e.g.
    % p16-16-8). Their content equals the clamped build to <= 2e-14 (measured
    % 2026-08-18 on all four spectral branches), so they remain valid: serve
    % them instead of rebuilding. New writes always use the effective name.
    if cfg.laplace && cfg.int_pad ~= int_pad_eff
        legacy = fullfile(out_dir, mkname(cfg.int_pad));
        if exist(legacy, 'file')
            S = load(legacy, 'TensorObj', 'Basis', 'Kernel');
            TensorObj = S.TensorObj; Basis = S.Basis; Kernel = S.Kernel;
            was_cached = true; filepath = legacy;
            fprintf('[cache hit ] %s  (legacy requested-padding name)\n', mkname(cfg.int_pad));
            return;
        end
    end

    fprintf('[build     ] %s\n', filename);
    kp = struct('zeta',cfg.zeta,'delta',cfg.delta,'omega',cfg.omega);
    if cfg.eta_hat   ~= 0, kp.eta_hat   = cfg.eta_hat;   kp.zeta_hat   = cfg.zeta_hat;   end
    if cfg.eta_hat_f ~= 0, kp.eta_hat_f = cfg.eta_hat_f; kp.zeta_hat_f = cfg.zeta_hat_f; end
    Kernel = ScatteringKernel('DSMC', kp);
    Basis  = SpectralBasis(cfg.K_max, cfg.L_max, cfg.I_max, Kernel.nu);
    TensorObj = GeneralCollisionTensor(Basis, Kernel);
    TensorObj.conserve_invariants = cfg.conserve;
    % Set explicitly (not only when true): cfg must control the path regardless
    % of the class default, else an untagged (algebraic-keyed) cache entry could
    % silently be built on the Laplace path.
    TensorObj.laplace_extended = cfg.laplace;
    TensorObj.laplace_Ns = cfg.laplace_Ns;
    TensorObj.frozen_no_etr = cfg.frozen_no_etr;
    TensorObj.loss_only = cfg.loss_only;
    tic;
    TensorObj.generate_R_tensor_sumfac(cfg.rad_pad, cfg.tan_pad, cfg.int_pad);
    fprintf('             built in %.1f s\n', toc);
    % Name-content assertion: the filename must record what the build used.
    if TensorObj.effective_padding.internal_effective ~= int_pad_eff
        error('build_or_load_dsmc_tensor:PaddingNameMismatch', ...
            ['Effective internal padding %d differs from the filename''s %d; ' ...
             'refusing to write a name-content mismatch (%s).'], ...
            TensorObj.effective_padding.internal_effective, int_pad_eff, filename);
    end
    save(filepath, 'TensorObj', 'Basis', 'Kernel', '-v7.3');
    was_cached = false;
end
