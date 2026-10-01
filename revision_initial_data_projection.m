%% REVISION ITEM 20 (part 1): projection error of the bi-Gaussian and discontinuous initial data
% Definitions from Wang & Cai (2019) as given in Hanke & Torrilhon (2023), eqs (45)-(46), in their scaling
% (Maxwellian (2pi)^{-3/2} e^{-c^2/2}); transformed to our thermal scaling v = c/sqrt(2), f_ours(v) = 2^{3/2} f_HT(sqrt(2) v).
% Both are axisymmetric about z (m = 0 only); the bi-Gaussian is even in z (even l only).
clear; clc; addpath('src', 'src/mex', 'src/SHL');
out_dir = '/Users/ekke/dev/simkinetic/papers/collision-kernel-factorization/revision/data';
bigauss = @(v) 2^1.5 / (2 * pi^1.5) * (exp(-2 * (v(:,1).^2 + v(:,2).^2 + (v(:,3) - sqrt(3)/2).^2)) + exp(-2 * (v(:,1).^2 + v(:,2).^2 + (v(:,3) + sqrt(3)/2).^2)));
cdis = 2^1.5 * 2^0.25 * (2 - sqrt(2)) / pi^1.5;
discont = @(v) (v(:,3) > 0) .* cdis .* exp(-sqrt(2) * sum(v.^2, 2)) + (v(:,3) <= 0) .* (cdis / 4) .* exp(-sum(v.^2, 2) / sqrt(2));
% quadrature: radial Gauss-Laguerre on v^2 (weight e^{-v^2}), polar split at cos(theta) = 0 (the discontinuity), azimuthal trapezoid
N_rad = 120; qr_rad = Gauss.generalized_laguerre(N_rad, 0.5); r = sqrt(qr_rad.x); w_r = qr_rad.w / 2;
N_pol = 60; qp = Gauss.legendre(N_pol, 0, 1); ct = [-flipud(qp.x); qp.x]; w_p = [flipud(qp.w); qp.w]; st = sqrt(1 - ct.^2);
N_azi = 1; ph = 0; w_a = 2*pi;                 % axisymmetric data: project on the m = 0 sub-basis only (exact azimuthal integral)
[ia, ip, iz] = ndgrid(1:N_rad, 1:2*N_pol, 1:N_azi);
v = [r(ia(:)) .* st(ip(:)) .* cos(ph(iz(:))), r(ia(:)) .* st(ip(:)) .* sin(ph(iz(:))), r(ia(:)) .* ct(ip(:))];
W = w_r(ia(:)) .* w_p(ip(:)) * w_a;             % includes e^{-v^2} v^2 dv dOmega
cases = {'bigaussian', bigauss; 'discontinuous', discont};
fid = fopen(fullfile(out_dir, 'initial_data_projection.csv'), 'w'); fprintf(fid, 'case,K,L,N,mass,momz,energy,relL2err,sigma_zz,q_z\n');
for ic = 1:2
    f = cases{ic, 2}(v); F = f .* exp(sum(v.^2, 2));            % F = f / e^{-v^2}
    normf2 = sum(W .* F.^2);                                      % ||f/w||^2 in L2(w), w = e^{-v^2}: equals sum_alpha c_alpha^2 for the full expansion
    mass = sum(W .* F); momz = sum(W .* F .* v(:,3)); energy = sum(W .* F .* sum(v.^2, 2));
    fprintf('%s: mass %.6f  mom_z %.2e  energy %.6f (3/2 expected)  ||f||^2 %.6f\n', cases{ic,1}, mass, momz, energy, normf2);
    for K = [2 4 6 8 10]
        for L = [4 8 12 16 20]
            B = SpectralBasis(K, L); q_map = zeros(B.N_Q, 1); q0 = (0:L).^2 + 1 + (0:L); q_map(q0) = 1:L+1;   % m = 0 modes
            Psi = B.evaluate_sub_basis(v, q_map); c = Psi' * (W .* F);                                     % c indexed [k*(L+1) + l + 1]
            err = sqrt(max(normf2 - sum(c.^2), 0) / normf2);    % relative L2 error of the truncated expansion (orthonormal basis)
            szz = c(3); qz = c((L + 1) + 2);                     % (k=0,l=2) and (k=1,l=1) m=0 coefficients (proportional to sigma_zz and q_z)
            fprintf(fid, '%s,%d,%d,%d,%.10f,%.3e,%.10f,%.4e,%.8e,%.8e\n', cases{ic,1}, K, L, B.N_terms, mass, momz, energy, err, szz, qz);
            fprintf('  K=%2d L=%2d N=%5d  rel L2 err %.3e\n', K, L, B.N_terms, err);
        end
    end
end
fclose(fid); fprintf('written\n');
