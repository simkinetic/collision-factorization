%% REVISION ITEM 20 (part 2): relaxation of the bi-Gaussian and discontinuous initial data, hard spheres.
% Factorized operator evaluated with the angular-first kernel inside an adaptive RK integrator (ode45).
% Time is reported in units of the first-order viscous relaxation time, tau = |J_ss| t, with J_ss the
% (k=0, l=2) diagonal of the equilibrium Jacobian (basis independent: the first Chapman-Cowling bracket).
% Output per (K, L): tau, sigma_zz = int (v_z^2 - v^2/3) f dv, q_z = 1/2 int v_z v^2 f dv, the fourth
% scalar moment m4 = int v^4 f dv, and the drift of mass, momentum and energy.
function revision_initial_data_dynamics(KL_list)
addpath('src', 'src/mex', 'src/SHL');
if nargin < 1, KL_list = {[4 8], [4 12]}; end
out_dir = fullfile('revision', 'data'); if ~exist(out_dir, 'dir'), mkdir(out_dir); end
bigauss = @(v) 2^1.5 / (2 * pi^1.5) * (exp(-2 * (v(:,1).^2 + v(:,2).^2 + (v(:,3) - sqrt(3)/2).^2)) + exp(-2 * (v(:,1).^2 + v(:,2).^2 + (v(:,3) + sqrt(3)/2).^2)));
cdis = 2^1.5 * 2^0.25 * (2 - sqrt(2)) / pi^1.5;
discont = @(v) (v(:,3) > 0) .* cdis .* exp(-sqrt(2) * sum(v.^2, 2)) + (v(:,3) <= 0) .* (cdis / 4) .* exp(-sum(v.^2, 2) / sqrt(2));
cases = {'bigaussian', bigauss; 'discontinuous', discont};
% quadrature (m = 0 sub-basis; polar rule split at cos(theta) = 0)
N_rad = 120; qr_rad = Gauss.generalized_laguerre(N_rad, 0.5); r = sqrt(qr_rad.x); w_r = qr_rad.w / 2;
N_pol = 60; qp = Gauss.legendre(N_pol, 0, 1); ct = [-flipud(qp.x); qp.x]; w_p = [flipud(qp.w); qp.w]; st = sqrt(1 - ct.^2);
[ia, ip] = ndgrid(1:N_rad, 1:2*N_pol);
v = [r(ia(:)) .* st(ip(:)), zeros(numel(ia), 1), r(ia(:)) .* ct(ip(:))]; W = w_r(ia(:)) .* w_p(ip(:)) * 2*pi;
v2 = sum(v.^2, 2);
tau_end = 6.0;
for i = 1:numel(KL_list)
    K = KL_list{i}(1); L = KL_list{i}(2);
    fn = sprintf('src/precalc/collisiontensor_k%d_l%d_gamma1.00.mat', K, L);
    if ~exist(fn, 'file'), fprintf('missing %s\n', fn); continue; end
    d = load(fn, 'TensorObj', 'Basis'); T = d.TensorObj; B = d.Basis; N_Q = B.N_Q; K_len = K + 1; N = N_Q * K_len;
    [~, si] = sortrows([T.ic_map, T.gaunt_labels(:, 1)]); lb = T.gaunt_labels(si, :); vs = T.gaunt_vals(si); ic = T.ic_map(si); R = T.R_tensor;
    rhs = @(t, c) rhs_eval(c, N_Q, K_len, lb, vs, ic, R);
    % m = 0 sub-basis and moment vectors
    q_map = zeros(N_Q, 1); q0 = (0:L).^2 + 1 + (0:L); q_map(q0) = 1:L+1; Psi = B.evaluate_sub_basis(v, q_map);
    sub2full = reshape(((0:K)' * N_Q + q0)', [], 1);                   % sub index k*(L+1)+l+1 (l fastest) -> global index k*N_Q + q0(l)
    mom = @(g) Psi' * (W .* g);                                          % moment vector: int g f dv = mom(g)' * c_sub
    m_mass = mom(ones(size(v2))); m_momz = mom(v(:,3)); m_en = mom(v2); m_szz = mom(v(:,3).^2 - v2/3); m_qz = mom(0.5 * v(:,3) .* v2); m_4 = mom(v2.^2);
    % viscous relaxation rate: (k=0, l=2) diagonal of the Jacobian at equilibrium, J = C(.,eq,.) + C(.,.,eq)
    c000 = Psi(:, 1)' * (W .* (pi^-1.5 * ones(size(v2))));             % (k=0,l=0) coefficient of the unit-mass Maxwellian (mass of both data sets is 1)
    e1 = zeros(N, 1); e1(1) = c000; es = zeros(N, 1); es(q0(3)) = 1;     % equilibrium of unit mass, and the stress (k=0,l=2,m=0) unit vector
    Q_es_e1 = rhs(0, e1 + es) - rhs(0, e1) - rhs(0, es);                 % bilinear cross term C(es,e1) + C(e1,es) = J(e1) es
    Jss = Q_es_e1(q0(3)); fprintf('K=%d L=%d: c000 = %.6f, J_ss = %.6f\n', K, L, c000, Jss);
    for ic_ = 1:2
        f = cases{ic_, 2}(v); F = f .* exp(v2); c_sub = Psi' * (W .* F);
        c0 = zeros(N, 1); c0(sub2full) = c_sub;
        opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-12);
        tic; [t, c] = ode45(rhs, linspace(0, tau_end / abs(Jss), 201), c0, opts); fprintf('  %s: %d steps, %.1f s\n', cases{ic_,1}, numel(t), toc);
        cs = c(:, sub2full);
        fid = fopen(fullfile(out_dir, sprintf('relax_%s_k%d_l%d.csv', cases{ic_,1}, K, L)), 'w');
        fprintf(fid, 'tau,sigma_zz,q_z,m4,mass,momz,energy\n');
        fprintf(fid, '%.6f,%.10e,%.10e,%.10e,%.12e,%.3e,%.12e\n', [abs(Jss) * t, cs * m_szz, cs * m_qz, cs * m_4, cs * m_mass, cs * m_momz, cs * m_en]');
        fclose(fid);
        fprintf('    sigma_zz(0)=%.6e q_z(0)=%.6e m4(0)=%.6f | end: sigma=%.2e q=%.2e | mass drift %.1e energy drift %.1e\n', ...
            cs(1,:) * m_szz, cs(1,:) * m_qz, cs(1,:) * m_4, cs(end,:) * m_szz, cs(end,:) * m_qz, max(abs(cs * m_mass - cs(1,:) * m_mass)), max(abs(cs * m_en - cs(1,:) * m_en)));
    end
end
end

function q = rhs_eval(c, N_Q, K_len, lb, vs, ic, R)
% the angular-first kernel accumulates into its first argument in place (no return value)
Q = zeros(N_Q, K_len);
angular_first_collision_kernel_mex(Q, reshape(c, N_Q, K_len), lb, vs, ic, R, N_Q, K_len);
q = Q(:);
end
