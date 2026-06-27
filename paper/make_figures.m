% make_figures.m  -- publication figures for the DSMC / extended-model report.
%   fig_frozen_pr_bratio.pdf : b=Ps0/Pq0 vs zeta -- spectral MEX vs Monte-Carlo vs eq-42.
%   fig_convergence.pdf      : spectral resolution of (I/E)^zhat -- frozen & non-frozen
%                              channels, pointwise (algebraic) vs auxiliary-Laplace (spectral).
%   fig_reachability.pdf     : transport-coefficient plane (nu/mu,Pr); extended re-fit reaches
%                              the experimental target the base model cannot.
% NOTE: the non-frozen panel of fig_convergence uses light padding / modest Ns for speed
% (validation, not paper-quality precision); raise spatN/NsN/padsN for final figures.
proj = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(proj,'src')); addpath(genpath(fullfile(proj,'src','SHL'))); addpath(fullfile(proj,'src','mex'));
outdir = fullfile(proj,'paper','figures');
K_max=2; L_max=2; I_max=2; N_I=I_max+1; N_Q=(L_max+1)^2;
idx=@(k,i,l,m)(k*N_I+i)*N_Q+(l^2+l+m)+1;
clr_alg=[0.85 0.33 0.10]; clr_spc=[0 0.45 0.74];

% ---------------------------------------------------------------- FIG 1
delta=2.01; nu=delta/2-1;
zg = 0:0.1:0.9; b_mex = nan(size(zg));
for j=1:numel(zg)
    Basis=SpectralBasis(K_max,L_max,I_max,nu);
    Kd=ScatteringKernel('DSMC',struct('zeta',zg(j),'delta',delta,'omega',0.0));
    T=GeneralCollisionTensor(Basis,Kd); T.generate_R_tensor_sumfac(16,16,4); C=T.assemble_full_tensor();
    J=squeeze(C(:,:,1))+squeeze(C(:,1,:));
    Pq0=-J(idx(1,0,1,0),idx(1,0,1,0)); Ps0=-J(idx(0,1,1,0),idx(0,1,1,0));
    b_mex(j)=Ps0/Pq0;
end
eq42b=@(d,z) 3*(2*d+z+7)./((2+0.8*z).*(2*d+2*z+7));
b_eq42 = eq42b(delta, zg);
zmc=[0 0.25 0.53 0.8]; b_mc=nan(size(zmc)); N=4e6; rng(7);
for j=1:numel(zmc)
    z=zmc(j); v=randn(N,3); vs=randn(N,3);
    u=v-vs; um=sqrt(sum(u.^2,2)); w=um.^z; V=0.5*(v+vs);
    gdir=randn(N,3); gdir=gdir./vecnorm(gdir,2,2); half=0.5*um.*gdir; vp=V+half; vsp=V-half;
    phiq=@(x)(sum(x.^2,2)-5).*x(:,3);
    brS=0.5*mean((v(:,3)-vp(:,3)).^2.*w); normS=1;
    dq=phiq(v)+phiq(vs)-phiq(vp)-phiq(vsp); brQ=0.25*mean(dq.^2.*w); normQ=mean(phiq(v).^2);
    b_mc(j)=(brS/normS)/(brQ/normQ);
end
f=figure('Visible','off','Units','inches','Position',[0 0 5.4 4.0]);
plot(zg,b_mex,'-','LineWidth',2,'Color',clr_spc); hold on;
plot(zg,b_eq42,'--','LineWidth',2,'Color',clr_alg);
plot(zmc,b_mc,'o','MarkerSize',8,'LineWidth',1.5,'Color',[0 0 0],'MarkerFaceColor',[0.4 0.8 0.4]);
grid on; xlabel('\zeta = 2(1-s_{visc})'); ylabel('b = P_s^{(0)}/P_q^{(0)}');
legend({'Spectral operator (this work)','Paper analytic, eq. (42)','Monte-Carlo bracket'}, ...
       'Location','southwest','FontSize',9);
title('Frozen internal-heat-flux ratio'); set(gca,'FontSize',11);
exportgraphics(f, fullfile(outdir,'fig_frozen_pr_bratio.pdf'),'ContentType','vector'); close(f);
fprintf('FIG1 done.\n');

% ---------------------------------------------------------------- FIG 2
% Spectral convergence of the auxiliary s-integral (the published implementation): error of Ps0
% vs the Gauss-Jacobi node count Ns, at FIXED internal/spatial padding so those errors are
% common-mode and cancel, isolating the s-integral convergence. Both channels converge
% geometrically to machine precision. (We no longer plot the abandoned pointwise/algebraic scheme.)
zeta=0.533; eh=0.5; zh=0.965;
clr_fr=[0 0.45 0.74]; clr_nf=[0.85 0.33 0.10];
Ns_list=[6 10 16 24 32]; Ns_ref=44;
% base regression (eta=0): aux path vs legacy
Jbl=buildJloc(K_max,L_max,I_max,nu,0.0,0,0,0,0,[6 6 6],false);
Jbs=buildJloc(K_max,L_max,I_max,nu,0.0,0,0,0,0,[6 6 6],true,16);
fprintf('FIG2 base-reg frozen: max|aux-legacy|=%.2e (rel %.2e)\n', ...
    max(abs(Jbs(:)-Jbl(:))), max(abs(Jbs(:)-Jbl(:)))/max(abs(Jbl(:))));
% --- frozen channel (omega=0), fixed internal pad, sweep Ns ---
spatF=8; ipF=6;
Jr=buildJloc(K_max,L_max,I_max,nu,0.0,0,0,eh,zh,[spatF spatF ipF],true,Ns_ref);
pF=-Jr(idx(0,1,1,0),idx(0,1,1,0)); errF=nan(size(Ns_list));
for j=1:numel(Ns_list)
    J=buildJloc(K_max,L_max,I_max,nu,0.0,0,0,eh,zh,[spatF spatF ipF],true,Ns_list(j));
    errF(j)=abs(-J(idx(0,1,1,0),idx(0,1,1,0))-pF)+1e-16;
end
% --- non-frozen channel (omega=1), fixed internal pad, sweep Ns ---
spatN=6; ipN=6;
Jr=buildJloc(K_max,L_max,I_max,nu,1.0,eh,zh,0,0,[spatN spatN ipN],true,Ns_ref);
pN=-Jr(idx(0,1,1,0),idx(0,1,1,0)); errN=nan(size(Ns_list));
for j=1:numel(Ns_list)
    J=buildJloc(K_max,L_max,I_max,nu,1.0,eh,zh,0,0,[spatN spatN ipN],true,Ns_list(j));
    errN(j)=abs(-J(idx(0,1,1,0),idx(0,1,1,0))-pN)+1e-16;
end
fprintf('FIG2 s-axis frozen=%s\n        nonfrozen=%s\n', mat2str(errF,3), mat2str(errN,3));
save(fullfile(outdir,'saxis_data.mat'),'Ns_list','errF','errN');
f=figure('Visible','off','Units','inches','Position',[0 0 5.6 4.2]);
semilogy(Ns_list,errF,'-o','LineWidth',2,'Color',clr_fr,'MarkerFaceColor',clr_fr); hold on;
semilogy(Ns_list,errN,'-s','LineWidth',2,'Color',clr_nf,'MarkerFaceColor',clr_nf);
grid on; xlabel('auxiliary-integral nodes N_s'); ylabel('|P_s^{(0)}(N_s) - P_s^{(0)}_{ref}|');
title('Spectral convergence of the auxiliary s-integral'); set(gca,'FontSize',10);
legend({'frozen channel','non-frozen channel (9D)'},'Location','northeast','FontSize',9);
exportgraphics(f, fullfile(outdir,'fig_convergence.pdf'),'ContentType','vector'); close(f);
fprintf('FIG2 done.\n');

% ---------------------------------------------------------------- FIG 3
% Normalized reachability: each axis is a transport coefficient divided by its experimental
% value, so every gas's target collapses to (1,1) and a true circle (a relative-tolerance band)
% is meaningful regardless of the very different nu/mu scales (H2 ~ 30 vs N2/CO ~ 0.6). Our
% re-fit (filled dots) lands inside the target circle; the literature Table-4 parameters (x)
% sit outside, with a line tracing the re-fit displacement. Colours distinguish the gases.
gn={'N_2','CO','H_2'};
Pr_meas=[0.717 0.743 0.686];   num_meas=[0.73 0.55 30.0];     % experiment
Pr_e3  =[0.7188 0.7413 0.7116];num_e3  =[0.8532 0.6343 36.10];% literature Table-4 params
Pr_e4  =[0.7170 0.7430 0.6860];num_e4  =[0.7300 0.5500 30.00];% re-fit (this work)
cols=[0 0.45 0.74; 0.30 0.69 0.29; 0.85 0.33 0.10];
rx3=num_e3./num_meas; ry3=Pr_e3./Pr_meas;
rx4=num_e4./num_meas; ry4=Pr_e4./Pr_meas;
f=figure('Visible','off','Units','inches','Position',[0 0 5.8 4.8]);
hold on; box on;
th=linspace(0,2*pi,200); rr=0.05;
fill(1+rr*cos(th),1+rr*sin(th),[0.88 0.92 0.97],'EdgeColor',[0.2 0.2 0.2],'LineStyle','--','LineWidth',1.1);
plot(1,1,'k+','MarkerSize',9,'LineWidth',1);
for j=1:3
    plot([rx3(j) rx4(j)],[ry3(j) ry4(j)],'-','Color',cols(j,:),'LineWidth',1.0);
    plot(rx3(j),ry3(j),'x','MarkerSize',11,'LineWidth',2.2,'Color',cols(j,:));
    plot(rx4(j),ry4(j),'o','MarkerSize',8,'LineWidth',1.2,'Color',cols(j,:),'MarkerFaceColor',cols(j,:));
    text(rx3(j),ry3(j)+0.006, gn{j},'Color',cols(j,:),'FontWeight','bold','FontSize',10,'HorizontalAlignment','center');
end
h1=plot(nan,nan,'ko','MarkerFaceColor','k','MarkerSize',8);
h2=plot(nan,nan,'kx','MarkerSize',11,'LineWidth',2.2);
h3=fill(nan,nan,[0.88 0.92 0.97],'EdgeColor',[0.2 0.2 0.2],'LineStyle','--');
legend([h1 h2 h3],{'re-fit (this work)','literature Table-4 params','experimental target (\pm5%)'}, ...
       'Location','northwest','FontSize',9);
xlabel('(\nu/\mu) / (\nu/\mu)_{exp}'); ylabel('Pr / Pr_{exp}');
title('Reachability of the experimental transport coefficients');
axis equal; xlim([0.90 1.26]); ylim([0.95 1.07]); grid on; set(gca,'FontSize',10);
exportgraphics(f, fullfile(outdir,'fig_reachability.pdf'),'ContentType','vector'); close(f);
fprintf('FIG3 done.\n');
fprintf('ALL FIGURES WRITTEN to %s\n', outdir);

% ---- local helpers ----
function Pr = prandtl(J, delta, rho, N_I, N_Q, idx)
    P_sigma=-J(idx(0,0,2,0),idx(0,0,2,0));
    iq=idx(1,0,1,0); is=idx(0,1,1,0); JL1=J([iq is],[iq is]);
    Pq0=-JL1(1,1); Ps0=-JL1(2,2); Pq1=-JL1(1,2)*rho; Ps1=-JL1(2,1)/rho;
    Pr=(5+delta)/2*2*(Pq0*Ps0-Pq1*Ps1)/(P_sigma*(5*(Ps0-Ps1)+delta*(Pq0-Pq1)));
end

function J = buildJloc(K_max,L_max,I_max,nu, omega, eh,zh, ehf,zhf, pad, lap, Ns)
    Basis = SpectralBasis(K_max,L_max,I_max,nu);
    Kd = ScatteringKernel('DSMC', struct('zeta',0.533,'delta',2*(nu+1),'omega',omega, ...
        'eta_hat',eh,'zeta_hat',zh,'eta_hat_f',ehf,'zeta_hat_f',zhf));
    T = GeneralCollisionTensor(Basis, Kd);
    if nargin>=11 && lap, T.laplace_extended=true; end
    if nargin>=12 && ~isempty(Ns), T.laplace_Ns=Ns; end
    T.generate_R_tensor_sumfac(pad(1), pad(2), pad(3));
    C = T.assemble_full_tensor();
    J = squeeze(C(:,:,1)) + squeeze(C(:,1,:));
end
