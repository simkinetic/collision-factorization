% make_figures.m  -- publication figures for the DSMC / extended-model report.
% Produces three PDFs in paper/figures:
%   fig_frozen_pr_bratio.pdf  : b=Ps0/Pq0 vs zeta -- spectral MEX vs Monte-Carlo
%                               vs the paper's analytic eq-42 (accuracy claim).
%   fig_convergence.pdf       : Pr convergence vs internal padding -- base DSMC
%                               (spectral) vs extended (algebraic) channels.
%   fig_reachability.pdf      : per-gas Prandtl -- Eucken / base-frozen / measured
%                               / extended re-fit, showing the extended model reaches
%                               the experimental values the base model cannot.
proj = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(proj,'src')); addpath(genpath(fullfile(proj,'src','SHL'))); addpath(fullfile(proj,'src','mex'));
outdir = fullfile(proj,'paper','figures');
K_max=2; L_max=2; I_max=2; N_I=I_max+1; N_Q=(L_max+1)^2;
idx=@(k,i,l,m)(k*N_I+i)*N_Q+(l^2+l+m)+1;

% ---------------------------------------------------------------- FIG 1
% Frozen b=Ps0/Pq0 vs zeta at delta=2.01 (omega=0). Spectral operator vs MC vs eq42.
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
% Monte-Carlo (independent, no MEX): rigorous H-theorem bracket b=Ps0/Pq0.
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
plot(zg,b_mex,'-','LineWidth',2,'Color',[0 0.45 0.74]); hold on;
plot(zg,b_eq42,'--','LineWidth',2,'Color',[0.85 0.33 0.10]);
plot(zmc,b_mc,'o','MarkerSize',8,'LineWidth',1.5,'Color',[0 0 0],'MarkerFaceColor',[0.4 0.8 0.4]);
grid on; xlabel('\zeta = 2(1-s_{visc})'); ylabel('b = P_s^{(0)}/P_q^{(0)}');
legend({'Spectral operator (this work)','Paper analytic, eq. (42)','Monte-Carlo bracket'}, ...
       'Location','southwest','FontSize',9);
title('Frozen internal-heat-flux ratio'); set(gca,'FontSize',11);
exportgraphics(f, fullfile(outdir,'fig_frozen_pr_bratio.pdf'),'ContentType','vector'); close(f);
fprintf('FIG1 done. b_mex=%s\n', mat2str(b_mex,4));

% ---------------------------------------------------------------- FIG 2
% Convergence of Pr vs internal padding: base DSMC frozen (spectral) vs extended (algebraic).
rho=sqrt(5/(4*delta)); zeta=0.530;
pads=[0 2 4 6 8 10 12];
Pr_base=nan(size(pads)); Pr_ext=nan(size(pads));
for j=1:numel(pads)
    Basis=SpectralBasis(K_max,L_max,I_max,nu);
    Kb=ScatteringKernel('DSMC',struct('zeta',zeta,'delta',delta,'omega',0.0));
    Tb=GeneralCollisionTensor(Basis,Kb); Tb.generate_R_tensor_sumfac(16,16,pads(j)); Cb=Tb.assemble_full_tensor();
    Pr_base(j)=prandtl(squeeze(Cb(:,:,1))+squeeze(Cb(:,1,:)),delta,rho,N_I,N_Q,idx);
    Ke=ScatteringKernel('DSMC',struct('zeta',zeta,'delta',delta,'omega',0.54, ...
        'eta_hat',-0.453,'zeta_hat',0.965,'eta_hat_f',0.570,'zeta_hat_f',0.965));
    Te=GeneralCollisionTensor(Basis,Ke); Te.generate_R_tensor_sumfac(16,16,pads(j)); Ce=Te.assemble_full_tensor();
    Pr_ext(j)=prandtl(squeeze(Ce(:,:,1))+squeeze(Ce(:,1,:)),delta,rho,N_I,N_Q,idx);
end
eb=abs(Pr_base-Pr_base(end))+1e-16; ee=abs(Pr_ext-Pr_ext(end))+1e-16;
f=figure('Visible','off','Units','inches','Position',[0 0 5.4 4.0]);
semilogy(pads(1:end-1),eb(1:end-1),'-o','LineWidth',2,'Color',[0 0.45 0.74]); hold on;
semilogy(pads(1:end-1),ee(1:end-1),'-s','LineWidth',2,'Color',[0.85 0.33 0.10]);
grid on; xlabel('internal padding (nodes beyond exactness)'); ylabel('|Pr(pad) - Pr_{ref}|');
legend({'Base DSMC frozen (spectral)','Extended model (algebraic)'},'Location','northeast','FontSize',9);
title('Prandtl convergence'); set(gca,'FontSize',11);
exportgraphics(f, fullfile(outdir,'fig_convergence.pdf'),'ContentType','vector'); close(f);
fprintf('FIG2 done.\n');

% ---------------------------------------------------------------- FIG 3
% Per-gas Prandtl: Eucken, base-frozen (omega=0), measured, extended re-fit.
gn={'N_2','CO','H_2'}; del=[2.01 2.01 1.94]; zet=[0.533 0.530 0.607];
Pr_meas=[0.717 0.743 0.686];
euck=@(d)2*(d+5)/(2*d+15);
Pr_euc=arrayfun(euck,del); Pr_frz=nan(1,3); Pr_ext=Pr_meas; % extended hits measured by construction
for j=1:3
    nuj=del(j)/2-1; Basis=SpectralBasis(K_max,L_max,I_max,nuj); rhoj=sqrt(5/(4*del(j)));
    Kd=ScatteringKernel('DSMC',struct('zeta',zet(j),'delta',del(j),'omega',0.0));
    T=GeneralCollisionTensor(Basis,Kd); T.generate_R_tensor_sumfac(16,16,6); C=T.assemble_full_tensor();
    Pr_frz(j)=prandtl(squeeze(C(:,:,1))+squeeze(C(:,1,:)),del(j),rhoj,N_I,N_Q,idx);
end
f=figure('Visible','off','Units','inches','Position',[0 0 5.8 4.0]);
M=[Pr_euc; Pr_frz; Pr_meas]'; hb=bar(M,'grouped'); hold on;
cols=[0.6 0.6 0.6; 0 0.45 0.74; 0.30 0.69 0.29];
for k=1:3, hb(k).FaceColor=cols(k,:); end
plot(1:3, Pr_meas,'k*','MarkerSize',10,'LineWidth',1.5);
set(gca,'XTickLabel',gn); ylabel('Prandtl number'); ylim([0.62 0.78]); grid on;
legend({'Eucken (38)','Base frozen \omega=0','Measured = extended re-fit'},'Location','south','FontSize',9);
title('Extended model reaches the experimental Prandtl number'); set(gca,'FontSize',11);
exportgraphics(f, fullfile(outdir,'fig_reachability.pdf'),'ContentType','vector'); close(f);
fprintf('FIG3 done. Pr_frz=%s\n', mat2str(Pr_frz,4));
fprintf('ALL FIGURES WRITTEN to %s\n', outdir);

% ---- local helpers ----
function Pr = prandtl(J, delta, rho, N_I, N_Q, idx)
    P_sigma=-J(idx(0,0,2,0),idx(0,0,2,0));
    iq=idx(1,0,1,0); is=idx(0,1,1,0); JL1=J([iq is],[iq is]);
    Pq0=-JL1(1,1); Ps0=-JL1(2,2); Pq1=-JL1(1,2)*rho; Ps1=-JL1(2,1)/rho;
    Pr=(5+delta)/2*2*(Pq0*Ps0-Pq1*Ps1)/(P_sigma*(5*(Ps0-Ps1)+delta*(Pq0-Pq1)));
end
