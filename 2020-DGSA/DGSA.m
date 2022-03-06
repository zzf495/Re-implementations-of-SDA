function [acc,acc_ite] = DGSA(Xs,Ys,Xt,Yt,options)
%% Implementation of DGSA
%%% Authors:        Zhao et al.
%%% Paper:          2020-Discriminant Geometrical and Statistical Alignment With Density Peaks for Domain Adaptation
%% input
%%% T           iteration times
%%% gamma       The weight of Density Estimation
%%% rs          The rate that filter the source samples
%%% rt          The rate that filter the target samples
%%% nu          The weight of regularization
%%% lambda      The weight of MMD
%%% kernel_type: The kernel mapping, e.g., 'primal' (skip kernel mapping),'linear', 'rbf', and 'sam'
%%% kernel_param: The weight of kernel (sigma in kernel project)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
Xs = normr(Xs')';
Xt = normr(Xt')';
acc_ite=[];
options=defaultOptions(options,...
                            'T',10,...
                            'dim',30,...
                            'rs',0.2,...
                            'rt',0.2,...
                            'nu',0.1,...
                            'lambda',0.1,...
                            'gamma',1,...
                             'kernel_type','rbf',...
                            'kernel_param',1);
rs=options.rs;
rt=options.rt;
nu=options.nu;
lambda=options.lambda;
gamma=options.gamma;
dim=options.dim;
T=options.T;
[~,ns,nt,n,~,C] = datasetMsg(Xs,Ys,Xt);
% predict
Ytpesudo=classifyKNN(Xs,Ys,Xt,1);
% select high-confidence samples
rho_s=DGSA_estimate_density(Xs,Ys,gamma);
rho_t=DGSA_estimate_density(Xt,Ytpesudo,gamma);
% select Xs
landmark_s=sort(rho_s,'descend');
landmark_s=landmark_s(floor(ns*rs));
Xs2=Xs(:,rho_s>=landmark_s);
Ys2=Ys(rho_s>=landmark_s);
% select Xt
landmark_t=sort(rho_t,'descend');
landmark_t=landmark_t(floor(nt*rt));
Xt2=Xt(:,rho_t>=landmark_t);
Yt2=Ytpesudo(rho_t>=landmark_t);
% compute Sb
Sb=betweenScatter(Xs2,Ys2);
[Ps,~]=eigs(Sb,dim,'lm');
Sb=betweenScatter(Xt2,Yt2);
[Pt,~]=eigs(Sb,dim,'lm');
% project to GFK
G = GFK_core([Ps,null(Ps')], Pt(:,1:dim));
sq_G = real(G^(0.5));
Zs = (sq_G * Xs);
Zt = (sq_G * Xt);
Ytpesudo=classifyKNN(Zs,Ys,Zt,1);
Z=[Zs,Zt];
Z=normr(Z')';
% Z=Z';Z=zscore(Z,1);Z=Z';
K=kernelProject(options.kernel_type,Z,[],options.kernel_param);
% compute M0
M0=marginalDistribution(Zs,Zt,C);
hotY=hotmatrix([Ys;zeros(nt,1)],C);
E=blkdiag(eye(ns),zeros(nt));
for i=1:T
    % compute M according to (14)
    M=M0+conditionalDistribution(Zs,Zt,Ys,Ytpesudo,C);
    M=M/norm(M,'fro');
    % compute P according to (16)
    left=(E+lambda*M)*K+nu*eye(n);
    right=E*hotY;
    P=left\right;
    newZ=P'*K;
%     newZs=newZ(:,1:ns);
    newZt=newZ(:,ns+1:end);
    [~,Ytpesudo]=max(newZt,[],1);Ytpesudo=Ytpesudo';
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f\n',i,acc);
    end
end
end



function G = GFK_core(Q,Pt)
    % Input: Q = [Ps, null(Ps')], where Ps is the source subspace, column-wise orthonormal
    %        Pt: target subsapce, column-wise orthonormal, D-by-d, d < 0.5*D
    % Output: G = \int_{0}^1 \Phi(t)\Phi(t)' dt

    % ref: Geodesic Flow Kernel for Unsupervised Domain Adaptation.  
    % B. Gong, Y. Shi, F. Sha, and K. Grauman.  
    % Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

    % Contact: Boqing Gong (boqinggo@usc.edu)

    N = size(Q,2); % 
    dim = size(Pt,2);

    % compute the principal angles
    QPt = Q' * Pt;
    [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
    V2 = -V2;
    theta = real(acos(diag(Gam))); % theta is real in theory. Imaginary part is due to the computation issue.

    % compute the geodesic flow kernel
    eps = 1e-20;
    B1 = 0.5.*diag(1+sin(2*theta)./2./max(theta,eps));
    B2 = 0.5.*diag((-1+cos(2*theta))./2./max(theta,eps));
    B3 = B2;
    B4 = 0.5.*diag(1-sin(2*theta)./2./max(theta,eps));
    G = Q * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2] ...
        * [B1,B2,zeros(dim,N-2*dim);B3,B4,zeros(dim,N-2*dim);zeros(N-2*dim,N)]...
        * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2]' * Q';
end
