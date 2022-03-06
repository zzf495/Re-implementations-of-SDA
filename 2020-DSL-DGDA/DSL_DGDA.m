function [acc,acc_ite] = DSL_DGDA(Xs,Ys,Xt,Yt,options)
%% Implementation of DSL-DGDA
%%% Authors:        Elahe Gholenji et al.
%%% Paper:          2020-Joint discriminative subspace and distribution adaptation for unsupervised domain adaptation
%% input
%%% T           The iteration
%%% dim         The dimension
%%% sigma       The weight of regularization
%%% lambda      The weight of MMD
%%% gamma       The weight of manifold regularization
%%% k:       k-neighbors of graph
%%% kernel_type: The kernel mapping, e.g., 'primal' (skip kernel mapping),'linear', 'rbf', and 'sam'
%%% kernel_param: The weight of kernel (sigma in kernel project)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
% Xs = normr(Xs')';
% Xt = normr(Xt')';
acc_ite=[];
options=defaultOptions(options,...
                            'T',10,...
                            'dim',30,...
                            'sigma',0.1,...
                            'lambda',0.1,...
                            'gamma',1,...
                            'k',7,...
                             'kernel_type','rbf',...
                            'kernel_param',1);

T=options.T;
lambda=options.lambda;
gamma=options.gamma;
sigma=options.sigma;
[~,ns,nt,n,~,C] = datasetMsg(Xs,Ys,Xt);
% SA map
[Zs,Zt]=DGDA_SA_MAP(Xs,Ys,Xt,options);
Ytpesudo=classifyKNN(Zs,Ys,Zt,1);
% predict
acc=getAcc(Ytpesudo,Yt);
if isfield(options,'display')
    fprintf('[init] acc:%.4f\n',acc);
end
Z=[Zs,Zt];
Z=normr(Z')';
% Z=Z';Z=zscore(Z,1);Z=Z';
K=kernelProject(options.kernel_type,Z,[],options.kernel_param);
% compute M0
M0=marginalDistribution(Zs,Zt,C);
hotY=hotmatrix([Ys;zeros(nt,1)],C);
E=blkdiag(eye(ns),zeros(nt));
% compute L
manifold.k = options.k;
manifold.Metric = 'Cosine';
manifold.WeightMode = 'Cosine';
manifold.NeighborMode = 'KNN';
[Ls,~,Ws] = computeL(Zs,manifold);
[Lt,~,Wt] = computeL(Zt,manifold);
% disimilar
opt.k=options.k;
opt.gamma=1;
% [Lsb,~,~]=DGDA_dissimilarGraph(Zs,Ys,opt);
% update Ls_b
Wsb=ones(ns,ns);
for i=1:C
    idx=Ys==i;
    Wsb(idx,idx)=0;
end
Wsb=Wsb.*Ws;
Dsb = diag(sparse(sqrt(1./sum(Wsb))));
Lsb= speye(ns)-Dsb*Wsb*Dsb;

L=blkdiag(Ls-Lsb,Lt);
% L=blkdiag(Ls-Lsb,Lt);
L=L/norm(L,'fro');
for i=1:T
    % compute M according to (14)
    M=M0+conditionalDistribution(Zs,Zt,Ys,Ytpesudo,C);
    M=M/norm(M,'fro');
    % compute P according to (16)
    left=(E+lambda*M+gamma*L)*K+sigma*eye(n);
    right=E*hotY;
    P=left\right;
    newZ=P'*K;
    newZs=newZ(:,1:ns);
    newZt=newZ(:,ns+1:end);
    [~,Ytpesudo]=max(newZt,[],1);Ytpesudo=Ytpesudo';
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f\n',i,acc);
    end
    
end
end