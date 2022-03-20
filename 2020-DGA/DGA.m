function [acc,acc_ite] = DGA(Xs,Ys,Xt,Yt,options)
%% Implementation of DGA
%%% Authors:        Lingkun Luo et al.
%%% Paper:          2020-Discriminative and geometry-aware unsupervised domain adaptation
%% input
%%% T       iteration times
%%% dim     dimension
%%% lambda   regularization parameter
%%% alpha    manifold parameter , alpha=1/(1+u),(default 0.99)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
% Xs = normr(Xs')';
% Xt = normr(Xt')';
acc_ite=[];
options=defaultOptions(options,...
                            'T',10,...
                            'dim',30,...
                            'lambda',1,...
                            'alpha',0.99);
alpha=options.alpha;
lambda=options.lambda;
T=options.T;
dim=options.dim;
[X,ns,~,n,m,C] = datasetMsg(Xs,Ys,Xt);
% kernel project
% init parameter

% W=lapgraph(X,manifold);
H=centeringMatrix(n);
% init Ytpesudo
M0=marginalDistribution(Xs,Xt,C);
[A,~]=eigs(X*M0*X'+lambda*eye(m),X*H*X',dim,'sm');
Z=A'*X;
Zs=Z(:,1:ns);
Zt=Z(:,ns+1:end);
Ytpesudo=classifyKNN(Zs,Ys,Zt,1);
acc=getAcc(Ytpesudo,Yt);
if isfield(options,'display')
   fprintf('[%2d] (init) acc:%.4f\n',0,acc);
end

for i=1:T
    % compute Mcyd
    N=conditionalDistribution(Xs,Xt,Ys,Ytpesudo,C);
    Mcr1=repulsiveDistribution(Xs,Xt,Ys,Ytpesudo,C,1);%Mcr1=Mcr1./norm(Mcr1,'fro');
    Mcr2=repulsiveDistribution(Xs,Xt,Ys,Ytpesudo,C,2);%Mcr2=Mcr2./norm(Mcr2,'fro');
    M1=M0+N; %M1=M1./norm(M1,'fro');
    Mcyd=M1-(Mcr1+Mcr2);  %M0+N-Mcr;
    Mcyd=Mcyd/norm(Mcyd,'fro');
    % compute A according to (16)
    [A,~]=eigs(X*Mcyd*X'+lambda*eye(m),X*H*X',dim,'sm');
    % compute Y^* according to (18)
    Z=A'*X;
    Z=L2Norm(Z')';
    Zs=Z(:,1:ns);
    Zt=Z(:,ns+1:end);
    Ytpesudo=classifyKNN(Zs,Ys,Zt,1);
    manifold.Metric = 'Euclidean';
    manifold.NeighborMode = 'Supervised';
    manifold.WeightMode = 'HeatKernel';
    manifold.gnd=[Ys;Ytpesudo];
    manifold.k=0;
    W = lapgraph(Z',manifold);
    W=W-diag(diag(W));
    D = diag(sparse( sum(W) ));
    hotY=[hotmatrix(Ys,C);hotmatrix(Ytpesudo,C)];
    hotY_2=(D-alpha*W)\hotY;
    hotY_2(hotY_2==0)=-inf;
    [~,Ytpesudo]=max(hotY_2(ns+1:end,:),[],2);
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f\n',i,acc);
    end
end
end

