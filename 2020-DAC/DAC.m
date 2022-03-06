function [acc,acc_ite] = DAC(Xs,Ys,Xt,Yt,options)
%% Implementation of DAC
%%% Authors:        Yunyun Wang et al.
%%% Paper:          2020-Soft large margin clustering for unsupervised domain adaptation
%% input
%%% T           iteration times
%%% gamma       kernel parameter
%%% xi          MMD
%%% rho         manifold
%%% theta       SLMC
%%% mul         SLMC parameter
%%% k           KNN number
%%% gamma       kernel parameter
%%% kernel_type kernel type
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
Xs = normr(Xs')';
Xt = normr(Xt')';
acc_ite=[];
options=defaultOptions(options,...
                            'T',10,...
                            'lambda',1,...
                            'xi',1,...
                            'rho',1,...
                            'theta',1,...
                            'mul',2,...
                            'kernel_type','rbf',...
                            'gamma',1,...
                            'k',10);
rho=options.rho;
theta=options.theta;
lambda=options.lambda;
xi=options.xi;
T=options.T;
mul=options.mul;
[X,ns,nt,n,~,C] = datasetMsg(Xs,Ys,Xt);
% kernel project
K= kernelProject(options.kernel_type,X,[],options.gamma);
% init parameter
manifold.k = options.k;
manifold.Metric = 'Cosine';
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Cosine';
% W=lapgraph(X,manifold);
L=computeL(X,manifold);
L=L/norm(L,'fro');
% compute M0
M0=marginalDistribution(Xs,Xt,C);
% init Y hotmatrix
E=blkdiag(eye(ns),zeros(nt));
U=[];
Ytpesudo=[];
Yhot=hotmatrix([Ys;zeros(nt,1)],C);
for i=1:T
   
    UKc=0;
    ULc=0;
    if i>1
        for c=1:C
           UKc=UKc+diag(power(U(c,:),2));
           Lc=zeros(C,n);
           Lc(c,:)=1;
           ULc=ULc+diag(power(U(c,:),2))*Lc';
        end
    end
    N=0;
    if ~isempty(Ytpesudo)
        N=conditionalDistribution(Xs,Xt,Ys,Ytpesudo,C);
    end
    mu = DAC_estimate_mu(Xs',Ys,Xt',Ytpesudo);
    M = (1 - mu) * M0 + mu * N;
    M = M / norm(M,'fro');
    % compute P according to (13)
    left=(lambda*E'+xi*M+rho*L)*K+eye(n)+theta*UKc*K;
    right=(lambda*E*Yhot+theta*ULc);
    P=left\right;
    % compute u according to (16)
    Z=P'*K;
    F=zeros(C,n);
    % C*n    n*C
    for c=1:C
       a=zeros(C,n);
       a(c,:)=1;
       a=Z-a;
       F(c,:)=1./(power(sum(a.*a,1),1/(mul-1))); 
    end
    % C*n
    U=F./sum(F,1);
    % update L
%     L=computeL(Z,manifold);
% 	L=L/norm(L,'fro');
    % predict
    [~,Ytpesudo]=max(U(:,ns+1:end),[],1);
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f\n',i,acc);
    end
end
end

