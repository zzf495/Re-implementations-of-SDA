function [acc,acc_ite,A] = JGSA(Xs,Ys,Xt,Yt,options)
%% Implementation of JGSA
%%% Authors:    Zhang et al.
%%% Paper:      2017-Joint Geometrical and Statistical Alignment for Visual Domain Adaptation
%% input
%%% T:       Iteration times
%%% dim:     Reduced dimension 
%%% lambda:  Regularization parameter (default 1)
%%% beta:    The parameter of S_b in paper (default 1)
%%% mu:      The parameter of S_t in paper (default 1)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
acc_ite=[];
Xs = normr(Xs')';
Xt = normr(Xt')';
options=defaultOptions(options,...
    'dim',30,...
    'T',10,...
    'mu',1,...
    'beta',1,...
    'lambda',1);
dim = options.dim;
T=options.T;
beta=options.beta;
mu=options.mu;
% kernel_type = options.kernel_type;
% gamma = options.gamma;
lambda=options.lambda;
[~,ns,nt,~,m,C] = datasetMsg(Xs,Ys,Xt);

% update St according to (2)
Ht=centeringMatrix(nt);
St=Xt*Ht*Xt';
% update Sw according to (5)
Sw=withinScatter(Xs,Ys);
% update Sb accroding to (6)
Sb=betweenScatter(Xs,Ys);
Sb=Sb/norm(Sb,'fro');
% init M0
M0=marginalDistribution(Xs,Xt,C);

%% Iteration
t=ns+1;
Ytpesudo=classifyKNN(Xs,Ys,Xt,1);
for i=1:T
    N=0;
    if ~isempty(Ytpesudo)
        N=conditionalDistribution(Xs,Xt,Ys,Ytpesudo,C);
    end
    M=M0+N;
    M=M/norm(M,'fro');
    % update Ms according to (10)
    Ms=Xs*M(1:ns,1:ns)*Xs';
    % update Mt according to (11)
    Mt=Xt*M(t:end,t:end)*Xt';
    % update Mst according to (12)
    Mst=Xs*M(1:ns,t:end)*Xt';
    % update Mts according to (13)
    Mts=Xt*M(t:end,1:ns)*Xs';
    % solving W according (19)
    left=blkdiag(beta*Sb,mu*St);
    I=eye(m);
    right=[Ms+lambda*I+beta*Sw,Mst-lambda*I;
        Mts-lambda*I,Mt+(lambda+mu)*I];
    [W,~]=eigs(left,right,dim,'lm'); % or eigs(right,left,dim,'sm'); 
    % map and predict
    A=W(1:m,:);
    B=W(m+1:end,:);
    Zs=A'*Xs;
    Zt=B'*Xt;
    Ytpesudo=classifyKNN(Zs,Ys,Zt,1);
    % save record
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f\n',i,acc);
    end
end
end