function [Yt,f,Z,opt] = SRM(Xs,Xt,Ys,options)
%% Structure Risk Minimization (SRM)
%%% Input:
%%% Xs (m*ns): training samples
%%% Xt (m*nt): test samples
%%% Ys (ns*1): the labels of training samples
%%% options.Kernel: the kernel function, linear/1, rbf/2, sam/3
%%% options.gamma: the kernel parameter
%%% options.mu: the regularization parameter
%%% Output:
%%% f (m*C): learned projection matrix
%%% Yt (nt*1): the predicted labels
%%% Z (C*n): 
    X=[Xs,Xt];
    ns=size(Xs,2);
    nt=size(Xt,2);
    C=length(unique(Ys));
    Yt=[];
    options=defaultOptions(options,'Kernel',2,...
        'gamma',0.1,...
        'mu',0.1);
    Kernel=options.Kernel;
    gamma=options.gamma;
    mu=options.mu;
    
    if Kernel~=0
        X = kernelProject(Kernel,X,[],gamma);
        E=diag([ones(ns,1);zeros(nt,1)]);
        Y=[hotmatrix(Ys,C);zeros(nt,C)];
        m=size(X,1);
        f=((E*X)+mu*eye(m))\(E*Y);
    else
        m=size(X,1);
        E=diag([ones(ns,1);zeros(nt,1)]);
        Y=[hotmatrix(Ys,C);zeros(nt,C)];
        f=((X*E*X')+mu*eye(m))\(X*E*Y);
    end
    Z=f'*X;
    [~,Yt]=max(Z(:,ns+1:end),[],1);
    Yt=Yt';
    opt=options;
end
