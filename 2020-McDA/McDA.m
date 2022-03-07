function [acc,acc_ite] = McDA(Xs,Ys,Xt,Yt,options)
%% Implementation of PDR
%%% Authors:    Zhang et al.
%%% Paper:      2020-Maximum Mean and Covariance Discrepancy for Unsupervised Domain Adaptation
%% input
%%% T:       iteration times
%%% dim:     reduced dimension
%%% lambda:  The weight of regularization
%%% beta:    The weight of MCD
%%% kernel_type: The kernel mapping, e.g., 'primal' (skip kernel mapping),'linear', 'rbf', and 'sam'
%%% kernel_param: The weight of kernel (sigma in kernel project)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
    acc_ite=[];
    options=defaultOptions(options,...
                'T',10,...
                'dim',30,...
                'lambda',0.1,...
                'beta',1,...
                'kernel_type','primal',...
                'kernel_param',1);
    lambda=options.lambda;
    beta=options.beta;
    dim=options.dim;
    T=options.T;
    [X,ns,~,n,m,C] = datasetMsg(Xs,Ys,Xt);
    
    M0=marginalDistribution(Xs,Xt,C);
    Z0=MCD(Xs,Xt,Ys,[],0);
    H=centeringMatrix(n);
    Ytpseudo=[];
    for i=1:T
        Mc=0;
        Zc=0;
        if ~isempty(Ytpseudo)
            Zc=MCD(Xs,Xt,Ys,Ytpseudo,1);
            Mc= conditionalDistribution(Xs,Xt,Ys,Ytpseudo,C);
        end
        M=M0+Mc;
        Z=Z0+Zc;
        if ~strcmp(options.kernel_type,'primal')
            K=kernelProject(options.kernel_type,X,[],options.kernel_param);
            left=K*(M+beta*Z*(K'*K)*Z)*K'+lambda*eye(n);
            right=K*H*K';
            [A,~]=eigs(left,right,dim,'sm');
            Z=A'*K;
        else
            left=X*(M+beta*Z*(X'*X)*Z)*X'+lambda*eye(m);
            right=X*H*X';
            [A,~]=eigs(left,right,dim,'sm');
            Z=A'*X;
        end
        Z=L2Norm(Z')';
        Zs=Z(:,1:ns);
        Zt=Z(:,ns+1:end);
        Ytpseudo=classifyKNN(Zs,Ys,Zt,1);
        acc=getAcc(Ytpseudo,Yt);
        acc_ite=[acc_ite,acc];
        if isfield(options,'display')
           fprintf('[%2d] acc:%.4f\n',i,acc); 
        end
    end
end

