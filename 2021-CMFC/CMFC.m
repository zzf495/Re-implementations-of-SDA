function [acc,acc_ite] = CMFC(Xs,Ys,Xt,Yt,options)
%% Implementation of CMFC
%%% Authors:    Chang et al.
%%% Paper:      2021-Unsupervised domain adaptation based on cluster matching and Fisher criterion for image classification
%% input
%%% T:       Iteration times
%%% dim:     Reduced dimension
%%% alpha:   The weight of Marginal distribution (M)
%%% beta:    The weight of `Cluster matching` (Eq. (1))
%%% gamma:   The weight of `Cluster matching` (Eq. (2))
%%% lambda   The weight of regularization 
%%% kernel_type: The kernel mapping, e.g., 'primal' (skip kernel mapping),'linear', 'rbf', and 'sam'
%%% kernel_param: The weight of kernel (sigma in kernel project)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
    addpath(genpath('./liblinear-2.30'));
    which train;
    acc_ite=[];
    options=defaultOptions(options,...
                'T',10,...
                'dim',100,...
                'alpha',0.1,...
                'beta',0.5,...
                'gamma',30,...
                'kernel_type','primal',...
                'kernel_param',1,...
                'lambda',0.01);%% the authors set lambda=0.01 empirically
    alpha=options.alpha;
    lambda=options.lambda;
    gamma=options.gamma;
    beta=options.beta;
    dim=options.dim;
    T=options.T;
    Xs=normr(Xs')';
    Xt=normr(Xt')';
    [X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt);
    % get Ytpseudo
    svmmodel = train(double(Ys), sparse(double(Xs')),'-s 1 -B 1.0 -q');
    %%% the Yt below is used to calculate the accuracy only,
    %%% Yt is not involved in the training.
    [Ytpseudo,~,~] = predict(Yt, sparse(Xt'), svmmodel,'-q');
    % init
    H=centeringMatrix(n);
    E=blkdiag(zeros(ns,ns),eye(nt,nt));
    if ~isempty(Ytpseudo)
        Vt=hotmatrix(Ytpseudo,C)';
    end
    V=[zeros(C,ns),Vt];
    Qs=hotmatrix(Ys,C,1);
    Q=[Qs;zeros(nt,C)];
    M=marginalDistribution(Xs,Xt,1);
    Ls = intraScatter(Xs,Ys,C);
    Ds = interScatter(Xs,Ys,C);
    for i=1:T
        
        % init L by (3),(4)
        Lt = intraScatter(Xt,Ytpseudo,C);
        L=blkdiag(Ls,Lt);
        % init D by (6),(7)
        Dt = interScatter(Xt,Ytpseudo,C);
        D=blkdiag(Ds,Dt);
        % update P by (15)
        %%% calculate F
        A=inv(V*V'+gamma/beta*eye(C));
        B=E*V'+gamma/beta*Q;
        F=beta*(E*E'+gamma/beta*(Q*Q')-B*(A'+A)*B'+ ((B*A)*(V*V'+gamma/beta*eye(C))*(A'*B')) );
        if ~strcmp(options.kernel_type,'primal')
            K=kernelProject(options.kernel_type,X,[],options.kernel_param);
            [P,~]=eigs(K*(L+alpha*M-D+F)*K'+lambda*eye(n),K*H*K',dim,'sm');
            Z=P'*K;
        else
            [P,~]=eigs(X*(L+alpha*M-D+F)*X'+lambda*eye(m),X*H*X',dim,'sm');
            Z=P'*X;
        end
%%     notice: If we update `Omega` first, the performance will be limited,
%%%            since the `Omega` only affect the classification.
%%%            Here, we use the operations used in CMMS, it seems work well.
        % update Omega by (13)
        Z = Z-mean(Z,2);
        Z = Z*diag((1./sqrt(sum(Z.^2))));
        Omega=(Z*E*V'+gamma/beta *Z*Q)/(V*V'+gamma/beta*eye(C));

        % update V by (16)
        Zt=Z(:,ns+1:end);
        knn_model = fitcknn(Omega',(1:C),'NumNeighbors',1);
        Ytpseudo = knn_model.predict(Zt');
        % update Vt by Ytpseudo
        Vt=hotmatrix(Ytpseudo,C)';
        V=[zeros(C,ns),Vt];
        % get acc
        acc=getAcc(Ytpseudo,Yt);
        acc_ite=[acc_ite,acc];
        if isfield(options,'display')
           fprintf('[%2d] acc:%.4f\n',i,acc); 
        end
    end
end

