function [acc,acc_ite] = PDALC(Xs,Ys,Xt,Yt,options)
%% Implementation of PDALC
%%% Authors:    Li et al.
%%% Paper:      2021-Progressive Distribution Alignment Based on Label Correction for Unsupervised Domain Adaptation
%% input
%%% T/t:       iteration times
%%% dim:       the dimension
%%% beta, lambda, mu:   the hyper-parameters
%%% SRM_Kernel, SRM_gamma, SRM_mu: the hyper-parameters of SRM
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
    options=defaultOptions(options,...
                'T',5,...
                't',15,...
                'dim',30,...
                'beta',0.06,...
                'lambda',0.32,...
                'mu',1,...
                'SRM_Kernel','linear',... %SRM Kernel
                'SRM_gamma',1,... % SRM gamma
                'SRM_mu',1);% SRM mu
    T=options.T;
    t=options.t;
    beta=options.beta;
    lambda=options.lambda;
    mu=options.mu;
    dim=options.dim;
    opt.Kernel=options.SRM_Kernel;
    opt.gamma=options.SRM_gamma;
    opt.mu=options.SRM_mu;
    
    Xs=normr(Xs')';
    Xt=normr(Xt')';
%     Xs = double(zscore(Xs',1))';
%     Xt = double(zscore(Xt',1))';
    [X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt,1);
    acc_ite=[];
    M0=marginalDistribution(Xs,Xt,C);
    H=centeringMatrix(n);
    barXs=Xs*hotmatrix(Ys,C,1); % m * C
    E=centeringMatrix(C)*C;
    E=E./norm(E,'fro');
    Ytpseudo=zeros(nt,1);
    V=ones(n,C);
    G=zeros(nt,1);
    lt=zeros(nt,1);
    for i=1:T
        for j=1:t
            Mc=0;
        if ~isempty(find(Ytpseudo>0, 1))
           Mc=conditionalDistribution(Xs,Xt,Ys,Ytpseudo,C);
        end
        % compute MMD
        M=M0+Mc;
        M=M/norm(M,'fro');
        % compute Q
        Y=[Ys;Ytpseudo];
        Q=0;
        for k=1:C
            idx= (Y==k);
            len=length(find(idx));
            Rc=diag(V(idx,k));
            Xsc=X(:,idx);
            onesNc=ones(len,1);
            tmpQ=Xsc*Rc*Xsc'-2*Xsc*Rc*onesNc*barXs(:,k)'+trace(Rc)*barXs(:,k)*barXs(:,k)';
            Q=Q+tmpQ;
        end
        if Q~=0
            Q=Q./norm(Q,'fro');
        end
        % compute (6)
        [A,~]=eigs(Q+mu*X*M*X'-2*beta*barXs*E*barXs'+lambda*eye(m),X*H*X',dim,'sm');
        Z=A'*X;
        Z = Z*diag((1./sqrt(sum(Z.^2))));
        Zs=Z(:,1:ns);
        Zt=Z(:,ns+1:end);
        [Ytpseudo,~,~] = SRM(Zs,Zt,Ys,opt);
        % update Vt by (10)
        phi=1/t;
        V(ns+1:end,:)=V(ns+1:end,:)+phi*hotmatrix(Ytpseudo,C);
        % update Vs
        V(1:ns,:)=(1/(nt*C)*sum(sum(V(ns+1:end,:))));
        % update Ytpseudo by (14)
        idx=lt~=0;
        Ytpseudo(idx)=lt(idx);
        acc=getAcc(Ytpseudo,Yt);
        if isfield(options,'display')
            fprintf('[%2d]-[%2d] acc:%.4f\n',i,j,acc);
        end
        end
        % compute p(V)
        p=V(ns+1:end,:)./sum(V(ns+1:end,:),2);
        % compute G
        G= - sum(informationEntropy(p),2);
        tau=getMinMax(G,3,'min',1e-6);
        % compute lt by (13)
        [~,lt]=max(V,[],2);
        lt(G>=tau)=0;
        lt=lt(ns+1:end);
        % update Ytpseudo by (14)
        idx=lt~=0;
        Ytpseudo(idx)=lt(idx);
        % set V=1
        V=ones(n,C);
        acc=getAcc(Ytpseudo,Yt);
        acc_ite=[acc_ite,acc];
        if isfield(options,'display')
            fprintf('[%2d] acc:%.4f\n',i,acc);
        end
    end
    
    
end

