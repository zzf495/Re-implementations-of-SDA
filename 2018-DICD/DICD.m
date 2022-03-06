function [acc,acc_ite] = DICD(Xs,Ys,Xt,Yt,options)
%% Implementation of DICD
%%% Authors:    Li et al.
%%% Paper:      2018-Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation
%% input
%%% T:       iteration times
%%% kernel_type:        kernel 
%%% gamma:   kernel parameter
%%% dim:     reduced dimension
%%% lambda:  regularization parameter
%%% rho:  Ddiff parameter in paper
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
options=defaultOptions(options,...
    'dim',10,...
    'T',10,...
    'lambda',0.1,...
    'rho',0.1,...
    'kernel_type','primal',...
    'gamma',1);
lambda = options.lambda;
dim = options.dim;
rho=options.rho;
T = options.T;
[X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt);
acc_ite = [];
Ypseudo = [];
M0=marginalDistribution(Xs,Xt,C);
H=centeringMatrix(n);
% init D_same according to (7)
Dsame_S=0;
for c=1:C
    e=zeros(ns,1);
    e(Ys==c)=sqrt(1/length(find(Ys==c)));
    e2=-(e*e');
    e2=e2-diag(diag(e2));
    Dsame_S=Dsame_S+ns*e2;
end
Dsame_S=Dsame_S+ns*eye(ns);
% init D_diff according to (12)
Ddiff_S=0;
diagE=zeros(ns,1);
for c=1:C
    e=zeros(ns,1);
    e(Ys==c)=1;
    diagE(Ys==c)=ns-length(find(Ys==c));
    e2=e*e';
    Ddiff_S=Ddiff_S+e2;
end
Ddiff_S=Ddiff_S-ones(ns)+diag(diagE);
Dsame_T=zeros(nt,nt);
Ddiff_T=zeros(nt,nt);
%% Iteration
for i = 1 : T
    %%% Mc
    N = 0;
    if ~isempty(Ypseudo)
        
        N=conditionalDistribution(Xs,Xt,Ys,Ypseudo,C);
        % update Dsame_T
        Dsame_T=0;
        for c=1:C
            e=zeros(nt,1);
            e(Ypseudo==c)=sqrt(1/length(find(Ypseudo==c)));
            e2=-(e*e');
            e2=e2-diag(diag(e2));
            Dsame_T=Dsame_T+nt*e2;
        end
        Dsame_T=Dsame_T+nt*eye(nt);
        % update Ddiff_T
        Ddiff_T=0;
        diagE=zeros(nt,1);
        for c=1:C
            e=zeros(nt,1);
            e(Ypseudo==c)=1;
            diagE(Ypseudo==c)=nt-length(find(Ypseudo==c));
            e2=e*e';
            Ddiff_T=Ddiff_T+e2;
        end
        Ddiff_T=Ddiff_T-ones(nt)+diag(diagE);
    end
    M =  M0 + N;
    Dsame=blkdiag(Dsame_S,Dsame_T);
    Ddiff=blkdiag(Ddiff_S,Ddiff_T);
    Dsame=Dsame/norm(Dsame,'fro');
    Ddiff=Ddiff/norm(Ddiff,'fro');
    Lambda=M+Dsame-rho*Ddiff;
    %% Calculation
    if strcmpi(options.kernel_type,'primal')
       [A,~] = eigs(X*Lambda*X'+lambda*eye(m),X*H*X',dim,'SM');
       Z = A'*X;
    else
        K=kernelProject(options.kernel_type,X,[],options.gamma);
        [A,~] = eigs(K*Lambda*K'+lambda*eye(n),K*H*K',dim,'SM');
        Z = A'*K;
    end

    %normalization for better classification performance
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
    Zs = Z(:,1:ns);
    Zt = Z(:,ns+1:end);
    Ypseudo=classifyKNN(Zs,Ys,Zt,1);
    acc=getAcc(Ypseudo,Yt);
    acc_ite = [acc_ite,acc];
    if isfield(options,'display')
        fprintf('[%2d]-th acc: %0.4f\n',i,acc);
    end
end
end