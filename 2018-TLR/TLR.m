function [acc,acc_ite] = TLR(Xs,Ys,Xt,Yt,options)
%% Implementation of TLR
%%% Authors:    Xiao, Pan, et al.
%%% Paper:      2018-TLR: Transfer latent representation for unsupervised domain adaptation
%% input
%%% T:      iteration times
%%% dim:    reduced dimension
%%% alpha:  tradeoff parameter
%%% beta:   tradeoff parameter
%%% lambda: regularization parameter
%%% gamma:  kernel parameter
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)


options=defaultOptions(options,...
    'dim',30,...
    'T',10,...
    'alpha',1,...
    'beta',1,...
    'lambda',0.1,...
    'gamma',1,...
    'kernel_type','linear');
    kernel_type=options.kernel_type;
    alpha=options.alpha;
    beta=options.beta;
    dim=options.dim;
    gamma=options.gamma;
    lambda=options.lambda;
    [K,ns,nt,n,~,C] = datasetMsg(Xs,Ys,Xt);
    % compute K according to (3)
    K=kernelProject(kernel_type,K,[],gamma);
    % compute Hs and Ht accroding to (3)
    Hs=K(1:ns,:);
    Ht=K(ns+1:end,:);
    % compute L accroding to (4)
    L=marginalDistribution(Xs,Xt,1); % i.e., M0
    % compute M accroding to (18)
    M=blkdiag(alpha*eye(ns),beta*eye(nt));
    % compute A and B according to (17)
    A=K*M*K;
    B=K*L*K;
    A=A/norm(A,'fro');
    B=B/norm(B,'fro');
    [U,S,V]=svd((eye(n)+B)\A);
    num=size(S,1);
    S=S.*blkdiag(eye(dim),zeros(num-dim));
    W=U*S*V';
    Ps=Hs*W;
    Pt=Ht*W;
    knn_model = fitcknn(Ps,Ys,'NumNeighbors',1);
    Ytpesudo = knn_model.predict(Pt);
    % save record
    acc=getAcc(Ytpesudo,Yt);
    if isfield(options,'display')
       fprintf('acc:%.4f\n',acc); 
    end
    acc_ite=[acc];
end

