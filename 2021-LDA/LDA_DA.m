function [acc,acc_ite] = LDA_DA(Xs,Ys,Xt,Yt,options)
%% Implementation of LDA
%%% Authors:    Xiao et al.
%%% Paper:      2021-Label Disentangled Analysis for unsupervised visual domain adaptation
%% Notice
%%% The paper uses the Back-Tracking Method to get the parameters with 
%%%     the highest classification accuracy.
%% input
%%% T:       iteration times
%%% dim:       the dimension
%%% mu:   the hyper-parameters of concat[X;hotY];
%%% lambda: the regularization term
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
    options=defaultOptions(options,...
                'T',10,...
                'dim',100,...
                'mu',0.1,...
                'lambda',0.1);
    addpath(genpath('./liblinear-2.30'));
    which train;
    T=options.T;
    dim=options.dim;
    lambda=options.lambda;
    mu=options.mu;
%     Ytpseudo=classifyKNN(Xs,Ys,Xt,1); % initialize the target pseudo labels by 1NN
    svmmodel = train(double(Ys), sparse(double(Xs')),'-s 1 -B 1.0 -q');
    %%% the Yt below is used to calculate the accuracy only,
    %%% Yt is not involved in the training.
    [Ytpseudo,~,~] = predict(double(Yt), sparse(double(Xt')), svmmodel,'-q');
    if isfield(options,'display')
        fprintf('[init] acc:%.4f\n',getAcc(Ytpseudo,Yt));
    end
    [m,ns]=size(Xs);
    nt=size(Xt,2);
    n=ns+nt;
    C=length(unique(Ys));
    acc_ite=[];
    H=centeringMatrix(n);
    hotYs=hotmatrix(Ys,C,0);
    hotYtpseudo=hotmatrix(Ytpseudo,C,0);
    Xsnew=[Xs;mu*hotYs'];
    Xtnew=[Xt;mu*hotYtpseudo'];
    % solve M0 by Eq. (9)
    M0=marginalDistribution(Xsnew,Xtnew,C);
    for i=1:T
        % update Xt_new by concat Xt and Ytpseudo
        hotYtpseudo=hotmatrix(Ytpseudo,C,0);
        Xtnew=[Xt;mu*hotYtpseudo'];
        % solve Mc by Eq. (11)
        Mc=conditionalDistribution(Xsnew,Xtnew,Ys,Ytpseudo,C);
        Xnew=[Xsnew,Xtnew];
        M=M0+Mc;
        M=M./norm(M,'fro');
        [P,~]=eigs(Xnew*M*Xnew'+lambda*eye(m+C),Xnew*H*Xnew',dim,'sm');
        Z=P'*Xnew;
        Z=Z-mean(Z,2);
        Z=L2Norm(Z')';
        Zs=Z(:,1:ns);
        Zt=Z(:,ns+1:end);
        svmmodel = train(double(Ys), sparse(double(Zs')),'-s 1 -B 1.0 -q');
        %%% the Yt below is used to calculate the accuracy only,
        %%% Yt is not involved in the training.
        [Ytpseudo,~,~] = predict(double(Yt), sparse(double(Zt')), svmmodel,'-q');
%         Ytpseudo=classifyKNN(Zs,Ys,Zt,1);
        acc=getAcc(Ytpseudo,Yt);
        acc_ite=[acc_ite,acc];
        if isfield(options,'display')
            fprintf('[%2d] acc:%.4f\n',i,acc);
        end
    end
    
    
end

