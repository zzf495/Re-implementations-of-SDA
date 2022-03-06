function [acc,acc_ite] = VDA(Xs,Ys,Xt,Yt,options)
%% Implementation of VDA
%%% Authors:    Tahmoresnezhad et al.
%%% Paper:      2017-Visual domain adaptation via transfer feature learning
%% input
%%% T:       iteration times
%%% dim:     reduced dimension
%%% lambda:  regularization parameter (default 1)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
    options=defaultOptions(options,...
                            'dim',10,...
                            'T',10,...
                            'lambda',0.1);
    lambda = options.lambda;
    dim = options.dim;
    T = options.T;
    [X,ns,~,n,m,C] = datasetMsg(Xs,Ys,Xt);
    acc_ite = [];
	Y_tar_pseudo = [];
    Sw=withinScatter(Xs,Ys);
    Sw=Sw/norm(Sw,'fro');
    M0=marginalDistribution(Xs,Xt,C);
    H=centeringMatrix(n);
    %% Iteration
    for i = 1 : T
        %%% Mc
        N = 0;
        if ~isempty(Y_tar_pseudo)
           N=conditionalDistribution(Xs,Xt,Ys,Y_tar_pseudo,C);
        end
        M =  M0 + N;
%         M = M / norm(M,'fro');
        %% Calculation
        [A,~] = eigs(X*M*X'+Sw+lambda*eye(m),X*H*X',dim,'SM');
        Z = A'*X;
        % normalization for better classification performance
		Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:ns);
        Zt = Z(:,ns+1:end);
        knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
        Y_tar_pseudo = knn_model.predict(Zt');
        acc=getAcc(Y_tar_pseudo,Yt);
        acc_ite = [acc_ite,acc];
        if isfield(options,'display')
            fprintf('[%2d]-th acc: %0.4f\n',i,acc);
        end
    end
end