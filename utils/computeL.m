function [L,D,W] = computeL(X,manifold)
%% input 
%%% X: fea*n
%%% manifold: the construct options of graph
    %% Construct graph Laplacian
    if ~isfield(manifold,'normr')
        manifold.normr=1;
    end
    n=size(X,2);
    W = lapgraph(X',manifold);
    D=diag(sparse(sum(W)));
    if manifold.normr==1
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n) - Dw * W * Dw;
    else
        L = D-W;
    end
    
end

