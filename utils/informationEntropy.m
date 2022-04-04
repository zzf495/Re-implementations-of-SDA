function [p] = informationEntropy(x)
%% input:
%%% x: m*n, the matrix
%% output:
%%% p: m*n, the information entropy, if x==0, then p==0, else p=-x*log(x)
    logP=log(x);
    logP(isinf(logP))=0;
    p=-x.*logP;
end

