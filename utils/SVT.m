function res = SVT(W,tau)
%% paper: A singular value thresholding algorithm for matrix completion
%% input:
%%%     W: dealed matrix
%%%     tau: threshold
%% ouput:
%%%     res:SVT results
    [U,S,V]=svd(W,'econ');
    S=sign(S).*max(abs(S)-tau,0);
    res=U*S*V';
end

