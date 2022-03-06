function M0 = marginalDistribution(Xs,Xt,C)
%% Inputs:
%%% Xs      : Source domain feature matrix, m*ns
%%% Xt      : Target domain feature matrix, m*nt
%%% C       : the number of labels
%% Output: M0 =  || 1/ns \sum Xs - 1/nt \sum Xt ||
ns = size(Xs,2);
nt = size(Xt,2);
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M0 = e * e' * C;  %multiply C for better normalization
end

