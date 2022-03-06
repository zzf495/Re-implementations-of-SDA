function [X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt,standard)
%% input:
%%%     Xs: source samples (m*ns) 
%%%     Xt: target samples (m*nt) 
%%%     Ys: source labels (ns*1) 
%% output
%%%     ns: the number of Xs
%%%     nt: the number of Xt
%%%     n:  ns+nt
%%%     m:  the number of features
%%%     C:  the number of unique Ys

[a1,a2]=size(Xs);
if length(Ys)==a1
   Xs=Xs';Xt=Xt'; 
end
X=[Xs,Xt];
if nargin==3
    standard=1;
end
if standard==1
    X=L2Norm(X')';
end
[m,ns]=size(Xs);
nt=size(Xt,2);
n=ns+nt;
C=length(unique(Ys));
end

