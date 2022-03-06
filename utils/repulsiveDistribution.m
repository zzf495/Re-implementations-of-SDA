function [Mrep] = repulsiveDistribution(Xs,Xt,Ys,Yt,C,mode)
%% Inputs:
%%% Xs      : Source domain feature matrix, m*ns
%%% Xt      : Target domain feature matrix, m*nt
%%% Ys      : Source labels ns*1
%%% Yt      : Target labels nt*1
%%% C       : the number of labels
%%% mode    : 1: S->T, 2:T->S, 3: S->S, 4: T->T
%% Output: M_rep = \sum || 1/ns^c \sum Xs^c - 1/nt^(~c) \sum Xt^(~c) ||
ns=size(Xs,2);
nt=size(Xt,2);
n=ns+nt;
Mrep=zeros(n,n);
if mode==1
      for c = 1:C
        e = zeros(n,1);
        e(Ys==c) = 1 / length(find(Ys==c));
        e(ns+find(Yt~=c)) = -1 / length(find(Yt~=c));
        e(isinf(e)) = 0;
        Mrep = Mrep + e*e';
      end
elseif mode==2
    for c = 1:C
        e = zeros(n,1);
        e(ns+find(Yt==c)) = 1 / length(find(Yt==c));
        e(Ys~=c) = -1 / length(find(Ys~=c));
        e(isinf(e)) = 0;
        Mrep = Mrep + e*e';
    end
elseif mode==3
    for c = 1:C
        e = zeros(n,1);
        e(Ys==c) = 1 / length(find(Ys==c));
        e(Ys~=c) = -1 / length(find(Ys~=c));
        e(isinf(e)) = 0;
        Mrep = Mrep + e*e';
    end
elseif mode==4
    for c = 1:C
        e = zeros(n,1);
        e(ns+find(Yt==c)) = 1 / length(find(Yt==c));
        e(ns+find(Yt~=c)) = -1 / length(find(Yt==c));
        e(isinf(e)) = 0;
        Mrep = Mrep + e*e';
    end
end
end