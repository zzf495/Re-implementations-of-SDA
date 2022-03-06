function [Mc] = conditionalDistribution(Xs,Xt,Ys,Ytpseudo,C)
%% Inputs:
%%% Xs      : Source domain feature matrix, m*ns
%%% Xt      : Target domain feature matrix, m*nt
%%% Ys      : Source labels ns*1
%%% Yt      : Target labels nt*1
%%% C       : the number of labels
%% Output: Mc = \sum || 1/ns^c \sum Xs^c - 1/nt^c \sum Xt^c ||
if size(Ys,2)>1
   Ys=Ys'; 
end
if size(Ytpseudo,2)>1
   Ytpseudo=Ytpseudo'; 
end
ns=size(Xs,2);
nt=size(Xt,2);
n=ns+nt;
Mc=zeros(n,n);
if ~isempty(Ytpseudo)
    for c = 1:C
        cYs=find(Ys==c);
        cYt=find(Ytpseudo==c);
        cY=[cYs;ns+cYt];
        nc=length(cY);
        nc_Xs=length(cYs);
        nc_Xt=length(cYt);
        e=zeros(nc,1);
        e(1:nc_Xs)=1/nc_Xs;
        e(nc_Xs+1:end)=-1/nc_Xt;
        e(isinf(e)) = 0;
        Mc(cY,cY)=e*e';
    end
end
end

