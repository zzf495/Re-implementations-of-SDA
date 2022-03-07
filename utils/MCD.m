function [Z] = MCD(Xs,Xt,Ys,Ytpseudo,mode)
%% input
%%% Xs: source samples, m*n_s
%%% Xt: target samples, m*n_t
%%% Ys: source labels, n_s*1
%%% Yt: target pseudo labels, n_t*1
%%% mode: 
%%%%        mode==0: marginal cross-domain distributions, 
%%%%        mode==1: conditional cross-domain distributions
    ns=size(Xs,2);
    nt=size(Xt,2);
    if mode==0
        es=1/ns*(eye(ns,ns)-1/ns*ones(ns,ns));
        et=1/nt*(eye(nt,nt)-1/nt*ones(nt,nt));
        Z=blkdiag(es,-et);
    else
        C=length(unique(Ys));
        Z=zeros(ns+nt,ns+nt);
        for i=1:C
            sIdx=find(Ys==i);
            nsc=length(sIdx);
            es=1/nsc*(eye(nsc,nsc)-1/nsc*ones(nsc,nsc));
            Z(sIdx,sIdx)=es;
            
            tIdx=find(Ytpseudo==i);
            ntc=length(tIdx);
            et=1/ntc*(eye(ntc,ntc)-1/ntc*ones(ntc,ntc));
            Z(ns+tIdx,ns+tIdx)=-et;
        end
    end
end

