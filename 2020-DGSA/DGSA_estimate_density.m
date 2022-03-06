function [delta] = DGSA_estimate_density(X,Y,sigma)
%% input:
%%%     X       m*n samples set
%%%     Y       n*1 labels set
%%%     gamma   the exp coefficient
%% output:
%%%     delta   the estimated density
    n=size(X,2);
    % calculate distance
    distClassMeans = EuDist2(X',X')./(2*sigma*sigma);
    K = exp(-distClassMeans);
    rho_a=zeros(n,1);rho_b=zeros(n,1);
    for i=1:n
        sameIdx=(Y==Y(i));sameIdx(i)=0;
        diffIdx=~sameIdx;diffIdx(i)=0;
        rho_a(i)=1/(length(find(sameIdx))-1) * sum(K(i,sameIdx));
        rho_b(i)=1/(length(find(diffIdx))) * sum(K(i,diffIdx));
    end
    delta=rho_a-rho_b;
end