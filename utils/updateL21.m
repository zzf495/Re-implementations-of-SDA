function [G] = updateL21(E)
 %% input
 %%%    ||E||_{2,1} => tr(E'GE)  size: (m*d)' m*m (m*d)
 %% output
 %%% G : 
    ec = sqrt(sum(E.*E,2)+eps);
    G = 0.5./ec;
    n=length(G);
    G = spdiags(G,0,n,n);
end
