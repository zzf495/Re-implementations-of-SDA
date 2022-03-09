function [Ds,XDsX] = interScatter(Xs,Ys,C)
%% input
%%% Xs: the samples, m*ns
%%% Ys: the labels, n*1
%% output
%%% Ds: the inter-scatter matrix with dimension n*n
%%% Ds(i,j)= 
%%%                             \frac{1}{(n_s^k)^2}             if y_i=y_j=k, and
%%%                              - \frac{1}{n_s^k n_s^l}        if y_i=k, y_j=l,and k\neq l
%% formula
%%% $\sum\limits_{c_1\neq c_2}^C \|\frac{1}{n_s^{c_1}} \sum\limits_{i=1}^{n_s^{c_1}} x_i^s - \frac{1}{n_s^{c_2}} \sum\limits_{j=1}^{n_s^{c_2}} x_j^s \|$
        
        ns=size(Xs,2);
        Ds=zeros(ns,ns);
        for i=1:C
            %% if y_i=y_j=k, and - \frac{1}{n_s^k n_s^l}
            idx=find(Ys==i);
            Ds(idx,idx)=1/(length(idx)^2);
            %% if y_i=k, y_j=l,and k\neq l
            for j=i+1:C
                idx2=find(Ys==j);
                if ~isempty(idx2)
                    Ds(idx,idx2)=-1/(length(idx)*length(idx2));
                    Ds(idx2,idx)=-1/(length(idx)*length(idx2));
                end
            end
        end
        %% if output number is two
        if nargout==2
            XDsX=Xs*Ds*Xs';
        end
end

