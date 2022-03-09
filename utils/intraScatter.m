function [Ds,XDsX] = intraScatter(Xs,Ys,C)
%% input
%%% Xs: the samples, m*ns
%%% Ys: the labels, n*1
%% output
%%% Ds: the intra-scatter matrix with dimension n*n
%%% Ds(i,j)= 
%%%                             1-\frac{1}{n^c}                 if  i=j
%%%                              - \frac{1}{n^c}                if 	i\neq j
%%%                             0                               otherwise
%% formula
%%% \sum\limits_{c=1}^C \frac{1}{n_s^c} \sum\limits_{y_i=y_j=c} \|P'X_i-P'X_j\|^2     
        ns=size(Xs,2);
        Ds=zeros(ns,ns);
        for i=1:C
            %%  - \frac{1}{n^c}                if 	i\neq j
            idx=find(Ys==i);
            Ds(idx,idx)=-1/(length(idx));
        end
        diagIndex=logical(diag(ones(ns,1)));
        Ds(diagIndex)=1+Ds(diagIndex); %% Ds(diagIndex) is negative, if use '+', then it becomes 1+\frac{1}{n^c} 
        %% if output number is two
        if nargout==2
            XDsX=Xs*Ds*Xs';
        end
end

