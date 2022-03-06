function [Sb] = betweenScatter(X,Y)
    %% there are two methods construct the betweenScatter
    %%% 1, Sb=\sum_{c=1}^C n_c (x - u_c)(x - u_c)' (DIJDA)
    %%% 2, Sb=\sum_{c=1}^C n_c (u_c - mean(X,2))(u_c - mean(X,2))' (JGSA)
    % input 
    %   X: m*n
    %   Y: n*1
    % output
    %   Sb: m*m
    C=length(find(unique(Y)));
    n=length(Y);
    Sb=0;
    Fc=mean(X,2);
    for i=1:C
       Xc=X(:,Y==i);
       F=Xc-Fc;
       nc=size(Xc,2);
       Sb=Sb+nc*(F*F');
    end
%     Sb=Sb./n;
end