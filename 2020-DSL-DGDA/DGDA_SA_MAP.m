function [Zs,Zt] = DGDA_SA_MAP(Xs,Ys,Xt,options)
    % init parameter
    options=defaultOptions(options,'theta',1,'dim',30);
    theta=options.theta;
    dim=options.dim;
    [m,ns]=size(Xs);
    nt=size(Xt,2);
    % solve A according to (5)
    Sw=withinScatter(Xs,Ys);
    Sb=betweenScatter(Xs,Ys);
    Center=Xs*centeringMatrix(ns)*Xs';
    [A,~]=eigs(theta*Sw+eye(m),Center+theta*Sb,dim,'sm');
    % solve B according to (6)
    Center=Xt*centeringMatrix(nt)*Xt';
    [B,~]=eigs(eye(m),Center,dim,'sm');
    % solve M
    M=A'*B;
    Zs=Xs'*A*M;Zs=Zs';
    Zt=Xt'*B;Zt=Zt';
end