function [acc,acc_ite] = LRSR(Xs,Ys,Xt,Yt,options)
%% Implementation of LRSR
%%% Authors: Fang et al.
%%% Paper: 2015-Discriminative Transfer Subspace Learning via Low-Rank and Sparse Representation
%% input
%%% T:       iteration times
%%% u:       ADMM increasing parameter
%%% rho:     ADMM incrasing rate
%%% convergenceValue    the cut-off value of ADMM
%%% alpha    ||Z2||_1
%%% beta     ||E||_1
%%% gamma    kernel parameter
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
acc_ite=[];
options=defaultOptions(options,...
                            'T',10,...
                            'u',0.1,...
                            'rho',1.01,...
                            'convergenceValue',1e-6,...
                            'alpha',1,...
                            'beta',1,...
                            'gamma',1,...
                            'kernel_type','primal');
convergenceValue=options.convergenceValue;
u=options.u;
rho=options.rho;
alpha=options.alpha;
beta=options.beta;
T=options.T;
[X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt);
% kernel
if ~strcmpi(options.kernel_type,'primal')
    X = kernelProject(options.kernel_type,X,[],options.gamma);
    Xs=X(:,1:ns);
    Xt=X(:,ns+1:end);
    m=n;
end
XssX=Xs*Xs';
hotYs=hotmatrix(Ys,C)';
% compute B
B=hotYs;B(B==0)=-1;
% init parameter
M=eye(C,ns);
Z=zeros(ns,nt);
Z1=Z;Z2=Z;
E=zeros(C,nt);
Y1=zeros(C,nt);
Y2=zeros(ns,nt);
Y3=zeros(ns,nt);
for i=1:T
    % compute P according to (13)
    V1=hotYs+B.*M;
    V3=Xt-Xs*Z;
    V4=E-Y1/u;
    left=XssX+u*(V3*V3')+2*alpha*eye(m);
    cons=Xs*V1'+u*V3*V4';
    P=left\cons;
    % compute Z according to (15)
    V5=P'*Xt-E+Y1/u;
    V6=Z1-Y2/u;
    V7=Z2-Y3/u;
    left=Xs'*(P*P')*Xs+ 2*eye(ns);
    cons=Xs'*P*V5+V6+V7;
    Z=left\cons;
    % compute Z1 according to (17)
    Z1=SVT(Z+Y2/u, 1/u);
    % compute Z2 according to (19)
    Z2=shrink(Z+Y3/u,alpha/u);
    % compute E according to (21)
    E=shrink(P'*Xt-P'*Xs*Z+Y1/u,beta/u);
    % compute M according to (25)
    R=P'*Xs-hotYs;
    M=max(R.*B,0);
    % update multiplier
    Y1=Y1+u*(P'*Xt-P'*Xs*Z-E);
    Y2=Y2+u*(Z-Z1);
    Y3=Y3+u*(Z-Z2);
    u=min(rho*u,1e6); 
    % predict
    Zk=P'*X;
    Zs=Zk(:,1:ns);
    Zt=Zk(:,ns+1:end);
    knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
    Ytpesudo = knn_model.predict(Zt');
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    % compute the residual
    residualY1=norm(P'*Xt-P'*Xs*Z-E,'inf');
    residualY2=norm(Z-Z1,'inf');
    residualY3=norm(Z-Z2,'inf');
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f ,r1: %.6f,r2: %.6f,r3 %.6f\n',i,acc,residualY1,residualY2,residualY3);
    end
    if residualY1<=convergenceValue &&residualY2 <=convergenceValue &&residualY3 <=convergenceValue
        if isfield(options,'display')
            fprintf('function convergenceValue,break \n');
        end
        break;
    end
end
end

