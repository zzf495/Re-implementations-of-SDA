function [acc,acc_ite] = SPDA(Xs,Ys,Xt,Yt,options)
%% Implementation of SPDA
%%% Authors: Xiao et al.
%%% Paper: 2019-Structure preservation and distribution alignment in discriminative transfer subspace learning
%% input
%%% dim:     The dimension
%%% T:       The iterations
%%% u:       ADMM increasing parameter
%%% rho:     ADMM incrasing rate
%%% convergenceValue    the cut-off value of ADMM
%%% alpha    The weight of regularization of P
%%% beta     The weight to penalize the noises (Eq. (22))
%%% lambda   The weight to penalize the Z1  (Eq. (26))
%%% sigma    The weight of manifold term
%%% delta    The weight of conditional distribution
%%% k:       k-neighbors of graph
%%% kernel_type: The kernel mapping, e.g., 'primal' (skip kernel mapping),'linear', 'rbf', and 'sam'
%%% kernel_param: The weight of kernel (sigma in kernel project)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
Xs = normr(Xs')';
Xt = normr(Xt')';
acc_ite=[];
options=defaultOptions(options,...
    'dim',30,...
    'T',10,...
    'alpha',1,...
    'beta',1,...
    'delta',1,...
    'sigma',1,...
    'k',5,... 
    'lambda',1,...
    'convergenceValue',1e-6,...
    'rho',1.01,...
    'u',0.1,...
    'kernel_type','primal',...
    'kernel_param',1);

convergenceValue=options.convergenceValue;
u=options.u;
rho=options.rho;
k=options.k;
alpha=options.alpha;
lambda=options.lambda;
beta=options.beta;
T=options.T;
sigma=options.sigma;
delta=options.delta;
[X,ns,nt,~,m,C] = datasetMsg(Xs,Ys,Xt);
% kernel
if ~strcmpi(options.kernel_type,'primal')
    %COIL   0.01 rbf
    X = kernelProject(options.kernel_type,X,[],options.kernel_param);
    Xs=X(:,1:ns);
    Xt=X(:,ns+1:end);
    m=ns+nt;
end
XssX=Xs*Xs';
hotYs=hotmatrix(Ys,C)';
% compute L according to (8)
manifold.k = k;
manifold.Metric = 'Cosine';
manifold.WeightMode = 'Cosine';
manifold.NeighborMode = 'KNN';
[L,~] = computeL(X,manifold);
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
knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
Ytpesudo = knn_model.predict((Xt)');
for i=1:T
    % compute Mc in (13)
    N=0;
    if ~isempty(Ytpesudo)
        N=conditionalDistribution(Xs,Xt,Ys,Ytpesudo,C);
        N=N/norm(N,'fro');
    end
    % compute P according to (18)
    V1=hotYs+B.*M;
    V2=sigma*L+delta*N;
    V3=Xt-Xs*Z;
    V4=E-Y1/u;
    left=XssX+X*V2*X'+u*(V3*V3')+2*alpha*eye(m);
    cons=Xs*V1'+u*V3*V4';
    P=left\cons;
    % compute Z according to (20)
    V5=P'*Xt-E+Y1/u;
    V6=Z1-Y2/u;
    V7=Z2-Y3/u;
    left=Xs'*(P*P')*Xs+ 2*eye(ns);
    cons=Xs'*P*V5+V6+V7;
    Z=left\cons;
    % compute E according to (22)
    E=shrink(P'*Xt-P'*Xs*Z+Y1/u,beta/u);
    % compute Z1 according to (24)
    Z1=SVT(Z+Y2/u, 1/u);
    % compute Z2 according to (26)
    Z2=shrink(Z+Y3/u,lambda/u);
    % compute M according to (29)
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
%     % update L
%     [L,~] = computeL(P'*X,manifold);
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

