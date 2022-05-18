function [acc,acc_ite] = WCS_RAR(Xs,Ys,Xt,Yt,options)
%% Implementation of WCS-RAR
%%% Authors: Yang et al.
%%% Paper: 2020-Robust adaptation regularization based on within-class scatter for domain adaptation
%% input
%%% T:       iteration times
%%% u:       ADMM increasing parameter
%%% rho:     ADMM incrasing rate
%%% convergenceValue    the cut-off value of ADMM
%%% alpha    The weight of S
%%% beta     The weight of MMD M
%%% lambda   The weight of manifold regularization L
%%% mu       The weight of M0+Mc (MMD)
%%% kernel_type: The kernel mapping, e.g., 'primal' (skip kernel mapping),'linear', 'rbf', and 'sam'
%%% kernel_param: The weight of kernel (sigma in kernel project)
%%% eta      The regularization term
%%% k:       k-neighbors of graph
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)

%%%	======================Version=======================
%%%		V1			2022-03-07
%%% 	V2			2022-05-18
acc_ite=[];
options=defaultOptions(options,...
    'T',20,...
    'r',30,...
    'kernel_type','rbf',...
    'u',0.1,...
    'rho',1.01,...
    'convergenceValue',1e-6,...
    'k',7,...
    'mu',0.5,...
    'alpha',1,...
    'beta',1,...
    'lambda',1,...
    'eta',0.1);
convergenceValue=options.convergenceValue;
u=options.u;
rho=options.rho;
alpha=options.alpha;
beta=options.beta;
eta=options.eta;
lambda=options.lambda;
mu=options.mu;
T=options.T;
[X,ns,nt,n,~,C] = datasetMsg(Xs,Ys,Xt);
% Manifold feature learning
[Xs_new,Xt_new,~] = WCS_RAR_GFK_Map(Xs',Xt',options.r);
Xs = double(Xs_new');
Xt = double(Xt_new');
[X,ns,nt,n,~,C] = datasetMsg(Xs,Ys,Xt);
% kernel
X = kernelProject('rbf',X,[],sqrt(sum(sum(X .^ 2).^0.5)/(ns + nt)));
Xs=X(:,1:ns);
Xt=X(:,ns+1:end);
hotYs=hotmatrix(Ys,C);
hotYs=[hotYs;zeros(nt,C)];
A=blkdiag(eye(ns),zeros(nt));
% init parameter
E=zeros(n,C);
Y1=zeros(n,C);
% compute marginal M0 accroding to (22)
M0=marginalDistribution(Xs,Xt,C);
% compute manifold L accroding to (11),(24)
manifold.k = options.k;
manifold.Metric = 'Cosine';
manifold.WeightMode = 'Cosine';
manifold.NeighborMode = 'KNN';
[L,~] = computeL(X,manifold);
L=L/norm(L,'fro');
N=0;
% compute S according to (19)
S=[];
for i=1:C
    cNum=length(find(Ys==i));
    S=blkdiag(S, eye(cNum)-1/cNum*ones(cNum));
end
S=blkdiag(S,zeros(nt));
S=S/norm(S,'fro');
Ytpesudo=[];
for i=1:T
    % compute P according to (29)
    if ~isempty(Ytpesudo)
       % compute conditional Mc accroding to (23)
       N=conditionalDistribution(Xs,Xt,Ys,Ytpesudo,C); 
    end
    M=(1-mu)*M0+mu*N;
    M=M/norm(M,'fro');
    V0=A*hotYs+E-Y1/u;
    left=(u/2*(A'*A)+alpha*S+beta*M+lambda*L)*X+eta*eye(n);
    cons=u/2*A'*V0;
    P=left\cons;
    % compute E according (31)
    Q=A*(X*P-hotYs)+Y1/u;
 %   sqrtQ=diag(Q*Q');
 %   E=zeros(n,C);
 %   idx=sqrtQ>(1/u);
 %   E(idx,:)=((sqrtQ(idx)-1/u)/sqrtQ(idx))*Q(idx,:);
 	E=SolveL21Problem(Q,1/u);
    % update multiplier
    Y1=Y1+u*(A*(X*P-hotYs)-E);
    u=min(rho*u,1e7);
    % predict
    Zt=P'*Xt;
    [~,Ytpesudo] = max(Zt',[],2);
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    % compute the residual
    residualY1=norm(A*(X*P-hotYs)-E,'inf');
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f ,r1: %.6f\n',i,acc,residualY1);
    end
    if residualY1<=convergenceValue
        if isfield(options,'display')
            fprintf('function convergenceValue,break \n');
        end
        break;
    end
end
end

