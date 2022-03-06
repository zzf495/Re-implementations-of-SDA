function [acc,acc_ite] = ATL(Xs,Ys,Xt,Yt,options)
%% Implementation of ATL
%%% Authors:        Peng et al.
%%% Paper:          2020-Active Transfer Learning
%% input
%%% T:       iteration times
%%% u:       ADMM increasing parameter
%%% rho:     ADMM incrasing rate
%%% convergenceValue    the cut-off value of ADMM
%%% gamma    kernel parameter
%%% alpha      The weight of K (Eq. (5))
%%% beta      The weight of W  (Eq. (6))
%%% eta      The weight of regularization for projection matrix
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
% Xs = normr(Xs')';
% Xt = normr(Xt')';

acc_ite=[];
residualY1=[];
residualY2=[];
options=defaultOptions(options,...
                            'T',10,...
                            'u',0.1,...
                            'rho',1.01,...
                            'convergenceValue',1e-6,...
                            'alpha',1,...
                            'beta',1,...
                            'eta',0.1);
convergenceValue=options.convergenceValue;
u=options.u;
rho=options.rho;
alpha=options.alpha;
beta=options.beta;
eta=options.eta;
T=options.T;
Xt=Xt./size(Xt,2);
[X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt);
% init parameter
dim=floor(m*0.75);
Y1=0;
Y2=zeros(ns,1);
v=zeros(ns,1);
manifold.k = 5;
manifold.Metric = 'Cosine';
manifold.NeighborMode = 'Supervised';
manifold.WeightMode = 'Binary';
manifold.gnd=Ys;
W=lapgraph(Xs',manifold);
% init K
K=0;
for i=1:C
    e=zeros(ns,1);
    e(Ys==i)=1;
    K=K+e*e';
end
P=ones(m,dim);
XHX=X*centeringMatrix(n)*X';
for i=1:T
    % compute A according to (13)
    left=2/(ns*ns)*(Xs'*(P*P')*Xs+alpha*(K+K')+beta*(W+W')+2*u*ones(ns,ns));
    right=2/(ns*nt)*(Xs'*(P*P')*Xt*ones(nt,1) + u* ones(ns,1) - ones(ns,1)*Y1 +u*v-Y2);
    a=left\right; 
    % compute P according to (16)
    M=1/(ns*ns)*Xs*(a*a')*Xs'-2/(ns*nt)*Xs*a*ones(1,nt)*Xt'+...
        1/(nt*nt)*Xt*ones(nt,nt)*Xt'+eta*eye(m);
    M=M/norm(M,'fro');
    [P,~]=eigs(M,XHX,dim,'sm');
    % compute v according to (19)
    v=max(0,a+1/u *Y2);
    % update multipliers according to (20)
    Y1=Y1+u*(a'*ones(ns,1)- 1);
    Y2=Y2+u*(a-v);
    u=min(rho*u,1e6);
    % predict
    Zk=P'*X;
    Zs=(Zk(:,1:ns)); %.*(ones(dim,1)*a');
    Zt=(Zk(:,ns+1:end));
    Ytpesudo=classifyKNN(Zs,Ys,Zt,1);
    acc=getAcc(Ytpesudo,Yt);
    acc_ite=[acc_ite,acc];
    % residual
    residualY1=[residualY1,norm((a'*ones(ns,1)- 1),'inf')];
    residualY2=[residualY2,norm(a-v,'inf')];
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f ,r1: %.6f,r2: %.6f\n',i,acc,residualY1(end),residualY2(end));
    end
    if (residualY1(end)<=convergenceValue)&&residualY2(end)<=convergenceValue
        if isfield(options,'display')
            fprintf('function convergenceValue,break \n');
        end
        break;
    elseif i>1
        if (abs((residualY1(end)-residualY1(end-1)))<=convergenceValue)&&(abs(residualY2(end)-residualY2(end-1)))<=convergenceValue
            if isfield(options,'display')
                fprintf('function convergenceValue,break \n');
            end
            break;
        end
    end
end
end

