function [acc,acc_ite]= DGB_DA(Xs,Ys,Xt,Yt,options)
%% Implementation of DGB-DA
%%% Authors: Du et al.
%%% Paper: 2022-Dynamic-graph-based Unsupervised Domain Adaptation
%% input
%%% T              iteration times
%%% subsetRate     The subset selected of pseudo labels 
%%%                                as input in 1st iteration
%%% p              The neighbor numbers of lapgraph
%%% alpha          The weight of Label Propagation (LP)
%%% delta          The value of delta in (9)
%% output
%%% acc:        the classification accuracy (number,0~1)
%%% acc_ite:    the classification accuracy in each iteration (list)
acc_ite=[];
options=defaultOptions(options,...
            'T',10,...
            'subsetRate',0.8,...
            'p',15,...
            'alpha',0.9,...
            'delta',0.2);
subsetRate=options.subsetRate;
alpha=options.alpha;
delta=options.delta;
p=options.p;
T=options.T;
Xs=normr(Xs')';
Xt=normr(Xt')';
[~,ns,nt,n,~,C] = datasetMsg(Xs,Ys,Xt);
Y=hotmatrix([Ys;zeros(nt,1)],C);
Ytpseudo=classifyKNN(Xs,Ys,Xt,1); % use to calculate acc
selectiveYtpseudo=zeros(nt,1); % use as the selective pseudo labels during iterations

%% let selectiveYtpseudo be the subset of the target pseudo labels as shown in Alg.2
idx=randperm(nt,floor(subsetRate*nt));
selectiveYtpseudo(idx)=Ytpseudo(idx);
for iter = 1:T
    % construct W_{ss},W_{tt} by Eq.(2)
    opt.Metric='Cosine';
    opt.NeighborMode='KNN';
    opt.WeightMode='Cosine';
    opt.k=p;
    Wss=lapgraph(Xs',opt);
    Wtt=lapgraph(Xt',opt);
    % construct W_{st} by Eq.(3)
    hotYs=hotmatrix(Ys,C);
    if iter==1
        hotYt=hotmatrix(selectiveYtpseudo,C);
    else
        hotYt=predictedMultipleY;
    end
    Wst=hotYs*hotYt';% ns*nt
    W=[Wss,Wst;Wst',Wtt];
    % solve S by Eq.(6)
    Dw = diag(sparse(sqrt(1 ./ sum(W))));
    S= Dw * W * Dw;
    % solve F by Eq.(8)
    F=(1-alpha)*(eye(n,n)-alpha*S)\Y;
    softF=F./repmat( sum(F,2),1,C);
    softFt=softF(ns+1:end,1:C);
    %% distance to class means
    [highestProb,Ytpseudo]=max(softFt,[],2);
    probDelta=repmat(highestProb,1,C)-softFt;
    predictedMultipleY=probDelta<=delta;
    %% SPL
    selelctPercet=iter/T;
    rate=(1-selelctPercet);   %*(1-options.selective);
    [sortedProb,index] = sort(highestProb);
    sortedPredLabels = Ytpseudo(index);
    trustable = zeros(nt,1);
    for i = 1:C
        thisClassProb = sortedProb(sortedPredLabels==i);
        if ~isempty(thisClassProb)
            highestProbInClass=thisClassProb(floor(length(thisClassProb)*rate)+1);
            trustable = trustable + (highestProb>highestProbInClass).*(Ytpseudo==i);
        end
    end
    predictedMultipleY=predictedMultipleY.*repmat(trustable,1,C);
    acc=getAcc(Ytpseudo,Yt);
    acc_ite(iter)=acc;
    if isfield(options,'display')
        fprintf('[%2d] acc:%.4f\n',iter,acc);
    end
end
end
