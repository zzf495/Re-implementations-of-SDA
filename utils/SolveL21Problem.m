function [E] = SolveL21Problem(Q,alpha)
%% problem
%%%         \min\limits_{E} \frac{1}{\eta}\|E\|_{2,1}+\frac{1}{2}\|E-Q\|_F^2
%%% solve:
%%%         [E^*]_{:,i}=
%%%                     \begin{cases}   
%%%                     \large \frac{\|[Q]_{:,i}\|_2-\alpha }{ \|[Q]_{:,i}\|_2}Q_{:,i} 
%%%                                                 & \text{if } \|[Q]_{:,i}\|_2>\alpha; 
%%%                     \\ 0, 
%%%                                                 &\text{otherwise}  
%%%                     \end{cases}
 %% input
 %%%    Q:          A tractable matrix with m*n
 %%%    alpha:      The hyperparameter
 %% output
 %%%    E:          The pursued matrix
    [m,n]=size(Q);
    E=zeros(m,n);
    sumSqrtQ=sqrt(sum(Q.*Q,1));
    flag=sumSqrtQ>alpha;
    if sum(flag)>0
        score=sumSqrtQ(flag);
        score=(score-alpha)./(score);
        score=repmat(score,m,1);
        E(:,flag)=score.*Q(:,flag);
    end
end
