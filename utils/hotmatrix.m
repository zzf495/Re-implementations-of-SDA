function [matrix] = hotmatrix(labels,C,weight)
%% input:
%%% labels: n*1, the labels of samples
%%% C: integer, the number of the classes
%%% weight: integer, if weight==1, then the value is 1/length(classes)
%% output:
%%% matrix: n*C, the output hotmatrix
    if nargin==2
       weight=0; % weight =0,then Y={0,1} , weight = 1, then Y={0,1/n^c} 
    end
    n=length(labels);
    matrix=zeros(n,C);
    weightY=zeros(C,1);
    for i=1:C
        if weight==0
            weightY(i)=1;
        else
            weightY(i)=1/length(find(labels==i));
        end
    end
    for i=1:n
        if(labels(i)>0 &&labels(i)<=C)
            matrix(i,labels(i))=weightY(labels(i));
        end
    end
    % other implementation
%     full(sparse(1:ns,Ys,1));
end

