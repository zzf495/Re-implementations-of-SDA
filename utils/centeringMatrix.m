function [H] = centeringMatrix(n)
    H=eye(n)-1/n*ones(n,n);
end

